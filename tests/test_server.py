import asyncio
import hashlib
import io
import os
import time

import cv2
import numpy as np
import pytest

from faceforge_eval import models as M
from faceforge_server import protocol as P
from faceforge_server.server import OffloadServer, SwapEngine


# ---------------------------------------------------------------- protocol --

def test_header_validation():
    good = P.encode(P.HELLO, b"{}")
    assert P.decode_header(good[:12]) == (P.HELLO, 2)
    with pytest.raises(P.ProtocolError):
        P.decode_header(b"XXXX" + good[4:12])
    with pytest.raises(P.ProtocolError):
        P.decode_header(good[:4] + bytes([99]) + good[5:12])
    with pytest.raises(P.ProtocolError):
        P.decode_header(P.HEADER.pack(P.MAGIC, P.FRAME, 0, 0, P.MAX_PAYLOAD + 1))
    with pytest.raises(P.ProtocolError):
        P.encode(42)
    with pytest.raises(P.ProtocolError):
        P.unpack_frame(b"\x00" * 5)
    with pytest.raises(P.ProtocolError):
        P.parse_json(b"[1,2]")


# ------------------------------------------------------------ server logic --

class StubEngine:
    provider, device = "CPUExecutionProvider", ""

    def __init__(self):
        self.source = None
        self.settings = {}

    def set_source(self, jpeg):
        self.source = jpeg
        return {"ok": True, "faces": 1, "message": ""}

    def apply_settings(self, s):
        self.settings.update(s)

    def process(self, frame):
        return P.Result(frame.frame_id, frame.client_time_us, 1.0, self.source is not None, frame.jpeg[::-1])


async def _client(port, msgs, read=1):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    for m in msgs:
        writer.write(m)
    await writer.drain()
    out = [await asyncio.wait_for(P.read_message(reader), 5) for _ in range(read)]
    return reader, writer, out


def run_with_server(engine, coro_fn):
    async def main():
        srv_obj = OffloadServer(engine, "246810", printer=lambda s: None)
        srv = await asyncio.start_server(srv_obj.handle, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        try:
            return await coro_fn(port)
        finally:
            srv.close()
            await srv.wait_closed()
    return asyncio.run(main())


def hello(token="246810"):
    return P.encode_json(P.HELLO, {"protocol": 1, "token": token, "client": "test"})


def test_wrong_token_rejected():
    async def go(port):
        _, w, out = await _client(port, [hello("000000")])
        w.close()
        return out
    (t, payload), = run_with_server(StubEngine(), go)
    assert t == P.ERROR and "pairing" in P.parse_json(payload)["message"]


def test_must_start_with_hello():
    async def go(port):
        _, w, out = await _client(port, [P.pack_frame(P.Frame(1, 1, b"x"))])
        w.close()
        return out
    (t, _), = run_with_server(StubEngine(), go)
    assert t == P.ERROR


def test_session_flow_and_ordering():
    eng = StubEngine()

    async def go(port):
        r, w, out = await _client(port, [hello()])
        assert out[0][0] == P.WELCOME
        # Frame before source: passed through, face_found False
        w.write(P.pack_frame(P.Frame(1, 111, b"abc")))
        w.write(P.encode(P.SOURCE, b"src"))
        w.write(P.encode_json(P.SETTINGS, {"jpeg_quality": 70}))
        for i in range(2, 5):
            w.write(P.pack_frame(P.Frame(i, 100 + i, b"xyz")))
        await w.drain()
        msgs = [await asyncio.wait_for(P.read_message(r), 5) for _ in range(5)]
        w.close()
        return msgs
    msgs = run_with_server(eng, go)
    first = P.unpack_result(msgs[0][1])
    assert msgs[0][0] == P.RESULT and first.frame_id == 1 and not first.face_found and first.jpeg == b"cba"
    assert msgs[1][0] == P.SOURCE_ACK and P.parse_json(msgs[1][1])["ok"]
    ids = [P.unpack_result(p).frame_id for t, p in msgs[2:]]
    assert ids == [2, 3, 4]
    assert all(P.unpack_result(p).face_found for t, p in msgs[2:])
    assert eng.settings == {"jpeg_quality": 70}


def test_second_client_gets_busy():
    async def go(port):
        r1, w1, out1 = await _client(port, [hello()])
        _, w2, out2 = await _client(port, [hello()])
        w1.close()
        w2.close()
        return out1, out2
    out1, out2 = run_with_server(StubEngine(), go)
    assert out1[0][0] == P.WELCOME
    assert out2[0][0] == P.ERROR and "busy" in P.parse_json(out2[0][1])["message"]


# ---------------------------------------------------------------- models ----

class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_download_verifies_size_and_hash(tmp_path):
    data = os.urandom(3000)
    good = M.Asset("x.bin", "https://example/x", len(data), hashlib.sha256(data).hexdigest(), "test")
    dest = tmp_path / "x.bin"
    M.download(good, str(dest), opener=lambda url, timeout: FakeResp(data))
    assert dest.read_bytes() == data
    bad = M.Asset("y.bin", "https://example/y", len(data), "0" * 64, "test")
    with pytest.raises(M.DownloadError):
        M.download(bad, str(tmp_path / "y.bin"), opener=lambda url, timeout: FakeResp(data))
    assert not (tmp_path / "y.bin").exists() and not list(tmp_path.glob("*.part"))


def test_ensure_models_needs_licence_acceptance(tmp_path):
    calls = []
    with pytest.raises(M.DownloadError) as e:
        M.ensure_models(str(tmp_path), accept_license=False, opener=lambda *a, **k: calls.append(a))
    assert "non-commercial" in str(e.value) and not calls


def test_catalog_uses_publisher_urls():
    for a in (M.INSWAPPER, M.BUFFALO_L):
        assert a.url.startswith("https://github.com/deepinsight/insightface/releases/download/")
        assert len(a.sha256) == 64 and a.size > 1_000_000


def test_extract_buffalo_real(weights_dir, tmp_path):
    z = weights_dir / "buffalo_l.zip"
    if not z.exists():
        pytest.skip("buffalo_l.zip not in FACEFORGE_WEIGHTS")
    M.extract_buffalo(str(z), str(tmp_path / "buffalo_l"))
    for name, (size, sha) in M.BUFFALO_MEMBERS.items():
        assert M.verify(str(tmp_path / "buffalo_l" / name), size, sha)


# ------------------------------------------------- end to end, real models --

def test_end_to_end_swap_over_tcp(weights_dir):
    img_path = os.environ.get("FACEFORGE_TEST_IMAGE")
    if not img_path:
        pytest.skip("FACEFORGE_TEST_IMAGE not set")
    from faceforge_eval.detector import SCRFDDetector
    from faceforge_eval.recognizer import ArcFaceEmbedder
    from faceforge_eval.swapper import InSwapper
    engine = SwapEngine(SCRFDDetector(str(weights_dir / "buffalo_l" / "det_10g.onnx")),
                        ArcFaceEmbedder(str(weights_dir / "buffalo_l" / "w600k_r50.onnx")),
                        InSwapper(str(weights_dir / "inswapper_128.onnx")), "CPUExecutionProvider")
    img = cv2.imread(img_path)
    faces = sorted(engine.detector.detect(img), key=lambda f: f.bbox[0])
    x1, y1, x2, y2 = faces[0].bbox.astype(int)
    pad = 40
    src = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    tx1, ty1, tx2, ty2 = faces[1].bbox.astype(int)
    target = np.ascontiguousarray(img[max(0, ty1 - 60):ty2 + 60, max(0, tx1 - 60):tx2 + 60])
    jpeg = lambda a: cv2.imencode(".jpg", a)[1].tobytes()  # noqa: E731

    async def go(port):
        r, w, out = await _client(port, [hello()])
        w.write(P.encode(P.SOURCE, jpeg(src)))
        await w.drain()
        ack = await asyncio.wait_for(P.read_message(r), 60)
        t0 = time.perf_counter()
        w.write(P.pack_frame(P.Frame(9, 42, jpeg(target))))
        await w.drain()
        res = await asyncio.wait_for(P.read_message(r), 60)
        w.close()
        return out, ack, res, time.perf_counter() - t0

    out, ack, res, rtt = run_with_server(engine, go)
    assert out[0][0] == P.WELCOME
    assert ack[0] == P.SOURCE_ACK and P.parse_json(ack[1])["ok"]
    result = P.unpack_result(res[1])
    assert result.frame_id == 9 and result.client_time_us == 42 and result.face_found
    decoded = cv2.imdecode(np.frombuffer(result.jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == target.shape
    assert np.abs(decoded.astype(int) - target.astype(int)).mean() > 1.0  # the face changed
    print(f"\n[e2e] CPU server round trip {rtt * 1000:.0f} ms, server {result.server_ms:.0f} ms")
