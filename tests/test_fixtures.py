"""Golden fixtures shared with the Swift tests stay in sync with the Python reference."""
import json

import numpy as np
import onnx
from onnx import numpy_helper

from conftest import REPO

FIX = REPO / "tests" / "fixtures"


def test_committed_fixtures_match_generator(tmp_path):
    from make_fixtures import generate
    names = generate(str(tmp_path))
    for name in names:
        fresh, committed = (tmp_path / name).read_bytes(), (FIX / name).read_bytes()
        assert fresh == committed, f"{name} is stale: run python tools/make_fixtures.py"


def test_scrfd_fixture_has_overlap_suppressed():
    d = json.loads((FIX / "scrfd_decode.json").read_text())
    assert len(d["expected"]) == 3  # 4 confident anchors, one removed by NMS
    scores = [e["score"] for e in d["expected"]]
    assert scores == sorted(scores, reverse=True)


def test_onnx_fixture_last_initializer_is_emap():
    meta = json.loads((FIX / "onnx_initializer.json").read_text())
    g = onnx.load(str(FIX / meta["file"])).graph
    last = g.initializer[-1]
    assert last.name == meta["name"]
    assert np.allclose(numpy_helper.to_array(last).reshape(-1), meta["values"])
    assert all(last.name not in n.input for n in g.node)


def test_protocol_fixture_roundtrip():
    from faceforge_server import protocol as P
    d = json.loads((FIX / "protocol_v1.json").read_text())
    frame = P.unpack_frame(bytes.fromhex(d["frame"]["hex"])[P.HEADER.size:])
    assert (frame.frame_id, frame.client_time_us) == (7, 1234567890123)
    res = P.unpack_result(bytes.fromhex(d["result"]["hex"])[P.HEADER.size:])
    assert res.server_ms == 12.5 and res.face_found and res.jpeg.hex() == d["result"]["jpeg_hex"]
    assert P.decode_header(bytes.fromhex(d["hello"]["hex"])[:12]) == (P.HELLO, len(json.dumps(
        d["hello"]["json"], separators=(",", ":"), sort_keys=True)))
