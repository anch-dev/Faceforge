"""Generate cross-language golden fixtures (tests/fixtures/).

Python is the verified reference (bit-identical to insightface 0.7.3 on real
weights); the Swift package ios/FaceforgeCore must reproduce these outputs.
tests/test_fixtures.py regenerates them and fails if the committed copies drift.

    python tools/make_fixtures.py [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from faceforge_eval.detector import decode, nms  # noqa: E402
from faceforge_eval.geometry import ARCFACE_DST, estimate_norm  # noqa: E402
from faceforge_server import protocol as P  # noqa: E402


def r(x, n=6):
    return np.round(np.asarray(x, dtype=np.float64), n).tolist()


def alignment_fixture():
    rng = np.random.default_rng(42)
    cases = []
    # Real-ish faces: template under rotation/scale/translation plus noise.
    for i in range(6):
        theta, s = rng.uniform(-0.6, 0.6), rng.uniform(0.5, 4.0)
        R = s * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        kps = ARCFACE_DST @ R.T + rng.uniform(0, 600, 2) + rng.normal(0, 1.5, (5, 2))
        kps = np.round(kps, 3)
        cases.append({"kps": r(kps, 3),
                      "M112": r(estimate_norm(kps, 112)),
                      "M128": r(estimate_norm(kps, 128)),
                      "M256": r(estimate_norm(kps, 256))})
    return {"arcface_dst": r(ARCFACE_DST, 4), "cases": cases}


def scrfd_fixture():
    rng = np.random.default_rng(7)
    size = 64                          # grids 8x8, 4x4, 2x2; 2 anchors each
    counts = [(size // s) ** 2 * 2 for s in (8, 16, 32)]
    scores = [rng.uniform(0, 0.45, (n, 1)).astype(np.float32) for n in counts]
    bboxes = [rng.uniform(0.5, 2.5, (n, 4)).astype(np.float32) for n in counts]
    kps = [rng.uniform(-1.5, 1.5, (n, 10)).astype(np.float32) for n in counts]
    # A few confident detections, two of them overlapping (NMS must drop one).
    for lvl, idx, sc in [(0, 27, 0.93), (0, 29, 0.88), (1, 10, 0.71), (2, 3, 0.66)]:
        scores[lvl][idx] = sc
    # Round first so the values written to JSON are exactly what was decoded.
    outs = [np.round(o, 6).astype(np.float32) for o in scores + bboxes + kps]
    det_scale = 0.4                    # model-input px per frame px
    s, b, k = decode(outs, (size, size), threshold=0.5)
    b, k = b / det_scale, k / det_scale
    order = s.argsort()[::-1]
    dets = np.hstack([b, s[:, None]])[order]
    k = k[order]
    keep = nms(dets, 0.4)
    return {
        "input_size": size, "strides": [8, 16, 32], "num_anchors": 2,
        "threshold": 0.5, "nms_threshold": 0.4, "det_scale": det_scale,
        "outputs": [r(o.reshape(-1), 6) for o in outs],
        "expected": [{"bbox": r(dets[i, :4], 3), "score": round(float(dets[i, 4]), 6),
                      "kps": r(k[i], 3)} for i in keep],
    }


def onnx_fixture(out_dir):
    """Tiny model whose LAST initializer is an unused 8x8 matrix, like
    inswapper's emap. Swift's ONNX initializer reader must find it."""
    rng = np.random.default_rng(3)
    w = numpy_helper.from_array(rng.standard_normal((2, 3, 1, 1)).astype(np.float32), "w")
    emap = rng.standard_normal((8, 8)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"])], "emap_fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])],
        [w, numpy_helper.from_array(emap, "emap")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="faceforge-fixture")
    model.ir_version = 8
    path = os.path.join(out_dir, "emap_fixture.onnx")
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    return {"file": "emap_fixture.onnx", "name": "emap", "dims": [8, 8],
            "values": [float(v) for v in emap.reshape(-1)]}


def protocol_fixture():
    hello = {"client": "ios", "protocol": 1, "token": "123456"}
    jpeg = bytes.fromhex("ffd8ffe000104a46494600ffd9")
    return {
        "hello": {"json": hello, "hex": P.encode_json(P.HELLO, hello).hex()},
        "frame": {"frame_id": 7, "client_time_us": 1234567890123, "jpeg_hex": jpeg.hex(),
                  "hex": P.pack_frame(P.Frame(7, 1234567890123, jpeg)).hex()},
        "result": {"frame_id": 7, "client_time_us": 1234567890123, "server_ms": 12.5, "face_found": True,
                   "jpeg_hex": jpeg.hex(),
                   "hex": P.pack_result(P.Result(7, 1234567890123, 12.5, True, jpeg)).hex()},
        "error": {"json": {"message": "wrong pairing code"},
                  "hex": P.encode_json(P.ERROR, {"message": "wrong pairing code"}).hex()},
    }


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "alignment.json": alignment_fixture(),
        "scrfd_decode.json": scrfd_fixture(),
        "onnx_initializer.json": onnx_fixture(out_dir),
        "protocol_v1.json": protocol_fixture(),
    }
    for name, obj in data.items():
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")
    return sorted(list(data) + ["emap_fixture.onnx"])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "tests", "fixtures")
    print(generate(out))
