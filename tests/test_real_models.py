"""Optional tests against real, user-obtained weights.

Run with FACEFORGE_WEIGHTS pointing at a folder containing
inswapper_128.onnx and buffalo_l/{det_10g,w600k_r50}.onnx, and
FACEFORGE_TEST_IMAGE at a photo with at least two faces. Skipped otherwise
(CI has no weights: Faceforge does not redistribute them).
"""
import os

import cv2
import numpy as np
import pytest

import faceforge_runtime as ffr
from faceforge_eval.detector import SCRFDDetector
from faceforge_eval.recognizer import ArcFaceEmbedder, cosine
from faceforge_eval.swapper import InSwapper


def need(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    return str(path)


def test_inswapper_fp16_fidelity(weights_dir, tmp_path):
    src = need(weights_dir / "inswapper_128.onnx")
    dst = tmp_path / "inswapper_128_fp16.onnx"
    assert ffr.convert_fp16(src, str(dst)) == ["buff2fs"]
    sw32, sw16 = InSwapper(src), InSwapper(str(dst))
    rng = np.random.default_rng(7)
    tgt = rng.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32)
    emb = rng.standard_normal(512).astype(np.float32)
    lat = sw32.latent(emb / np.linalg.norm(emb))
    a =sw32.session.run(None, {"target": tgt, "source": lat})[0]
    b = sw16.session.run(None, {"target": tgt, "source": lat})[0]
    cos = float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
    assert not np.isnan(b).any()
    assert cos > 0.9999
    assert np.abs(a - b).mean() * 255 < 1.0


def test_swap_changes_identity(weights_dir):
    img_path = os.environ.get("FACEFORGE_TEST_IMAGE")
    if not img_path:
        pytest.skip("FACEFORGE_TEST_IMAGE not set")
    det = SCRFDDetector(need(weights_dir / "buffalo_l" / "det_10g.onnx"))
    rec = ArcFaceEmbedder(need(weights_dir / "buffalo_l" / "w600k_r50.onnx"))
    sw = InSwapper(need(weights_dir / "inswapper_128.onnx"))
    img = cv2.imread(img_path)
    faces = det.detect(img)
    assert len(faces) >= 2
    src = rec.embed(img, faces[0])
    fake, _ = sw.swap(img, faces[1], src)
    # The 128 crop is the 112 ArcFace template shifted +8 px in x, so the
    # ArcFace-aligned view of the swapped face is a plain slice.
    out_emb = rec.embed_aligned(np.ascontiguousarray(fake[0:112, 8:120]))
    assert cosine(out_emb, src) > 0.6 > cosine(rec.embed(img, faces[1]), src)
