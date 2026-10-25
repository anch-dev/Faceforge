import numpy as np
import onnxruntime as ort
import pytest

from faceforge_eval.compositor import Compositor
from faceforge_eval.detector import decode, nms
from faceforge_eval.geometry import ARCFACE_DST, estimate_norm, norm_crop, umeyama
from faceforge_eval.pipeline import Pipeline, summarize
from faceforge_eval.runtime import BackendUnavailable, create_session
from faceforge_eval.swapper import InSwapper
from faceforge_eval.types import Face


def similarity(scale, theta, tx, ty):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty]])


def test_umeyama_recovers_similarity():
    M = similarity(1.7, 0.3, 12.0, -5.0)
    src = ARCFACE_DST.astype(np.float64)
    dst = src @ M[:, :2].T + M[:, 2]
    est = umeyama(src, dst)[:2]
    assert np.allclose(est, M, atol=1e-6)


def test_estimate_norm_conventions():
    # Landmarks already on the 112 template -> identity.
    assert np.allclose(estimate_norm(ARCFACE_DST, 112), [[1, 0, 0], [0, 1, 0]], atol=1e-5)
    # 128 convention: same scale, +8 px x shift (InsightFace / inswapper training crop).
    assert np.allclose(estimate_norm(ARCFACE_DST, 128), [[1, 0, 8], [0, 1, 0]], atol=1e-5)
    # 256 == FaceFusion 'arcface_128' template x 256 (used by HyperSwap).
    ff_arcface_128 = np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469],
                               [0.50019687, 0.56044219], [0.38710391, 0.72160547],
                               [0.61507734, 0.72034453]]) * 256
    M = estimate_norm(ARCFACE_DST, 256)
    mapped = ARCFACE_DST @ M[:, :2].T + M[:, 2]
    assert np.abs(mapped - ff_arcface_128).max() < 0.01
    with pytest.raises(ValueError):
        estimate_norm(ARCFACE_DST, 100)


def test_scrfd_decode_single_anchor_hit():
    # 32x32 input -> strides 8/16/32 give 4x4, 2x2, 1x1 grids, 2 anchors each.
    sizes = [32, 8, 2]
    scores = [np.zeros((n, 1), np.float32) for n in sizes]
    bboxes = [np.zeros((n, 4), np.float32) for n in sizes]
    kps = [np.zeros((n, 10), np.float32) for n in sizes]
    # Anchor index 10 at stride 8: grid cell 5 -> (x=1, y=1) -> centre (8, 8).
    scores[0][10] = 0.9
    bboxes[0][10] = [0.5, 0.5, 0.5, 0.5]           # x stride 8 -> 4 px each side
    kps[0][10] = np.tile([0.25, -0.25], 5)         # x stride 8 -> (+2, -2)
    s, b, k = decode(scores + bboxes + kps, (32, 32), threshold=0.5)
    assert s.shape == (1,)
    assert np.allclose(b[0], [4, 4, 12, 12])
    assert np.allclose(k[0], np.tile([10, 6], (5, 1)))


def test_nms():
    dets = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], np.float32)
    assert nms(dets, 0.4) == [0, 2]


def test_create_session_refuses_silent_fallback(tiny_conv):
    assert create_session(tiny_conv, "cpu").get_providers()[0] == "CPUExecutionProvider"
    missing = next(p for p in ("dml", "cuda", "openvino", "coreml")
                   if {"dml": "DmlExecutionProvider", "cuda": "CUDAExecutionProvider",
                       "openvino": "OpenVINOExecutionProvider",
                       "coreml": "CoreMLExecutionProvider"}[p] not in ort.get_available_providers())
    with pytest.raises(BackendUnavailable):
        create_session(tiny_conv, missing)
    assert create_session(tiny_conv, missing, allow_cpu_fallback=True).get_providers()[0] == "CPUExecutionProvider"


def test_summarize_one_percent_low():
    frame_ms = [10.0] * 99 + [100.0]
    s = summarize(frame_ms)
    assert s["frames"] == 100
    assert s["one_pct_low_fps"] == 10.0
    assert abs(s["avg_fps"] - 1000 * 100 / (990 + 100)) < 0.01


class FixedDetector:
    def __init__(self, face):
        self.face, self.calls = face, 0

    def detect(self, frame, max_num=0):
        self.calls += 1
        return [self.face]


def test_pipeline_end_to_end_with_fake_swapper(fake_inswapper):
    frame = np.full((240, 320, 3), 128, np.uint8)
    face = Face(bbox=np.array([100, 60, 220, 200], np.float32), score=0.99,
                kps=(ARCFACE_DST * 1.2 + [100, 60]).astype(np.float32))
    det = FixedDetector(face)
    swapper = InSwapper(fake_inswapper)
    assert swapper.emap.shape == (512, 512)
    pipe = Pipeline(det, swapper, detect_every=3)
    emb = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    for _ in range(6):
        out, t = pipe.process(frame, emb)
    assert det.calls == 2  # frames 0 and 3 only
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert set(t.stages_ms) == {"swap", "composite"}
    # Only the face region changes.
    changed = np.argwhere(np.any(out != frame, axis=2))
    assert changed.size and changed[:, 0].min() >= 50 and changed[:, 1].min() >= 90


def test_compositor_noop_when_mask_off_frame():
    frame = np.zeros((50, 50, 3), np.uint8)
    fake = np.full((128, 128, 3), 255, np.uint8)
    M = np.array([[1, 0, 10000], [0, 1, 10000]], np.float64)  # face maps far outside
    assert np.array_equal(Compositor().paste(frame, fake, M), frame)


def test_norm_crop_shape():
    frame = np.zeros((200, 200, 3), np.uint8)
    crop, M = norm_crop(frame, ARCFACE_DST + 40, 112)
    assert crop.shape == (112, 112, 3) and M.shape == (2, 3)
