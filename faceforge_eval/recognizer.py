"""ArcFace-style identity embedder (e.g. w600k_r50.onnx, glintr100.onnx)."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import norm_crop
from .runtime import create_session
from .types import Face


class ArcFaceEmbedder:
    input_mean = 127.5
    input_std = 127.5

    def __init__(self, model_path: str, provider: str = "cpu", allow_cpu_fallback=False):
        self.session = create_session(model_path, provider, allow_cpu_fallback)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 112

    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1.0 / self.input_std, (self.size, self.size),
                                     (self.input_mean,) * 3, swapRB=True)
        feat = self.session.run(None, {self.input_name: blob})[0].reshape(-1)
        return feat / np.linalg.norm(feat)

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        crop, _ = norm_crop(frame, face.kps, self.size)
        return self.embed_aligned(crop)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
