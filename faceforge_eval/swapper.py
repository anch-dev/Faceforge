"""Face swappers behind one interface.

Swapper.swap(frame, target_face, source_embedding) -> (aligned_fake_bgr, M)

`source_embedding` is the L2-normalised ArcFace (w600k_r50) embedding: the
identity space every embedding-conditioned swapper in this ecosystem expects
(inswapper, HyperSwap, SimSwap, GHOST via converter). Swapping in a different
recognizer is not a drop-in change.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnx
from onnx import numpy_helper

from .geometry import norm_crop
from .runtime import create_session
from .types import Face


class InSwapper:
    """InsightFace inswapper_128 (fp32 or locally converted fp16)."""

    def __init__(self, model_path: str, provider: str = "cpu", allow_cpu_fallback=False):
        graph = onnx.load(model_path, load_external_data=False).graph
        self.emap = numpy_helper.to_array(graph.initializer[-1]).astype(np.float32)
        if self.emap.shape != (512, 512):
            raise ValueError(f"{model_path}: last initializer is {self.emap.shape}, expected emap 512x512")
        self.session = create_session(model_path, provider, allow_cpu_fallback)
        ins = self.session.get_inputs()
        self.target_name, self.source_name = ins[0].name, ins[1].name
        self.size = int(ins[0].shape[2])

    def latent(self, source_embedding: np.ndarray) -> np.ndarray:
        lat = source_embedding.reshape(1, -1).astype(np.float32) @ self.emap
        return lat / np.linalg.norm(lat)

    def swap(self, frame: np.ndarray, target: Face, source_embedding: np.ndarray):
        aligned, M = norm_crop(frame, target.kps, self.size)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255.0, (self.size, self.size),
                                     (0.0, 0.0, 0.0), swapRB=True)
        pred = self.session.run(None, {self.target_name: blob,
                                       self.source_name: self.latent(source_embedding)})[0]
        fake = np.clip(pred[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
        return fake, M


class HyperSwap:
    """FaceFusion HyperSwap 1a/1b/1c (256px). Interface per FaceFusion 3.9.1:
    inputs 'source' (1,512 normalised ArcFace) and 'target' (1,3,256,256,
    RGB in [-1,1]); outputs 'output' (same range) and 'mask'.

    UNVERIFIED in this repo: the weights could not be retrieved in the audit
    sandbox, so this adapter has only been exercised against synthetic models."""

    def __init__(self, model_path: str, provider: str = "cpu", allow_cpu_fallback=False):
        self.session = create_session(model_path, provider, allow_cpu_fallback)
        names = {i.name: i for i in self.session.get_inputs()}
        self.size = int(names["target"].shape[2])

    def swap(self, frame: np.ndarray, target: Face, source_embedding: np.ndarray):
        # FaceFusion's 'arcface_128' template x 256 == InsightFace 128-convention
        # template scaled to 256, which estimate_norm(kps, 256) produces.
        aligned, M = norm_crop(frame, target.kps, self.size)
        x =(aligned[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]
        out = self.session.run(None, {"source": source_embedding.reshape(1, -1).astype(np.float32),
                                      "target": x})[0][0]
        out = (out.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
        fake = (out[:, :, ::-1] * 255).astype(np.uint8)
        return fake, M
