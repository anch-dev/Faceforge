"""SCRFD face detector (e.g. InsightFace det_10g.onnx) decode on ONNX Runtime."""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .runtime import create_session
from .types import Face


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 0] - distance[:, 0], points[:, 1] - distance[:, 1],
                     points[:, 0] + distance[:, 2], points[:, 1] + distance[:, 3]], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


def nms(dets: np.ndarray, thresh: float) -> List[int]:
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


def decode(outputs: List[np.ndarray], input_hw: Tuple[int, int], strides=(8, 16, 32),
           num_anchors: int = 2, threshold: float = 0.5):
    """Decode SCRFD heads: outputs ordered [scores x3, bboxes x3, kps x3]."""
    fmc = len(strides)
    cache: Dict[Tuple[int, int, int], np.ndarray] = {}
    scores_l, bboxes_l, kps_l = [], [], []
    ih, iw = input_hw
    for idx, stride in enumerate(strides):
        scores = outputs[idx].reshape(-1)
        bbox_preds = outputs[idx + fmc].reshape(-1, 4) * stride
        kps_preds = outputs[idx + fmc * 2].reshape(-1, 10) * stride
        h, w = ih // stride, iw // stride
        key = (h, w, stride)
        if key not in cache:
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            if num_anchors > 1:
                centers = np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)
            cache[key] = centers
        centers = cache[key]
        pos = np.where(scores >= threshold)[0]
        scores_l.append(scores[pos])
        bboxes_l.append(distance2bbox(centers, bbox_preds)[pos])
        kps_l.append(distance2kps(centers, kps_preds).reshape(-1, 5, 2)[pos])
    return np.concatenate(scores_l), np.concatenate(bboxes_l), np.concatenate(kps_l)


class SCRFDDetector:
    input_mean = 127.5
    input_std = 128.0

    def __init__(self, model_path: str, provider: str = "cpu", det_size: int = 320,
                 threshold: float = 0.5, nms_thresh: float = 0.4, allow_cpu_fallback=False):
        self.session = create_session(model_path, provider, allow_cpu_fallback)
        self.input_name = self.session.get_inputs()[0].name
        self.det_size = det_size
        self.threshold = threshold
        self.nms_thresh = nms_thresh

    def detect(self, frame: np.ndarray, max_num: int = 0) -> List[Face]:
        s = self.det_size
        im_ratio = frame.shape[0] / frame.shape[1]
        if im_ratio > 1:
            nh, nw = s, int(s / im_ratio)
        else:
            nw, nh = s, int(s * im_ratio)
        scale = nh / frame.shape[0]
        det_img = np.zeros((s, s, 3), dtype=np.uint8)
        det_img[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / self.input_std, (s, s),
                                     (self.input_mean,) * 3, swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})
        scores, bboxes, kps = decode(outs, (s, s), threshold=self.threshold)
        if scores.size == 0:
            return []
        bboxes, kps = bboxes / scale, kps / scale
        order = scores.argsort()[::-1]
        dets = np.hstack([bboxes, scores[:, None]])[order]
        kps = kps[order]
        keep = nms(dets, self.nms_thresh)
        faces = [Face(bbox=dets[i, :4].astype(np.float32), score=float(dets[i, 4]),
                      kps=kps[i].astype(np.float32)) for i in keep]
        if max_num and len(faces) > max_num:
            faces = sorted(faces, key=lambda f: f.area, reverse=True)[:max_num]
        return faces
