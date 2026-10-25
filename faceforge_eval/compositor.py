"""Paste an aligned fake face back into the frame with a feathered mask."""
from __future__ import annotations

import cv2
import numpy as np


class Compositor:
    def __init__(self, feather_ratio: float = 0.08):
        self.feather_ratio = feather_ratio
        self._mask_cache: dict = {}

    def _soft_mask(self, size: int) -> np.ndarray:
        if size not in self._mask_cache:
            m = np.zeros((size, size), np.float32)
            pad = max(2, int(size * 0.06))
            m[pad:-pad, pad:-pad] = 1.0
            k = max(3, int(size * self.feather_ratio)) | 1
            self._mask_cache[size] = cv2.GaussianBlur(m, (k, k), 0)
        return self._mask_cache[size]

    def paste(self, frame: np.ndarray, fake: np.ndarray, M: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        IM = cv2.invertAffineTransform(M)
        mask = cv2.warpAffine(self._soft_mask(fake.shape[0]), IM, (w, h), borderValue=0.0)
        ys, xs = np.nonzero(mask > 0)
        if ys.size == 0:
            return frame
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        # Warp only the face region's bounding box, not the whole frame.
        IM_roi = IM.copy()
        IM_roi[:, 2] -= (x1, y1)
        fake_roi = cv2.warpAffine(fake, IM_roi, (x2 - x1, y2 - y1), borderMode=cv2.BORDER_REPLICATE)
        a = mask[y1:y2, x1:x2, None]
        out = frame.copy()
        roi = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = (a * fake_roi + (1 - a) * roi).astype(np.uint8)
        return out
