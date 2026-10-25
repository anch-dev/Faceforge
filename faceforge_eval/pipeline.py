"""End-to-end swap pipeline with per-stage timing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .compositor import Compositor
from .types import Face


@dataclass
class FrameTiming:
    stages_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.stages_ms.values())


class Pipeline:
    """detect -> (optional cached) target face -> swap -> composite.

    `detect_every` mirrors Deep-Live-Cam's live mode, which re-runs detection
    only every N frames and reuses the last face in between.
    """

    def __init__(self, detector, swapper, compositor: Optional[Compositor] = None,
                 detect_every: int = 1):
        self.detector = detector
        self.swapper = swapper
        self.compositor = compositor or Compositor()
        self.detect_every = max(1, detect_every)
        self._frame_idx = 0
        self._cached: Optional[Face] = None

    def process(self, frame: np.ndarray, source_embedding: np.ndarray):
        t = FrameTiming()
        if self._frame_idx % self.detect_every == 0 or self._cached is None:
            t0 = time.perf_counter()
            faces = self.detector.detect(frame, max_num=1)
            t.stages_ms["detect"] = (time.perf_counter() - t0) * 1000
            self._cached = faces[0] if faces else None
        self._frame_idx += 1
        if self._cached is None:
            return frame, t
        t0 = time.perf_counter()
        fake, M = self.swapper.swap(frame, self._cached, source_embedding)
        t.stages_ms["swap"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        out = self.compositor.paste(frame, fake, M)
        t.stages_ms["composite"] = (time.perf_counter() - t0) * 1000
        return out, t


def summarize(frame_ms: List[float]) -> Dict[str, float]:
    """FPS figures from per-frame processing times.

    avg_fps = frames / total time. one_pct_low_fps = FPS implied by the mean of
    the slowest 1% of frames (at least one frame) — the usual "1% low" metric.
    """
    if not frame_ms:
        return {}
    arr = np.sort(np.asarray(frame_ms, dtype=np.float64))
    n_low = max(1, int(np.ceil(len(arr) * 0.01)))
    return {
        "frames": int(len(arr)),
        "avg_fps": round(1000.0 * len(arr) / arr.sum(), 2),
        "one_pct_low_fps": round(1000.0 / arr[-n_low:].mean(), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr[-1]), 2),
    }
