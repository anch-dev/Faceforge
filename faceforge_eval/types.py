from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Face:
    bbox: np.ndarray            # (4,) x1, y1, x2, y2 in frame pixels
    score: float
    kps: np.ndarray             # (5, 2) eyes, nose, mouth corners
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
