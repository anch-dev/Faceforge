from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VoiceProcessor(Protocol):
    sample_rate: int
    block_size: int
    algorithmic_latency_samples: int

    def process(self, block: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


def check_block(block: np.ndarray, block_size: int) -> None:
    if block.dtype != np.float32 or block.ndim != 1 or block.shape[0] != block_size:
        raise ValueError(f"expected float32 mono block of {block_size} samples, "
                         f"got {block.dtype} {block.shape}")


class Passthrough:
    """Reference processor: output == input. Baseline for harness overhead."""

    def __init__(self, sample_rate: int = 48000, block_size: int = 480):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.algorithmic_latency_samples = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        check_block(block, self.block_size)
        return block.copy()

    def reset(self) -> None:
        pass


class Delay:
    """Reference processor with a known algorithmic latency, used to verify
    that the harness measures latency correctly."""

    def __init__(self, delay_samples: int, sample_rate: int = 48000, block_size: int = 480):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.algorithmic_latency_samples = delay_samples
        self._buf = np.zeros(delay_samples, np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        check_block(block, self.block_size)
        joined = np.concatenate([self._buf, block])
        out, self._buf = joined[:self.block_size], joined[self.block_size:]
        return out.astype(np.float32)

    def reset(self) -> None:
        self._buf[:] = 0
