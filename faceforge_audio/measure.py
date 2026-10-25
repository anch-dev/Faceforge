"""Measure a VoiceProcessor: real latency, real-time factor, underruns."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from .interface import VoiceProcessor


def _test_signal(n: int, sample_rate: int, seed: int = 0) -> np.ndarray:
    """Band-limited noise burst train: unambiguous cross-correlation peak."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32) * 0.2
    gate = (np.arange(n) // (sample_rate // 10)) % 2 == 0
    return (x * gate).astype(np.float32)


def run_stream(proc: VoiceProcessor, signal: np.ndarray):
    """Feed `signal` block by block; returns output and per-block seconds."""
    bs = proc.block_size
    n = len(signal) // bs * bs
    out = np.empty(n, np.float32)
    times = np.empty(n // bs)
    for i in range(n // bs):
        t = time.perf_counter()
        out[i * bs:(i + 1) * bs] = proc.process(signal[i * bs:(i + 1) * bs])
        times[i] = time.perf_counter() - t
    return out, times


def measure_latency_samples(proc: VoiceProcessor, seconds: float = 2.0) -> int:
    """Actual input->output delay of the processor, by cross-correlation.
    Meaningful for processors that preserve the signal's timing structure
    (pass-through, pitch/timbre conversion); the model's declared
    algorithmic_latency_samples should match this."""
    proc.reset()
    sig = _test_signal(int(seconds * proc.sample_rate), proc.sample_rate)
    out, _ = run_stream(proc, sig)
    sig = sig[:len(out)]
    corr = np.fft.irfft(np.fft.rfft(out, 2 * len(out)) * np.conj(np.fft.rfft(sig, 2 * len(out))))
    lag = int(np.argmax(corr[:len(out)]))
    return lag


def stream_benchmark(proc: VoiceProcessor, seconds: float = 5.0) -> Dict[str, float]:
    proc.reset()
    sig = _test_signal(int(seconds * proc.sample_rate), proc.sample_rate, seed=1)
    _, times = run_stream(proc, sig)
    block_s = proc.block_size / proc.sample_rate
    p95 = float(np.percentile(times, 95))
    return {
        "sample_rate": proc.sample_rate,
        "block_ms": round(block_s * 1000, 2),
        "algorithmic_latency_ms": round(proc.algorithmic_latency_samples / proc.sample_rate * 1000, 2),
        "proc_p50_ms": round(float(np.median(times)) * 1000, 3),
        "proc_p95_ms": round(p95 * 1000, 3),
        "real_time_factor": round(float(times.mean()) / block_s, 4),   # < 1 keeps up
        "underrun_blocks": int((times > block_s).sum()),
        "blocks": int(len(times)),
        # Excludes device input/output buffers, which the platform adds.
        "pipeline_latency_ms": round((block_s + proc.algorithmic_latency_samples / proc.sample_rate + p95) * 1000, 2),
    }
