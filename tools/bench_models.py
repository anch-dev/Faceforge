"""Benchmark ONNX face-pipeline models on a chosen ONNX Runtime provider.

Usage:
    python tools/bench_models.py --provider cpu|cuda|dml|openvino [--runs 50] MODEL.onnx [...]

Reports, per model: the provider that actually ran it (not just the one
requested), p50/p95 latency, and process memory growth. Feeds random inputs
with the model's declared shapes; dynamic dims get sensible defaults for the
known face models.

Prints a JSON line per model so results can be collected from CI logs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

# Defaults for dynamic dimensions on known models (name fragment -> HxW).
DYNAMIC_HW = {"det_10g": 320, "scrfd": 320, "yunet": 320}

ORT_TYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
}


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return float("nan")


def fake_inputs(session: ort.InferenceSession, model_name: str) -> dict:
    hw = next((v for k, v in DYNAMIC_HW.items() if k in model_name), 256)
    feeds = {}
    rng = np.random.default_rng(0)
    for inp in session.get_inputs():
        shape = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(1 if i == 0 else (3 if i == 1 else hw))
        dtype = ORT_TYPES.get(inp.type, np.float32)
        feeds[inp.name] = rng.standard_normal(shape).astype(dtype)
    return feeds


def bench(path: str, provider: str, runs: int, warmup: int) -> dict:
    wanted = PROVIDERS[provider]
    providers = [wanted] if wanted == "CPUExecutionProvider" else [wanted, "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3  # det_10g declares 640px output shapes; quiet that noise
    mem0 = rss_mb()
    t0 = time.perf_counter()
    session = ort.InferenceSession(path, sess_options=opts, providers=providers)
    load_s = time.perf_counter() - t0
    active = session.get_providers()[0]
    feeds = fake_inputs(session, os.path.basename(path))
    for _ in range(warmup):
        session.run(None, feeds)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {
        "model": os.path.basename(path),
        "requested": wanted,
        "active": active,
        "fell_back_to_cpu": active == "CPUExecutionProvider" and wanted != active,
        "inputs": {k: list(v.shape) for k, v in feeds.items()},
        "load_s": round(load_s, 2),
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[max(0, int(len(times) * 0.95) - 1)], 2),
        "mem_growth_mb": round(rss_mb() - mem0, 1),
        "runs": runs,
        "ort": ort.__version__,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="cpu", choices=sorted(PROVIDERS))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("models", nargs="+")
    args = ap.parse_args(argv)
    rc = 0
    for m in args.models:
        result = bench(m, args.provider, args.runs, args.warmup)
        print(json.dumps(result), flush=True)
        if result["fell_back_to_cpu"]:
            print(f"WARNING: {result['model']} requested {result['requested']} but ran on CPU",
                  file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
