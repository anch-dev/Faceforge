"""Evaluate a swap model end to end: speed, memory and identity quality.

Example (Windows, DirectML):
    python tools/eval_pipeline.py --provider dml \
        --det path/det_10g.onnx --rec path/w600k_r50.onnx \
        --swapper inswapper:path/inswapper_128.onnx \
        --judge path/glintr100.onnx --target group.jpg --frames 300

Reports (JSON on stdout):
  environment   OS, CPU, GPU (if nvidia-smi), Python/ORT versions, providers
  startup_s     time to create all sessions (model load + graph optimisation)
  perf          avg FPS, 1%-low FPS, p50/p95/p99 frame latency, per-stage p50
  memory        peak process RSS; peak VRAM via nvidia-smi on NVIDIA only
  quality       identity similarity of swapped faces to the source, scored by
                an independent recognizer (--judge) when given

Notes
* Frames are a still image processed repeatedly unless --video is given; this
  measures model + pre/post-processing throughput, not camera capture or
  display. Camera-to-display latency must be measured in the app.
* Quality uses every ordered (source, target) pair of faces in --target.
  Scoring with the same ArcFace model that conditions the swap inflates
  results; prefer a different recognizer as --judge.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faceforge_eval.detector import SCRFDDetector  # noqa: E402
from faceforge_eval.pipeline import Pipeline, summarize  # noqa: E402
from faceforge_eval.recognizer import ArcFaceEmbedder, cosine  # noqa: E402
from faceforge_eval.runtime import active_provider  # noqa: E402
from faceforge_eval.swapper import HyperSwap, InSwapper  # noqa: E402

SWAPPERS = {"inswapper": InSwapper, "hyperswap": HyperSwap}


def environment() -> dict:
    env = {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": platform.processor() or "",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    if os.path.exists("/proc/cpuinfo"):  # platform.processor() is just "x86_64" on Linux
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                env["cpu"] = line.split(":", 1)[1].strip()
                break
    gpu = nvidia_smi("name,driver_version,memory.total")
    if gpu:
        env["gpu"] = gpu
    return env


def nvidia_smi(fields: str):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class PeakSampler(threading.Thread):
    """Samples process RSS (and NVIDIA VRAM when available) every 100 ms."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.peak_rss_mb = 0.0
        self.peak_vram_mb = None

    def run(self):
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:
            proc = None
        while not self.stop_flag.is_set():
            if proc:
                self.peak_rss_mb = max(self.peak_rss_mb, proc.memory_info().rss / 2**20)
            used = nvidia_smi("memory.used")
            if used:
                v = float(used.splitlines()[0])
                self.peak_vram_mb = v if self.peak_vram_mb is None else max(self.peak_vram_mb, v)
            self.stop_flag.wait(0.1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="report instead of failing when the provider falls back to CPU")
    ap.add_argument("--det", required=True)
    ap.add_argument("--rec", required=True, help="ArcFace w600k_r50 (identity input to the swapper)")
    ap.add_argument("--swapper", required=True, help="TYPE:PATH, TYPE in " + ",".join(SWAPPERS))
    ap.add_argument("--judge", help="independent recognizer for quality scoring")
    ap.add_argument("--target", required=True, help="image with one or more faces")
    ap.add_argument("--video", help="optional video for throughput instead of a repeated still")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--detect-every", type=int, default=1)
    ap.add_argument("--skip-quality", action="store_true")
    args = ap.parse_args(argv)

    kind, _, path = args.swapper.partition(":")
    sampler = PeakSampler()
    sampler.start()
    fb = args.allow_cpu_fallback

    t0 = time.perf_counter()
    det = SCRFDDetector(args.det, args.provider, det_size=args.det_size, allow_cpu_fallback=fb)
    rec = ArcFaceEmbedder(args.rec, args.provider, allow_cpu_fallback=fb)
    swapper = SWAPPERS[kind](path, args.provider, allow_cpu_fallback=fb)
    judge = ArcFaceEmbedder(args.judge, "cpu") if args.judge else None
    startup_s = time.perf_counter() - t0

    image = cv2.imread(args.target)
    if image is None:
        raise SystemExit(f"cannot read {args.target}")
    faces = det.detect(image)
    if not faces:
        raise SystemExit("no faces found in --target")
    src_emb = rec.embed(image, faces[0])

    # ---- throughput -------------------------------------------------------
    pipe = Pipeline(det, swapper, detect_every=args.detect_every)
    frames = []
    if args.video:
        cap = cv2.VideoCapture(args.video)
        while len(frames) < args.frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    if not frames:
        frames = [image] * args.frames
    pipe.process(frames[0], src_emb)  # warm-up (first-run kernel selection)
    frame_ms, stages = [], {}
    for f in frames:
        t = time.perf_counter()
        _, timing = pipe.process(f, src_emb)
        frame_ms.append((time.perf_counter() - t) * 1000)
        for k, v in timing.stages_ms.items():
            stages.setdefault(k, []).append(v)

    # ---- quality ------------------------------------------------------------
    quality = None
    if not args.skip_quality and len(faces) >= 2:
        scorer = judge or rec
        sims, base = [], []
        embs = [rec.embed(image, f) for f in faces]
        for i, j in itertools.permutations(range(len(faces)), 2):
            fake, M = swapper.swap(image, faces[j], embs[i])
            out = pipe.compositor.paste(image, fake, M)
            # Re-detect in the composited frame and score the face nearest target j.
            cand = det.detect(out)
            if not cand:
                continue
            cj = min(cand, key=lambda c: np.abs(c.bbox - faces[j].bbox).sum())
            sims.append(cosine(scorer.embed(out, cj), scorer.embed(image, faces[i])))
            base.append(cosine(scorer.embed(image, faces[j]), scorer.embed(image, faces[i])))
        quality = {
            "scorer": os.path.basename(args.judge) if judge else os.path.basename(args.rec) + " (biased: same as conditioning model)",
            "pairs": len(sims),
            "id_similarity_mean": round(float(np.mean(sims)), 4) if sims else None,
            "id_similarity_min": round(float(np.min(sims)), 4) if sims else None,
            "baseline_unswapped_mean": round(float(np.mean(base)), 4) if base else None,
        }

    sampler.stop_flag.set()
    sampler.join()
    report = {
        "environment": environment(),
        "config": {"provider": args.provider, "swapper": kind, "model": os.path.basename(path),
                   "det_size": args.det_size, "detect_every": args.detect_every,
                   "frames": len(frames), "faces_in_target": len(faces)},
        "active_providers": {"detector": active_provider(det.session), "recognizer": active_provider(rec.session),
                             "swapper": active_provider(swapper.session)},
        "startup_s": round(startup_s, 2),
        "perf": {**summarize(frame_ms),
                 "stage_p50_ms": {k: round(float(np.median(v)), 2) for k, v in stages.items()}},
        "memory": {"peak_rss_mb": round(sampler.peak_rss_mb, 1),
                   "peak_vram_mb": sampler.peak_vram_mb if sampler.peak_vram_mb is not None
                   else "not measured (needs nvidia-smi; DirectML/OpenVINO VRAM not exposed here)"},
        "quality": quality,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
