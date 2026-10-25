"""Run the Faceforge offload server.

    python -m faceforge_server --accept-model-license

Prints the pairing code and this PC's addresses to enter on the iPhone.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "patches"))

import onnxruntime as ort  # noqa: E402

from faceforge_eval.detector import SCRFDDetector  # noqa: E402
from faceforge_eval.models import DownloadError, ensure_models  # noqa: E402
from faceforge_eval.recognizer import ArcFaceEmbedder  # noqa: E402
from faceforge_eval.runtime import BackendUnavailable, resolve  # noqa: E402
from faceforge_eval.swapper import InSwapper  # noqa: E402

from . import protocol as P  # noqa: E402
from .server import OffloadServer, SwapEngine, advertise_bonjour, lan_addresses  # noqa: E402

AUTO_ORDER = ["cuda", "dml", "openvino", "cpu"]


def preload_gpu_dlls() -> None:
    # onnxruntime-gpu >= 1.21: load CUDA/cuDNN from the nvidia-* pip wheels
    # installed by onnxruntime-gpu[cuda,cudnn]; no system CUDA install needed.
    if hasattr(ort, "preload_dlls") and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()
        except Exception as e:  # pragma: no cover - platform specific
            print(f"[server] preload_dlls: {e}")


def gpu_name() -> str:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    except (OSError, IndexError):
        return ""


def build_engine(paths: dict, provider: str, det_size: int, models_dir: str):
    candidates = AUTO_ORDER if provider == "auto" else [provider]
    last_err = None
    for p in candidates:
        if resolve(p) not in ort.get_available_providers():
            continue
        try:
            swap_path = paths["swapper"]
            if p != "cpu":  # fp16 copy derived locally from the publisher's fp32 file
                from faceforge_runtime import ensure_local_fp16
                fp16 = os.path.join(models_dir, "inswapper_128_fp16.onnx")
                if ensure_local_fp16(swap_path, fp16, lambda: False):
                    swap_path = fp16
            det = SCRFDDetector(paths["detector"], p, det_size=det_size)
            rec = ArcFaceEmbedder(paths["recognizer"], p)
            sw = InSwapper(swap_path, p)
            return SwapEngine(det, rec, sw, provider=resolve(p), device=gpu_name() if p == "cuda" else "")
        except BackendUnavailable as e:
            print(f"[server] {p}: not usable here ({e}); trying next backend")
            last_err = e
    raise SystemExit(f"No usable backend. Last error: {last_err}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=P.DEFAULT_PORT)
    ap.add_argument("--provider", default="auto", choices=["auto"] + AUTO_ORDER)
    ap.add_argument("--models-dir", default=os.path.join(REPO, "models"))
    ap.add_argument("--det-size", type=int, default=320, choices=[160, 320, 640])
    ap.add_argument("--token", help="pairing code (default: random 6 digits)")
    ap.add_argument("--accept-model-license", action="store_true",
                    help="allow downloading InsightFace models (non-commercial research licence)")
    args = ap.parse_args(argv)

    preload_gpu_dlls()
    try:
        paths = ensure_models(args.models_dir, args.accept_model_license)
    except DownloadError as e:
        print(f"[server] {e}")
        return 2
    engine = build_engine(paths, args.provider, args.det_size, args.models_dir)
    token = args.token or f"{secrets.randbelow(10**6):06d}"
    server = OffloadServer(engine, token)

    async def run():
        srv = await asyncio.start_server(server.handle, args.host, args.port)
        bonjour = advertise_bonjour(args.port)
        print("=" * 60)
        print(f" Faceforge server running on {engine.provider} {engine.device}".rstrip())
        for a in lan_addresses() or ["(no LAN address found)"]:
            print(f"   Address:      {a}:{args.port}")
        print(f"   Pairing code: {token}")
        print(" Enter these in the Faceforge app (Settings > PC offload).")
        print(" Windows may ask to allow Python on private networks: allow it.")
        print("=" * 60, flush=True)
        try:
            async with srv:
                await srv.serve_forever()
        finally:
            if bonjour:
                bonjour[0].unregister_service(bonjour[1])
                bonjour[0].close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
