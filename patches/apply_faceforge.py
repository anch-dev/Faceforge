"""Faceforge patches applied on top of upstream Deep-Live-Cam at build time.

Usage:  python apply_faceforge.py <path-to-upstream-checkout>

Every patch is an exact text replacement. If upstream changes the code a
patch targets, the script exits non-zero so the build fails loudly instead
of silently shipping an unpatched (slow) app.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
failures: list[str] = []


def patch(rel: str, old: str, new: str, label: str) -> None:
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[skip] {label} (already applied)")
        return
    count = text.count(old)
    if count != 1:
        failures.append(f"{label}: expected 1 match in {rel}, found {count}")
        print(f"[FAIL] {label}")
        return
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[ok]   {label}")


# 1. Frozen-app exit: quit() is a site builtin that doesn't exist under PyInstaller.
patch(
    "modules/core.py",
    "    if to_quit:\n        quit()",
    "    if to_quit:\n        sys.exit()",
    "core: sys.exit instead of quit()",
)

# 2. Live-friendly default detection size. 320 roughly quarters detection cost
#    vs 640 and is plenty for a face filling a webcam frame. Still selectable
#    in the UI (160/320/640); a saved UI choice overrides this default.
patch(
    "modules/globals.py",
    "DEFAULT_DET_SIZE: int = 640",
    "DEFAULT_DET_SIZE: int = 320  # Faceforge: 320 for live speed (was 640)",
    "globals: default det_size 320",
)

# 3. Pick the fp16 swap model and CUDA graphs based on the ONNX Runtime
#    provider actually in use, not on PyTorch being installed. Packaged
#    builds ship without torch, so upstream silently fell back to the slow
#    fp32 path and skipped CUDA graphs on every GPU.
patch(
    "modules/processors/frame/face_swapper.py",
    "            use_fp16 = _HAS_TORCH_CUDA and os.path.exists(fp16_path)",
    "            use_fp16 = _faceforge_prefers_fp16() and os.path.exists(fp16_path)",
    "face_swapper: fp16 by provider",
)
patch(
    "modules/processors/frame/face_swapper.py",
    "    variants = [\"inswapper_128.onnx\", \"inswapper_128_fp16.onnx\"]\n    if _HAS_TORCH_CUDA:",
    "    variants = [\"inswapper_128.onnx\", \"inswapper_128_fp16.onnx\"]\n    if _faceforge_prefers_fp16():",
    "face_swapper: download fp16 first on GPU",
)
patch(
    "modules/processors/frame/face_swapper.py",
    "                if _HAS_TORCH_CUDA and any(\n",
    "                if _faceforge_cuda_graphs_ok() and any(\n",
    "face_swapper: CUDA graphs without torch",
)

# 4. Say loudly when a GPU was requested but ONNX Runtime fell back to CPU
#    (the #1 cause of "it runs at 2 FPS": CUDA/cuDNN missing or mismatched).
patch(
    "modules/processors/frame/face_swapper.py",
    "                update_status(\"Face swapper model loaded successfully.\", NAME)",
    "                _faceforge_report_providers(FACE_SWAPPER)\n"
    "                update_status(\"Face swapper model loaded successfully.\", NAME)",
    "face_swapper: CPU fallback warning",
)
patch(
    "modules/processors/frame/face_swapper.py",
    "_HAS_TORCH_CUDA = False\ntry:\n    import torch\n",
    '''_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "ROCMExecutionProvider",
)


def _faceforge_requested_gpu() -> list:
    return [p for p in modules.globals.execution_providers
            if (p[0] if isinstance(p, tuple) else p) in _GPU_PROVIDERS]


def _faceforge_prefers_fp16() -> bool:
    # FACEFORGE_FP32=1 forces fp32 (e.g. GTX 16xx cards that NaN on fp16).
    if os.environ.get("FACEFORGE_FP32") == "1":
        return False
    return _HAS_TORCH_CUDA or bool(_faceforge_requested_gpu())


def _faceforge_cuda_graphs_ok() -> bool:
    if os.environ.get("FACEFORGE_NO_CUDA_GRAPH") == "1":
        return False
    return "CUDAExecutionProvider" in modules.globals.execution_providers


def _faceforge_report_providers(swapper) -> None:
    session = getattr(swapper, "session", None)
    session = getattr(session, "_underlying", session)
    try:
        active = session.get_providers()
    except Exception:
        return
    print(f"[Faceforge] swap model running on: {active[0] if active else '?'}", flush=True)
    wanted = _faceforge_requested_gpu()
    if wanted and active and active[0] not in _GPU_PROVIDERS:
        msg = ("GPU NOT IN USE - fell back to CPU, expect very low FPS. "
               "Check GPU drivers (and CUDA 12 + cuDNN 9 for the cuda build).")
        print("\\n" + "!" * 72 + f"\\n[Faceforge] {msg}\\n" + "!" * 72 + "\\n", flush=True)
        update_status(msg, NAME)


_HAS_TORCH_CUDA = False
try:
    import torch
''',
    "face_swapper: Faceforge helpers",
)

# 5. The NSFW filter needs keras + a deep-learning backend we don't ship.
#    Make it degrade to "off" instead of crashing the run.
patch(
    "modules/ui.py",
    "    from numpy import ndarray\n    from modules.predicter import predict_frame, predict_image, predict_video\n",
    "    from numpy import ndarray\n"
    "    try:\n"
    "        from modules.predicter import predict_frame, predict_image, predict_video\n"
    "    except Exception:\n"
    "        update_status(\"NSFW filter is not included in this build; skipping.\")\n"
    "        return False\n",
    "ui: NSFW filter optional",
)

if failures:
    print("\nFaceforge patching FAILED - upstream changed:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\nAll Faceforge patches applied.")
