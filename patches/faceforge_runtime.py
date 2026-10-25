"""Faceforge runtime helpers, copied into upstream as modules/faceforge_runtime.py.

Kept free of Deep-Live-Cam imports so it can be unit-tested on its own.
Covers:
  * where model weights come from, and their licence notice (shown once
    before the first download; Faceforge release zips contain no weights)
  * local fp32 -> fp16 conversion (instead of downloading a third-party fp16 copy)
  * honest reporting of the provider each ONNX session actually runs on
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from typing import Callable, Iterable, List, Optional

GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "ROCMExecutionProvider",
)

INSIGHTFACE_TERMS = ("InsightFace models are released for non-commercial research purposes only "
                     "(https://github.com/deepinsight/insightface#license).")

# Publisher download locations. Anything listed here is fetched from the
# publisher instead of the upstream Hugging Face mirror.
PUBLISHER_URLS = {
    "inswapper_128.onnx":
        "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
}

# Files never fetched from a third-party mirror; Faceforge derives them locally.
MIRROR_BLOCKED = {"inswapper_128_fp16.onnx"}

LICENSE_NOTES = {
    "inswapper_128.onnx": INSIGHTFACE_TERMS,
    "buffalo_l": INSIGHTFACE_TERMS,
    "GPEN-BFR-256.onnx": "GPEN weights: non-commercial use (per FaceFusion's licence list); "
                         "downloaded from the Deep-Live-Cam mirror on Hugging Face.",
    "GPEN-BFR-512.onnx": "GPEN weights: non-commercial use (per FaceFusion's licence list); "
                         "downloaded from the Deep-Live-Cam mirror on Hugging Face.",
    "gfpgan-1024.onnx": "GFPGAN: Apache-2.0 code; ONNX export downloaded from the "
                        "Deep-Live-Cam mirror on Hugging Face.",
}

_noticed: set = set()


def requested_gpu(execution_providers: Iterable) -> List[str]:
    names = [p[0] if isinstance(p, tuple) else p for p in execution_providers]
    return [n for n in names if n in GPU_PROVIDERS]


def prefers_fp16(execution_providers: Iterable, has_torch_cuda: bool = False) -> bool:
    """fp16 swap model on any GPU backend. FACEFORGE_FP32=1 forces fp32
    (e.g. GTX 16-series cards that produce NaN/black faces in fp16)."""
    if os.environ.get("FACEFORGE_FP32") == "1":
        return False
    return has_torch_cuda or bool(requested_gpu(execution_providers))


def cuda_graphs_ok(execution_providers: Iterable) -> bool:
    if os.environ.get("FACEFORGE_NO_CUDA_GRAPH") == "1":
        return False
    return "CUDAExecutionProvider" in [p[0] if isinstance(p, tuple) else p for p in execution_providers]


def publisher_url(name: str) -> Optional[str]:
    return PUBLISHER_URLS.get(name.replace(os.sep, "/"))


def download_allowed(name: str) -> bool:
    return name.replace(os.sep, "/") not in MIRROR_BLOCKED


def license_notice(name: str, url: str, printer: Callable[[str], None] = print) -> None:
    key = name.replace(os.sep, "/")
    if key in _noticed:
        return
    _noticed.add(key)
    note = LICENSE_NOTES.get(key) or LICENSE_NOTES.get(key.split("/")[0])
    printer(f"[Faceforge] Downloading {key} from {url}")
    if note:
        printer(f"[Faceforge] Licence: {note} You are responsible for complying with it.")


def session_active_provider(session) -> Optional[str]:
    session = getattr(session, "_underlying", session)  # CUDA-graph adapter
    try:
        providers = session.get_providers()
    except Exception:
        return None
    return providers[0] if providers else None


def report_session(label: str, session, execution_providers: Iterable,
                   printer: Callable[[str], None] = print,
                   status: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Print which provider `session` runs on. Returns a warning message if a
    GPU backend was requested but ONNX Runtime fell back to CPU, else None."""
    active = session_active_provider(session)
    if active is None:
        return None
    printer(f"[Faceforge] {label} running on: {active}")
    if requested_gpu(execution_providers) and active not in GPU_PROVIDERS:
        msg = (f"GPU NOT IN USE for {label} - fell back to CPU, expect very low FPS. "
               "Check GPU drivers (and CUDA 12 + cuDNN 9 for the cuda build).")
        printer("\n" + "!" * 72 + f"\n[Faceforge] {msg}\n" + "!" * 72 + "\n")
        if status:
            status(msg)
        return msg
    return None


def report_face_analyser(analyser, execution_providers: Iterable,
                         printer: Callable[[str], None] = print,
                         status: Optional[Callable[[str], None]] = None) -> List[str]:
    warnings_out = []
    for task, model in getattr(analyser, "models", {}).items():
        w = report_session(f"face analyser [{task}]", getattr(model, "session", None),
                           execution_providers, printer, status)
        if w:
            warnings_out.append(w)
    return warnings_out


def convert_fp16(src: str, dst: str) -> List[str]:
    """fp32 -> fp16 ONNX with float32 I/O. Initializers no node consumes (e.g.
    inswapper's 'emap', read directly by insightface) keep their dtype.
    Written atomically. Returns the names of initializers kept as-is."""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(src)
    consumed = {name for node in model.graph.node for name in node.input}
    detached = {}
    for init in model.graph.initializer:
        if init.name not in consumed:
            keep = onnx.TensorProto()
            keep.CopyFrom(init)
            detached[init.name] = keep

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # per-weight clamp warnings
        converted = float16.convert_float_to_float16(model, keep_io_types=True)

    for init in converted.graph.initializer:
        if init.name in detached:
            init.CopyFrom(detached[init.name])

    out_dir = os.path.dirname(os.path.abspath(dst)) or "."
    fd, tmp = tempfile.mkstemp(suffix=".onnx.part", dir=out_dir)
    os.close(fd)
    try:
        onnx.save(converted, tmp)
        os.chmod(tmp, 0o644)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return sorted(detached)


def ensure_local_fp16(fp32_path: str, fp16_path: str, obtain_fp32: Callable[[], bool],
                      printer: Callable[[str], None] = print) -> bool:
    """Make sure an fp16 swap model exists, deriving it from the publisher's
    fp32 file. Returns False (caller uses fp32) on any failure."""
    if os.path.exists(fp16_path):
        return True
    if not os.path.exists(fp32_path) and not obtain_fp32():
        return False
    if not os.path.exists(fp32_path):
        return False
    printer(f"[Faceforge] Creating fp16 copy of {os.path.basename(fp32_path)} for GPU (one time)...")
    try:
        convert_fp16(fp32_path, fp16_path)
        return True
    except Exception as e:  # missing converter, disk full, corrupt source...
        printer(f"[Faceforge] fp16 conversion failed ({e}); using fp32.")
        return False


if __name__ == "__main__":  # CLI: python faceforge_runtime.py SRC.onnx DST.onnx
    if len(sys.argv) != 3:
        print("usage: faceforge_runtime.py SRC.onnx DST.onnx")
        sys.exit(2)
    print("kept fp32:", convert_fp16(sys.argv[1], sys.argv[2]))
