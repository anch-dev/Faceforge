"""ONNX Runtime session creation with honest provider reporting."""
from __future__ import annotations

import os
from typing import List, Optional

import onnxruntime as ort

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

# Verified (onnxruntime-openvino 1.24.1, CPU-only host): "AUTO:GPU,NPU,CPU"
# makes the provider fail to initialise when the listed devices are absent,
# and ONNX Runtime silently runs on CPU instead. Plain "AUTO" lets OpenVINO
# choose among the devices actually present.
PROVIDER_OPTIONS = {
    "OpenVINOExecutionProvider": {"device_type": os.environ.get("FACEFORGE_OV_DEVICE", "AUTO")},
}


class BackendUnavailable(RuntimeError):
    """Raised when a requested accelerator did not actually run the model."""


def resolve(provider: str) -> str:
    return PROVIDER_ALIASES.get(provider.lower(), provider)


def create_session(path: str, provider: str = "cpu", allow_cpu_fallback: bool = False,
                   threads: Optional[int] = None) -> ort.InferenceSession:
    """Create a session on `provider`.

    ONNX Runtime silently drops to CPU when a provider's DLLs are missing.
    Unless allow_cpu_fallback is set, that is turned into BackendUnavailable
    so callers (and benchmarks) never report CPU numbers as GPU numbers.
    """
    wanted = resolve(provider)
    chain: List = []
    if wanted != "CPUExecutionProvider":
        opts = PROVIDER_OPTIONS.get(wanted)
        chain.append((wanted, opts) if opts else wanted)
    chain.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, sess_options=so, providers=chain)
    active = sess.get_providers()[0]
    if active != wanted and not allow_cpu_fallback:
        raise BackendUnavailable(
            f"{path}: requested {wanted} but ONNX Runtime ran it on {active}. "
            f"Available here: {ort.get_available_providers()}"
        )
    return sess


def active_provider(sess: ort.InferenceSession) -> str:
    return sess.get_providers()[0]
