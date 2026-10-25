# PyInstaller runtime hook: runs inside the frozen app before run.py.
#
# 1. Registers GPU runtime DLL folders. Python 3.8+ ignores PATH for the
#    dependencies of extension modules, and upstream's own DLL setup looks in
#    site-packages, which doesn't exist inside a frozen app. Without this the
#    OpenVINO / CUDA providers fail to load and ONNX Runtime quietly runs on CPU.
# 2. For the CUDA build, checks the exact DLLs onnxruntime-gpu 1.26.0 links
#    against (read from its import table) and names any that are missing.
# 3. FACEFORGE_SELFTEST=<provider> builds a tiny synthetic model (no weights
#    are shipped), runs it on that provider, prints which provider actually
#    ran it, and exits 0 / 3. CI uses this.
import glob
import os
import sys

# Verified from onnxruntime_providers_cuda.dll (onnxruntime-gpu 1.26.0, CUDA 12.8 build).
CUDA_REQUIRED_DLLS = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll",
                      "cufft64_11.dll", "cudnn64_9.dll"]

_dll_handles = []  # keep add_dll_directory handles alive for the process lifetime


def _register_dll_dirs():
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    candidates = [os.path.join(base, "openvino", "libs")]
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        candidates.append(os.path.join(cuda_path, "bin"))
    # cuDNN 9 installs to ...\CUDNN\v9.x\bin\12.x[\x64]
    for root in sorted(glob.glob(r"C:\Program Files\NVIDIA\CUDNN\v9.*"), reverse=True):
        for sub in sorted(glob.glob(os.path.join(root, "bin", "12.*")), reverse=True):
            candidates += [os.path.join(sub, "x64"), sub]
        break
    for d in candidates:
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                _dll_handles.append(os.add_dll_directory(d))
            except OSError:
                pass


def _cuda_preflight():
    import ctypes
    missing = []
    for dll in CUDA_REQUIRED_DLLS:
        try:
            ctypes.WinDLL(dll)
        except OSError:
            missing.append(dll)
    if missing:
        print("\n" + "!" * 72, flush=True)
        print("[Faceforge] CUDA build: these NVIDIA runtime DLLs could not be loaded:", flush=True)
        for dll in missing:
            print(f"    {dll}", flush=True)
        print("  Install CUDA 12.x (cudart, cuBLAS, cuFFT) and cuDNN 9, or use the directml build.", flush=True)
        print("  Without them the GPU is NOT used and everything runs on the CPU.", flush=True)
        print("!" * 72 + "\n", flush=True)
    return missing


def _selftest(provider: str) -> int:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    wanted = {"dml": "DmlExecutionProvider", "cuda": "CUDAExecutionProvider",
              "openvino": "OpenVINOExecutionProvider", "cpu": "CPUExecutionProvider"}[provider]
    w = numpy_helper.from_array(np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), helper.make_node("Relu", ["c"], ["y"])],
        "selftest", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 64, 64])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    opts = {"OpenVINOExecutionProvider": {"device_type": os.environ.get("FACEFORGE_OV_DEVICE", "AUTO")}}
    chain = [(wanted, opts[wanted]) if wanted in opts else wanted]
    if wanted != "CPUExecutionProvider":
        chain.append("CPUExecutionProvider")
    print(f"[selftest] onnxruntime {ort.__version__}, available: {ort.get_available_providers()}", flush=True)
    sess = ort.InferenceSession(model.SerializeToString(), providers=chain)
    sess.run(None, {"x": np.zeros((1, 3, 64, 64), np.float32)})
    active = sess.get_providers()[0]
    print(f"[selftest] requested {wanted}, ran on {active}", flush=True)
    return 0 if active == wanted else 3


if sys.platform == "win32" and getattr(sys, "frozen", False):
    _register_dll_dirs()
    if any(a == "cuda" for a in sys.argv) and "--execution-provider" in sys.argv:
        _cuda_preflight()

_selftest_provider = os.environ.get("FACEFORGE_SELFTEST")
if _selftest_provider:
    sys.exit(_selftest(_selftest_provider))
