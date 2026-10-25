# PyInstaller runtime hook: runs inside the frozen app before run.py.
#
# 1. Registers GPU runtime DLL folders. Python 3.8+ ignores PATH for the
#    dependencies of extension modules, and upstream's own DLL setup looks in
#    site-packages, which doesn't exist inside a frozen app. Without this the
#    OpenVINO / CUDA providers fail to load and ONNX Runtime quietly runs on CPU.
# 2. FACEFORGE_SELFTEST=<provider> loads the swap model on that provider,
#    prints which provider actually ran it, and exits. CI uses this.
import glob
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
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
                os.add_dll_directory(d)
            except OSError:
                pass

_selftest = os.environ.get("FACEFORGE_SELFTEST")
if _selftest:
    import onnxruntime as ort

    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.argv[0])))
    model = os.path.join(base, "models", "inswapper_128_fp16.onnx")
    wanted = {
        "dml": "DmlExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
    }[_selftest]
    print(f"[selftest] available: {ort.get_available_providers()}", flush=True)
    opts = {"openvino": [(wanted, {"device_type": "AUTO:GPU,NPU,CPU"})]}.get(
        _selftest, [wanted]
    )
    sess = ort.InferenceSession(model, providers=opts + ["CPUExecutionProvider"])
    active = sess.get_providers()[0]
    print(f"[selftest] swap model ran on: {active}", flush=True)
    sys.exit(0 if active == wanted else 3)
