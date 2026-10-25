# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Faceforge (Deep-Live-Cam v2.8+, PySide6 UI).
# Run from the upstream checkout root: pyinstaller ../faceforge/build/faceforge.spec
import glob
import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

SPEC_DIR = SPECPATH  # set by PyInstaller: the folder containing this spec
# PyInstaller resolves relative paths against the spec's folder, not the
# working directory, so everything upstream is addressed from ROOT explicitly.
ROOT = os.path.abspath(os.environ.get("FACEFORGE_UPSTREAM_ROOT", os.getcwd()))
VARIANT = os.environ.get("FACEFORGE_VARIANT", "directml")

# No model weights are bundled: models/ only holds upstream's instructions.txt.
# Weights are downloaded on first use from their publishers (see
# modules/faceforge_runtime.py) under their own licences.
datas = [
    (os.path.join(ROOT, "models"), "models"),
    (os.path.join(ROOT, "modules", "ui.json"), "modules"),
    (os.path.join(ROOT, "locales"), "locales"),
]
datas += collect_data_files("insightface")
# onnxruntime/datasets holds tiny test models; not needed at runtime.
datas += collect_data_files("onnxruntime", excludes=["**/datasets/**"])

binaries = collect_dynamic_libs("onnxruntime")
binaries += collect_dynamic_libs("cv2")

if VARIANT == "openvino":
    # OpenVINO runtime + device plugins; registered at startup by the rthook.
    import openvino
    ov_libs = os.path.join(os.path.dirname(openvino.__file__), "libs")
    for f in glob.glob(os.path.join(ov_libs, "*")):
        if os.path.isfile(f):
            binaries.append((f, "openvino/libs"))

# Local app modules (collect_submodules can't see a source tree).
# Skip leftovers from the old Tk UI and the NSFW predictor (needs keras).
SKIP = {"modules.run", "modules.ui_tooltip", "modules.tkinter_fix", "modules.predicter"}
hiddenimports = []
for py_file in glob.glob(os.path.join(ROOT, "modules", "**", "*.py"), recursive=True):
    py_file = os.path.relpath(py_file, ROOT)
    mod = py_file.replace("\\", "/").replace("/", ".").removesuffix(".py")
    mod = mod.removesuffix(".__init__")
    if mod not in SKIP:
        hiddenimports.append(mod)
hiddenimports += collect_submodules("insightface")
# onnxconverter_common: local fp32 -> fp16 conversion of the swap model
# (modules.faceforge_runtime), so no third-party fp16 copy is downloaded.
hiddenimports += collect_submodules("onnxconverter_common")
hiddenimports += ["onnx", "onnx.numpy_helper", "onnxruntime", "onnxruntime.capi",
                  "cv2_enumerate_cameras", "psutil", "requests", "certifi", "sklearn.cluster"]
if sys.platform == "win32":
    hiddenimports += ["pygrabber", "pygrabber.dshow_graph"]

excludes = [
    "tkinter", "_tkinter", "customtkinter",
    "tensorflow", "keras", "opennsfw2", "jax",
    "torch", "torchvision", "torchaudio",
    "IPython", "notebook",
]

a = Analysis(
    [os.path.join(ROOT, "run.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPEC_DIR, "rthook_faceforge.py")],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Faceforge",
    console=True,   # keep the log window: it shows FPS problems and GPU fallback
    upx=False,
    strip=False,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, upx=False, strip=False, name="Faceforge")
