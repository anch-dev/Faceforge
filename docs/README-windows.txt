FACEFORGE (Windows)
===================

Start it with Faceforge.bat, not Faceforge.exe. The .bat picks the GPU backend.

FIRST RUN
Model weights are not included. The first time you click Live, the app
downloads them from InsightFace's own GitHub releases (about 840 MB) and
prints the source and licence first. InsightFace models are licensed for
non-commercial research use only. GPU builds then make a one-time fp16
copy of the swap model.

IS MY GPU BEING USED?
The console shows, for every model:
  [Faceforge] swap model running on: DmlExecutionProvider       <- good (directml)
  [Faceforge] swap model running on: CUDAExecutionProvider      <- good (cuda)
  [Faceforge] swap model running on: OpenVINOExecutionProvider  <- good (openvino)
  [Faceforge] swap model running on: CPUExecutionProvider       <- GPU NOT USED, very slow
The cuda build also lists any missing NVIDIA DLLs at startup.

FOR BEST FPS IN LIVE MODE
- Face Enhancer: off
- Detection size: 320
- Camera resolution: 640x480 or 720p
- Many faces / Mouth mask: off unless you need them

SETTINGS (environment variables)
  FACEFORGE_FP32=1          use the fp32 swap model (GTX 16-series black faces)
  FACEFORGE_OV_DEVICE=GPU   pin the OpenVINO device (default AUTO)

BUILD-DEPENDENCIES.txt lists the exact package versions in this build.

Use responsibly: get consent before using a real person's face, and label
anything you share as a deepfake.
