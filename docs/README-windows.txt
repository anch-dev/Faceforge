FACEFORGE (Windows)
===================

Start it with Faceforge.bat, not Faceforge.exe. The .bat picks the GPU backend.

First launch: the first time you click Live it downloads the face-detection
models (about 340 MB) to %USERPROFILE%\.insightface. That's a one-time wait.

IS MY GPU BEING USED?
Look at the console window for:
  [Faceforge] swap model running on: DmlExecutionProvider       <- good (directml)
  [Faceforge] swap model running on: CUDAExecutionProvider      <- good (cuda)
  [Faceforge] swap model running on: OpenVINOExecutionProvider  <- good (openvino)
  [Faceforge] swap model running on: CPUExecutionProvider       <- GPU NOT USED, slow

FOR BEST FPS IN LIVE MODE
- Face Enhancer: off
- Detection size: 320
- Camera resolution: 640x480 or 720p
- Many faces / Mouth mask: off unless you need them

Use responsibly: get consent before using a real person's face, and label
anything you share as a deepfake.
