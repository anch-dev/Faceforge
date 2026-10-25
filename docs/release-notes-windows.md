### Which zip do I download?

| Build | GPU | Extra installs | Pick it if |
|-------|-----|----------------|-----------|
| **cuda** | NVIDIA | CUDA 12.x + cuDNN 9 | You have an NVIDIA card and want the fastest backend. |
| **directml** | Any (NVIDIA, AMD, Intel) | None | You want it to just work, or have AMD. |
| **openvino** | Intel Arc / Intel iGPU | None | You have an Intel GPU. |

Extract, then run `Faceforge.bat` (not the .exe directly; the .bat selects the GPU backend).

### Models are not included

Faceforge does not redistribute model weights. On first launch the app downloads them from their publishers, about 840 MB in total:

- `inswapper_128.onnx` from InsightFace's GitHub release
- the `buffalo_l` detection and recognition pack from InsightFace's GitHub release

The console shows each download's source and licence before it starts. **InsightFace models are licensed for non-commercial research use only.** On GPU builds the app then makes a local fp16 copy of the swap model (one time, about 10 seconds).

### If it's slow

The console prints which backend each model actually runs on, e.g. `[Faceforge] swap model running on: DmlExecutionProvider`. If you see `CPUExecutionProvider` or the **GPU NOT IN USE** banner, the GPU isn't being used. On a CPU, expect well under 1 FPS. The CUDA build lists exactly which NVIDIA DLLs are missing.

- Leave **Face Enhancer** off for live use.
- Keep **Detection size** at 320 (the Faceforge default) and the camera at 640x480 or 720p.
- **Many faces** and **Mouth mask** add work on every frame.
- If faces come out black on a GTX 16-series card, set `FACEFORGE_FP32=1` before launching.
- On an Intel machine where OpenVINO won't load, set `FACEFORGE_OV_DEVICE=GPU` or `CPU` to pin a device.

> Get consent before using a real person's face, and label shared output as a deepfake.
