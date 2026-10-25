### Which zip do I download?

| Build | GPU | Extra installs | Pick it if |
|-------|-----|----------------|-----------|
| **cuda** | NVIDIA (RTX 20-series or newer) | CUDA 12.8 + cuDNN 9 | You have an NVIDIA card. Fastest. |
| **directml** | Any (NVIDIA, AMD, Intel) | None | You want it to just work, or have AMD. |
| **openvino** | Intel Arc / Intel iGPU / NPU | None | You have an Intel GPU. |

Extract, then run `Faceforge.bat` (not the .exe directly, since the .bat selects the GPU).

### If it's slow

Watch the console window. It prints `[Faceforge] swap model running on: ...`.
If it says `CPUExecutionProvider` or shows the **GPU NOT IN USE** banner, the GPU isn't being used. Install the extras for your build, or switch to the directml build.

- Leave **Face Enhancer** off for live use. It is the single biggest FPS cost.
- Keep **Detection size** at 320 (the Faceforge default) and camera at 640x480 or 720p.
- **Many faces** and **Mouth mask** both add work on every frame.
- GTX 16-series cards: if faces come out black, set `FACEFORGE_FP32=1` before launching.

> Get consent before using a real person's face, and label shared output as a deepfake.
