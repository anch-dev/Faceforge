# Faceforge

Performance-tuned builds of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam), the real-time face-swap tool. This is an unofficial community build, not affiliated with the Deep-Live-Cam developers.

> Get consent before using a real person's face, and label anything you share as a deepfake. You are responsible for how you use it.

## Download

Grab a zip from [Releases](../../releases/latest). The release notes explain which build to pick (cuda, directml or openvino).

## What Faceforge changes

Builds come from upstream source with these fixes applied on top (`patches/apply_faceforge.py`):

| Problem in stock portable builds | Faceforge fix |
|---|---|
| The fast fp16 swap model and CUDA graphs only switch on when PyTorch is installed, and portable builds ship without it, so every GPU ran the slower path | Chosen from the GPU backend actually in use |
| OpenVINO build shipped without the OpenVINO runtime DLLs, so it silently ran on CPU | Bundles `openvino` 2025.4.1 and registers its DLLs at startup |
| Silent CPU fallback when CUDA/cuDNN are missing | Console banner plus status message: **GPU NOT IN USE** |
| Packaging still targeted the old Tk UI (Deep-Live-Cam 2.8 moved to PySide6) | New PyInstaller spec for PySide6; drops TensorFlow, Torch and Tk |
| 640px face detection by default | 320px default for live (still changeable in the UI) |

CI also runs a self-test that loads the swap model on the build's GPU backend and fails if it lands on CPU.

## Build it yourself

Actions → **Build Faceforge (Windows)** → Run workflow. Leave `ref` blank for the latest upstream. A scheduled run checks upstream daily and builds only when there's a new commit.

## License

AGPL-3.0, same as upstream.
