# Faceforge

Performance-tuned Windows builds of the public [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) source, plus shared tooling for an iOS port. This is an unofficial community project, not affiliated with the Deep-Live-Cam developers or their paid "Ultimate" builds.

> Get consent before using a real person's face, and label anything you share as a deepfake. You are responsible for how you use it.

**Status:** pre-release. See [docs/REPORT.md](docs/REPORT.md) for the full audit: baseline, model licences, platform matrix, and plan.

## What Faceforge changes

Builds come from upstream source at the commit pinned in `patches/UPSTREAM.lock`, with the patches in `patches/apply_faceforge.py`:

| Problem in stock portable builds | Faceforge fix |
|---|---|
| The fp16 swap model and CUDA graphs only switch on when PyTorch is installed, and portable builds ship without it | Chosen by the GPU backend actually in use |
| Silent GPU→CPU fallback | Every model reports the backend it really runs on; a **GPU NOT IN USE** banner appears; the CUDA build names any missing NVIDIA DLLs |
| OpenVINO build lacks the OpenVINO runtime on Windows; upstream's `AUTO:GPU,NPU,CPU` device string fails on machines missing those devices | Bundles `openvino` 2025.4.1 (the exact build the provider expects) and uses `AUTO` |
| Model weights bundled or pulled from a third-party mirror | No weights shipped. InsightFace models are fetched from InsightFace's own releases with a licence notice; the fp16 copy is made locally. |
| 640px face detection by default | 320px default for live use (changeable in the UI) |

## Repository layout

| Path | Purpose |
|---|---|
| `patches/` | Upstream patches, the runtime helper module, and the pinned upstream commit |
| `build/` | PyInstaller spec, startup hook (DLL setup, CUDA preflight, backend self-test), launchers |
| `faceforge_eval/` | Reference face pipeline (detect → embed → swap → composite) on ONNX Runtime. It's the shared benchmark for Windows backends and the golden reference for iOS. |
| `faceforge_audio/` | Separate voice-processing contract and latency harness (no voice model yet) |
| `tools/` | `eval_pipeline.py` (FPS, 1%-low, latency, memory, identity quality), `bench_models.py`, `convert_fp16.py`, `convert_coreml.py` |
| `tests/` | `pip install -r requirements-dev.txt && pytest`. Set `FACEFORGE_UPSTREAM` (an upstream checkout) and `FACEFORGE_WEIGHTS` to run the optional tests. |

## Building

Actions → **Build Faceforge (Windows)** → Run workflow. Builds are manual, and releases stay drafts unless "publish" is ticked. A daily job only checks whether the patches still apply to the latest upstream.

## License

AGPL-3.0, same as upstream. Model weights are not part of this repository and carry their own licences. The InsightFace models used by default are for non-commercial research only.
