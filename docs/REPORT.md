# FaceForge audit and cross-platform plan

Audit date: 2026-10-24. Branch: `feature/windows-build` (local only, not pushed).

Every finding carries one of three labels:

- **[Verified]**: checked directly in source, binaries, official docs, or by running it in this session.
- **[Estimate]**: reasoned from evidence, not measured.
- **[Unresolved]**: could not be determined here; the reason is stated.

The test machine for every measurement is a Linux sandbox: 2 vCPU Intel Xeon @ 2.80 GHz, no GPU, Python 3.11.17 / 3.13. **No GPU, Windows or iPhone numbers were measured.** Those need your hardware.

---

## 1. Public-source baseline

| Item | Finding |
|---|---|
| Upstream | `hacksider/Deep-Live-Cam`, branch `main` |
| Latest public commit | **`f8f9b85275b06edc039a22d17867751e9f20933b`** (2026-10-22, "Update version from 2.7.5 to 2.8 in README") **[Verified]** |
| Version the free source reports | README heading **2.1.6**; `modules/metadata.py` **2.1.5** **[Verified]** |
| UI toolkit | PySide6 (Qt) since commit `81a1986` (2026-05-15); customtkinter before that **[Verified]** |
| History | 667 commits on `main` **[Verified]** |

**Release tags vs. the paid product.** **[Verified]**
- Every release tag points to a commit that is already on public `main`, including `2.7-ultimate`, `2.7.5-Ultimate` and `2.8`.
- The source at those tags still reports 2.0.3c or 2.1.5.
- The releases page says the Ultimate / Quickstart builds are available "exclusively" from deeplivecam.net.
- The README advertises "Pre-built Deep-Live-Cam 2.8 Ultimate" with "30+ exclusive features".

**Conclusion:** the newest version numbers belong to a pre-built product whose extra features are not in the public source. I did not download, buy or inspect the Ultimate build. Faceforge is built only from the public source at the commit above, pinned in `patches/UPSTREAM.lock`.

**Correction to my earlier message.** I said "upstream v2.8 switched to PySide6". The PySide6 switch is real and in the public source, but that source calls itself 2.1.6, not 2.8.

### Community forks

| Project | What it is | Status |
|---|---|---|
| `iikuzmychov/deep-live-cam-builds` | A packaging workflow only, with no app-code changes. Latest release was built from `f8f9b85` on Oct 23; repo HEAD is `905d4f4` (Oct 25). | **[Verified]** |
| Its packaging issues (still present at `905d4f4`) | 1. The fp16 swap model and CUDA graphs are switched on only when PyTorch is installed, which the builds remove (verified in source). 2. The OpenVINO build installs `onnxruntime-openvino` without the `openvino` runtime; the Windows wheel has no `openvino.dll` (verified by listing the wheel), so that backend cannot load (strong static evidence, not run on Windows). 3. Dead Tk and TensorFlow config (bloat, not breakage). | **[Verified]** / **[Verified, static]** |
| Other GitHub forks | Not assessed: the fork list and other fork pages are outside this session's network allowlist. | **[Unresolved]** |

### Hugging Face model mirror

`huggingface.co/hacksider/deep-live-cam` declares **GPL-3.0** and has an empty README **[Verified]**. It re-hosts third-party weights, including InsightFace's inswapper. InsightFace states its models are for non-commercial research only. A mirror's label cannot change the original publisher's terms, so Faceforge treats the publisher's terms as governing **[Estimate; not legal advice]**.

---

## 2. FaceForge repository audit

| Item | State |
|---|---|
| `github.com/anch-dev/Faceforge` | Public, **empty**: no branches, no commits, no Actions runs **[Verified]** |
| Local work | Branch `feature/windows-build`: `0b339f5` (initial Windows build) plus this session's commit. No `main` or release branches exist. Nothing is pushed. |
| Tests before this session | None |
| Tests now | 55 tests (see section 7) |

---

## 3. Model and licence evaluation

Code and weight licences are listed separately. "FF list" means FaceFusion 3.9.1's per-model licence registry (`facefusion/processors/modules/*/core.py`), cross-checked against publisher repos where possible.

### Face replacement (swap)

| Model | Weights licence | Source / maintenance | Input | Evidence |
|---|---|---|---|---|
| **inswapper_128** (+fp16) | **Non-commercial research only** (InsightFace README). The README marks the model "no longer maintained" and says licensing is via contact@insightface.ai. **[Verified]** | Official GitHub release `model-zoo`; 554,253,681 bytes **[Verified]** | 128x128; source = ArcFace w600k_r50 embedding x `emap` | See measurements below |
| inswapper-512-live | Academic or personal testing only **[Verified]** | **No weights published**; the repo is README-only, for InsightFace's Picsi apps **[Verified]** | 512x512; vendor figures 9.5M params, 12.1 GFLOPs vs inswapper_128's 174.7 GFLOPs **[Verified, vendor numbers]** | Not usable |
| **HyperSwap 1a/1b/1c (256)** | "ResearchRAIL-MS" per the repo badge and FF list. The repo's LICENSE file contains **only the licence name, no terms**. RAIL's naming scheme reserves "Research" variants for research-only use. **[Verified that the text is missing]** | Training code in `facefusion/facefusion-labs` (last commit 2025-10-10); weights 403 MB each on HF `facefusion/models-3.3.0` | 256x256; source = ArcFace w600k embedding (verified in FF code); outputs image + mask | **[Unresolved]**: weights unreachable from this sandbox; adapter written but untested on real weights |
| ReSwapper | Code AGPL-3.0 **[Verified]**. Weight licence **[Unresolved]** (HF page unreachable). Trained on inswapper outputs, InsightFace ArcFace and FFHQ, so the weights inherit their encumbrances **[Estimate]**. | `somanchiu/ReSwapper`, last commit 2025-06-14 | 128/256 | Not tested |
| SimSwap 256/512 | CC BY-NC 4.0 **[Verified]** | `neuralchen/SimSwap`, last commit 2024-08 | 256/512 | Not tested |
| GHOST 1/2/3 | Apache-2.0 **[Verified]**, but still conditioned on InsightFace ArcFace through a converter **[Verified in FF code]** | `ai-forever/ghost`, last commit 2025-02 | 256 | Not tested |
| BlendSwap, AlphaFace | Non-commercial (FF list) | via FaceFusion | 256 | Not tested |
| UniFace, HiFiFace (unofficial) | "None" / "Unknown" (FF list) | via FaceFusion | 256 | Not tested |

**Key structural finding [Verified in FaceFusion source].** Every embedding-conditioned swapper takes its source identity from InsightFace's non-commercial ArcFace `w600k_r50`:
- HyperSwap, inswapper, GHOST (via a converter), SimSwap and AlphaFace.

BlendSwap and UniFace take a source image instead, but their own weights are non-commercial or unlicensed. So **no swap pipeline found in this ecosystem is cleared for commercial use.**

### Detection, recognition, restoration

| Component | Model | Weights licence | Notes |
|---|---|---|---|
| Detection | SCRFD `det_10g` | Non-commercial (InsightFace) **[Verified]** | Used by Deep-Live-Cam |
| Detection | YuNet (OpenCV Zoo) | MIT **[Verified]** | Permissive. Weights are Git LFS and were not retrievable here. |
| Detection (iOS) | Apple Vision face landmarks | Part of the OS | No weights to license. Its 5-point landmarks need mapping; alignment quality is **[Unresolved]**. |
| Recognition | ArcFace `w600k_r50` | Non-commercial **[Verified]** | Required by swappers (above) |
| Recognition (judge only) | `glintr100` (antelopev2) | Non-commercial | Used only to score quality |
| Recognition | SFace (OpenCV Zoo) | Apache-2.0 **[Verified]** | Different 128-d embedding space, so **not** a drop-in for swappers |
| Restoration | GFPGAN | Apache-2.0 code with third-party exceptions **[Verified]**. ONNX weights come via the Deep-Live-Cam mirror. | FFHQ-trained; dataset terms not assessed |
| Restoration | GPEN-BFR 256/512 | Non-commercial (FF list) | |
| Restoration | CodeFormer | S-Lab 1.0, non-commercial **[Verified]** | |
| Restoration | RestoreFormer++ | Apache-2.0 (FF list) | |

### Measurements (sandbox CPU, real InsightFace weights) [Verified]

My reference pipeline (`faceforge_eval/`) was checked against insightface 0.7.3 on its own demo image (`t1.jpg`, 6 faces):

| Check | Result |
|---|---|
| Detections, boxes, landmarks, scores | Identical |
| Embeddings | cosine >= 0.9999998 |
| Swap output | max 1 gray level, mean 0.004 |

Performance and quality:

| Model / config | p50 latency | Throughput | Peak RAM | Quality |
|---|---|---|---|---|
| det_10g @320 | 50 ms | | | |
| w600k_r50 | 205 ms | | | |
| 2d106det | 7 ms | | | |
| inswapper_128 fp32, CPU EP, full pipeline | 2,020 ms/frame | **0.50 FPS** (1%-low 0.47) | 2.0 GB | Identity sim **0.848** (min 0.740) vs 0.022 unswapped |
| inswapper_128 fp16 (made locally), CPU EP | 2,017 ms/frame | 0.50 FPS | 2.1 GB | **0.848**, identical; output cosine 0.999994 vs fp32 |
| inswapper_128 fp32, OpenVINO EP (CPU device) | 1,847 ms/frame | 0.54 FPS | 2.1 GB | (not rerun) |

About these numbers:
- **Quality metric.** Identity similarity is the cosine between the swapped face and the source face, scored by `glintr100`, a different model from the one that conditions the swap. It averages 30 source-target pairs. Because it's the same vendor, it is still not a fully independent judge.
- **Startup.** Sessions took 3.9–4.9 s to create.
- **Not measured here.** GPU FPS, VRAM and camera-to-screen latency. `tools/eval_pipeline.py` reports all of them, plus VRAM through nvidia-smi on NVIDIA cards.

**Recommendation.**
1. **Keep inswapper_128 as the default.** It's the only swapper whose weights I could obtain from the publisher and verify end to end. Use it under InsightFace's non-commercial terms.
2. **Treat HyperSwap as the quality-upgrade candidate**, but only after the actual ResearchRAIL-MS terms are obtained from FaceFusion and its output is checked with `tools/eval_pipeline.py --swapper hyperswap:…`.
3. **Do not ship any weights** with Faceforge in either case.

---

## 4. Platform compatibility matrix

### Windows backends

| Backend | Package (pinned) | Runtime requirement | Status |
|---|---|---|---|
| **CUDA** | onnxruntime-gpu 1.26.0 | DLLs from its import table: `cudart64_12`, `cublas64_12`, `cublasLt64_12`, `cufft64_11`, `cudnn64_9`, so CUDA 12.x + cuDNN 9 **[Verified]** | CI checks that missing DLLs are named. GPU run **[Unresolved: no NVIDIA hardware]** |
| TensorRT | provider DLL ships inside onnxruntime-gpu 1.26.0 | also `nvinfer_10`, `nvonnxparser_10` (TensorRT 10.x) **[Verified]** | Not packaged: users would install TensorRT separately and wait for engine builds. Phase 2 candidate. |
| **DirectML** | onnxruntime-directml 1.24.4 | `DirectML.dll` 1.15.4 bundled in the wheel **[Verified]** | GPU run **[Unresolved]**. The CI self-test is advisory only, since runners have a software adapter. |
| **OpenVINO** | onnxruntime-openvino 1.24.1 + openvino 2025.4.1 | The Windows wheel lacks `openvino.dll`; the pinned runtime is build 2025.4.1.20426, an exact match to what the provider expects **[Verified]**. | Provider loads and runs on the CPU device (Linux) **[Verified]**. See the two notes below. Intel GPU **[Unresolved]**. |
| CPU | any | none | 0.5 FPS measured: not usable for live use **[Verified]** |

**OpenVINO note 1 (device string) [Verified].** Upstream's `device_type: AUTO:GPU,NPU,CPU` makes the provider *fail to initialise* on a machine without those devices, and ONNX Runtime silently uses CPU instead. Plain `AUTO` works, so Faceforge patches it. Whether an Intel GPU machine *without* an NPU hits the same failure is **[Unresolved]**.

**OpenVINO note 2 (Linux only) [Verified].** On Linux the onnxruntime-openvino wheel bundles OpenVINO, and loading the separate `openvino` package alongside it causes a symbol clash. Windows differs because the wheel has no DLL there.

**OpenVINO compiled-model cache [Verified on CPU].** It gave no measurable startup gain on the CPU device, so it isn't enabled. Its effect on an Intel GPU is **[Unresolved]**.

### iOS

| Mode | Feasibility | Basis |
|---|---|---|
| Still image | **Feasible** | All needed models convert to Core ML ML Program (fp16, iOS 17): det_10g 8.2 MB, w600k_r50 84 MB, inswapper_128 264 MB. The pre-conversion intermediate matches ONNX Runtime to <= 8e-7. **[Verified: conversion only]** Core ML prediction parity on device is **[Unresolved: needs macOS/iPhone]**. |
| Prerecorded video | **Feasible**, offline | Same models, with no real-time deadline **[Estimate]** |
| Live camera | **Uncertain** | inswapper_128 costs 174.7 GFLOPs per face per frame (InsightFace figure). InsightFace built a separate 12.1 GFLOPs model for its own iPhone live app. My guess for recent iPhones is roughly 5–15 FPS before thermal throttling **[Estimate, wide error bars]**; only an on-device benchmark settles it. |
| Showing up as a camera in other apps | **Not supported** | Core Media I/O camera extensions are macOS 12.3+ / Mac Catalyst only; Apple's docs list no iOS or iPadOS availability **[Verified]**. No supported path was found. |
| Feeding a virtual microphone to other apps | **Not supported** | No supported iOS path found. Audio Unit extensions run only inside apps that host them **[Estimate from platform knowledge; not re-verified this session]**. |
| Camera while multitasking | iPad only, with an entitlement | `isMultitaskingCameraAccessEnabled` (iOS/iPadOS 16+) requires `com.apple.developer.avfoundation.multitasking-camera-access`, and support varies by device **[Verified]**. Whether a free SideStore signing can carry it is **[Unresolved]**. |

Why use native Core ML rather than ONNX Runtime's Core ML provider? ONNX Runtime's ML Program operator table does not list **Pad** or **BatchNormalization** **[Verified]**. inswapper has 14 reflect-mode Pads **[Verified]**, so that route would likely split the model between CPU and Neural Engine. Upstream Deep-Live-Cam itself rewrites Pad for exactly this reason on Apple Silicon. Native conversion handled both operators.

---

## 5. iOS plan (Swift + Core ML)

Constraints: no Mac, so builds go through GitHub Actions macOS runners as an unsigned IPA for SideStore. Faceforge does not ship weights.

1. **Model delivery.** You convert your own copies on your PC with `tools/convert_coreml.py` and import the `.mlpackage` files through the Files app. The app compiles them on device with `MLModel.compileModel(at:)`.
2. **Pipeline.** It mirrors `faceforge_eval` stage for stage: Detector → Embedder (once per source) → Swapper → Compositor. The Python version is the golden reference.
3. **Camera.** `AVCaptureSession` delivering BGRA at 640x480 or 720p, with `alwaysDiscardsLateVideoFrames` on, so only the newest frame is processed. Detection runs every N frames, as in Deep-Live-Cam; compositing uses Core Image or Metal.
4. **Thermal.** Watch `ProcessInfo.thermalState` (iOS 11+ **[Verified]**):
   - `serious`: cap FPS and detect less often.
   - `critical`: pause.
5. **Memory.** About 360 MB of fp16 weights plus activations. Load the swapper lazily and release it when backgrounded. iOS memory limits for a sideloaded app are **[Unresolved]**.
6. **Output.** Recording inside the app and export through the share sheet. No claims of a system-wide virtual camera.

---

## 6. Voice-processing feasibility

The voice side is a separate module, `faceforge_audio/`. It has a streaming contract, reference processors, and a harness that measures real latency, real-time factor and underruns, with 10 tests of its own. No voice model is included yet.

| Candidate | Code licence | Maintenance | Claimed latency (their README) | Notes |
|---|---|---|---|---|
| RVC WebUI | MIT **[Verified]** | Active (2026-08) | ~170 ms end to end, ~90 ms with ASIO | Needs a per-voice trained model. Base-weight licences **[Unresolved]**. |
| w-okada VCClient | MIT **[Verified]** | Active (2026-09) | n/a (host app) | Runs RVC and Beatrice v2. Windows needs a virtual audio driver for other apps to hear the output. |
| Seed-VC | GPL-3.0 **[Verified]** | Last commit 2025-04 | ~300 ms algorithmic + ~100 ms device | Zero-shot cloning from a short clip, the highest impersonation risk |
| LLVC | MIT **[Verified]** | Stale (2023) | Designed for CPU low latency (paper) | Research code |
| OpenVoice | MIT **[Verified]** | Last commit 2025-04 | Not streaming-oriented | TTS plus tone conversion, not live voice conversion |
| so-vits-svc | AGPL-3.0 **[Verified]** | Stale (2023) | n/a | Not recommended |

All the latency figures are vendor claims, not measurements **[Estimate]**. Recommendation: evaluate **RVC through VCClient** on Windows first, limited to your own voice or voices whose owners consent. Use `faceforge_audio.stream_benchmark` to measure it, and don't pick an iOS voice model until the Windows numbers exist.

---

## 7. Implementation plan and acceptance criteria

| Phase | Work | Acceptance criteria |
|---|---|---|
| **0 (this branch)** | Patches, runtime helpers, eval and audio harnesses, CI, docs | All tests pass on Python 3.11 **[met]**. Release zips contain no `.onnx` (CI guard). Builds are manual, and releases are drafts by default. |
| 1. Windows on real hardware | Push after your approval, run the workflow, test each zip on your PC | For each backend you have: all 3 models report the requested provider. `eval_pipeline.py` on your GPU reports avg FPS, 1%-low, p95 latency and peak VRAM; target **>= 20 FPS avg, >= 12 FPS 1%-low at 640x480** on a mid-range GPU **[target, not a measurement]**. First-run download plus fp16 conversion completes with the licence notice shown. |
| 2. HyperSwap evaluation | Get the ResearchRAIL-MS terms; if acceptable, test the adapter | Output matches FaceFusion within 2/255 mean on aligned crops. Identity similarity >= 0.848 (the inswapper figure) on the same image set. Latency reported per backend. |
| 3. iOS still image | Swift app with user-imported Core ML models | Aligned-crop output within mean 2/255 of the Python reference. Single-image latency measured on your iPhone. |
| 4. iOS video file | Offline processing with progress and export | Processes a 10 s 720p clip without exceeding memory limits; time per frame reported. |
| 5. iOS live camera | Camera pipeline with thermal governor | 10 minutes sustained without reaching `critical`; reports median and 1%-low FPS. Proceed only if median >= 10 FPS. |
| 6. Voice | RVC/VCClient evaluation, then integration | Real-time factor < 0.5. Pipeline latency (block + algorithmic + p95 processing) <= 150 ms on your PC. Zero underruns over 5 minutes. |

---

## 8. Tests, files, and rollback

### Tests run [Verified]

- **Python 3.11.17, pinned `requirements-dev.txt`:** 54 passed, 1 skipped. The skip is the Core ML test, which needs torch and coremltools; it passed separately in the conversion environment.
- **Python 3.13 sandbox:** 44 passed with the pinned upstream and real weights (before the audio tests were added); 37 passed and 7 skipped without them.
- **Workflows:** pass `actionlint` 1.7.7.

Coverage includes:
- Patch application against the pinned upstream: dry run, apply, idempotence, drift with no partial writes, compile.
- fp16 conversion fidelity on a synthetic model, and on the real model when available.
- Provider-fallback detection.
- Detector decoding and alignment geometry, including a FaceFusion template cross-check.
- Pipeline caching, FPS and 1%-low math.
- Tool command-line interfaces, the runtime hook self-test, and workflow policy (manual builds, draft releases, no weights).
- Audio latency measurement.

Also run by hand, outside pytest: patched upstream Deep-Live-Cam code with real models and the reference insightface. Results:
- The fp16 copy is derived locally.
- The CPU-fallback banner fires.
- The face-analyser providers are reported.
- The third-party fp16 mirror is blocked.
- `buffalo_l` defers to InsightFace's release.

**Not run:** the Windows build itself, any GPU, macOS, or an iPhone. The CI workflow has not executed because nothing is pushed.

### Files

Added in this session:
- **Patches:** `patches/faceforge_runtime.py`, `patches/UPSTREAM.lock`
- **Eval package:** `faceforge_eval/` (9 files)
- **Audio package:** `faceforge_audio/` (3 files)
- **Tools:** `tools/bench_models.py`, `tools/eval_pipeline.py`, `tools/convert_fp16.py`, `tools/convert_coreml.py`
- **Tests:** `tests/` (8 files)
- **CI:** `.github/workflows/tests.yml`, `.github/workflows/upstream-drift.yml`
- **Other:** `requirements-dev.txt`, `docs/REPORT.md`

Changed in this session:
- `patches/apply_faceforge.py` (12 patches, dry-run gate)
- `build/rthook_faceforge.py` (CUDA DLL preflight, synthetic self-test)
- `build/faceforge.spec`
- `.github/workflows/build-windows.yml` (pinned commit, no weights, manual-only, draft releases)
- `docs/release-notes-windows.md`, `docs/README-windows.txt`, `README.md`

### Rollback

Nothing has been pushed, so all of this lives only in the local clone:
- To discard this session's work but keep the earlier commit: `git reset --hard 0b339f5` on `feature/windows-build`.
- To discard everything: `git branch -D feature/windows-build`.

The remote `anch-dev/Faceforge` is untouched, and no release or release branch exists.

## 9. Open questions for you

1. **Hardware.** Which GPU is in your Windows PC, and which iPhone model do you have? These decide which backends and targets get tested first.
2. **Commercial use.** Is any use commercial? If so, every swap pipeline evaluated here is blocked by its weight licence.
3. **HyperSwap terms.** Do you want me to pursue the full ResearchRAIL-MS text from FaceFusion?
