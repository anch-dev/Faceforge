"""Faceforge patches applied on top of upstream Deep-Live-Cam at build time.

Usage:
    python apply_faceforge.py <upstream-checkout>           apply
    python apply_faceforge.py <upstream-checkout> --check   report only, write nothing

Every patch is an exact text replacement. If upstream changes the code a patch
targets, the script exits non-zero so the build fails loudly instead of
silently shipping an unpatched app. Re-running is safe (already-applied
patches are skipped).

Tested against the upstream commit recorded in patches/UPSTREAM.lock.
"""
from __future__ import annotations

import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent


class Patcher:
    def __init__(self, root: pathlib.Path, check_only: bool = False):
        self.root = root
        self.check_only = check_only
        self.failures: list[str] = []
        self.applied: list[str] = []
        self.skipped: list[str] = []

    def replace(self, rel: str, old: str, new: str, label: str) -> None:
        path = self.root / rel
        if not path.exists():
            self.failures.append(f"{label}: {rel} does not exist")
            print(f"[FAIL] {label} (missing file)")
            return
        text = path.read_text(encoding="utf-8")
        if new in text:
            self.skipped.append(label)
            print(f"[skip] {label} (already applied)")
            return
        count = text.count(old)
        if count != 1:
            self.failures.append(f"{label}: expected 1 match in {rel}, found {count}")
            print(f"[FAIL] {label}")
            return
        if not self.check_only:
            path.write_text(text.replace(old, new), encoding="utf-8")
        self.applied.append(label)
        print(f"[ok]   {label}")

    def copy(self, src: pathlib.Path, rel: str, label: str) -> None:
        dst = self.root / rel
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            self.skipped.append(label)
            print(f"[skip] {label} (already applied)")
            return
        if not dst.parent.exists():
            self.failures.append(f"{label}: target folder {dst.parent} missing")
            print(f"[FAIL] {label}")
            return
        if not self.check_only:
            shutil.copyfile(src, dst)
        self.applied.append(label)
        print(f"[ok]   {label}")


def apply_all(p: Patcher) -> None:
    # 0. Faceforge helper module (licence notices, fp16 conversion, provider reports).
    p.copy(HERE / "faceforge_runtime.py", "modules/faceforge_runtime.py", "add modules/faceforge_runtime.py")

    # 1. Frozen-app exit: quit() is a site builtin that doesn't exist under PyInstaller.
    p.replace("modules/core.py",
              "    if to_quit:\n        quit()",
              "    if to_quit:\n        sys.exit()",
              "core: sys.exit instead of quit()")

    # 2. Live-friendly default detection size (still selectable in the UI;
    #    a saved UI choice overrides it).
    p.replace("modules/globals.py",
              "DEFAULT_DET_SIZE: int = 640",
              "DEFAULT_DET_SIZE: int = 320  # Faceforge: 320 for live speed (was 640)",
              "globals: default det_size 320")

    # 3. fp16 swap model and CUDA graphs chosen by the ONNX Runtime provider in
    #    use, not by PyTorch being installed (portable builds ship without torch,
    #    so upstream silently took the slower fp32 path on every GPU). The fp16
    #    file is derived locally from the publisher's fp32 file.
    p.replace("modules/processors/frame/face_swapper.py",
              "            use_fp16 = _HAS_TORCH_CUDA and os.path.exists(fp16_path)",
              "            from modules import faceforge_runtime as _ffr\n"
              "            use_fp16 = _ffr.prefers_fp16(modules.globals.execution_providers, _HAS_TORCH_CUDA) \\\n"
              "                and _ffr.ensure_local_fp16(fp32_path, fp16_path, pre_check)",
              "face_swapper: fp16 by provider, derived locally")
    p.replace("modules/processors/frame/face_swapper.py",
              "                if _HAS_TORCH_CUDA and any(\n",
              "                if _ffr.cuda_graphs_ok(modules.globals.execution_providers) and any(\n",
              "face_swapper: CUDA graphs without torch")

    # 4. Report the provider each session really runs on; loud banner on CPU fallback.
    p.replace("modules/processors/frame/face_swapper.py",
              "                update_status(\"Face swapper model loaded successfully.\", NAME)",
              "                _ffr.report_session(\"swap model\", FACE_SWAPPER.session,\n"
              "                                    modules.globals.execution_providers,\n"
              "                                    status=lambda m: update_status(m, NAME))\n"
              "                update_status(\"Face swapper model loaded successfully.\", NAME)",
              "face_swapper: provider report")
    p.replace("modules/face_analyser.py",
              "                _optimize_det_model(analyser, providers, det_size)\n",
              "                _optimize_det_model(analyser, providers, det_size)\n"
              "                from modules import faceforge_runtime as _ffr\n"
              "                _ffr.report_face_analyser(analyser, modules.globals.execution_providers)\n",
              "face_analyser: provider report")

    # 5. Model sources: publisher URLs where they exist, never a third-party
    #    fp16 inswapper copy, and a licence notice before every first download.
    p.replace("modules/model_downloader.py",
              "def resolve_url(name: str) -> str:\n",
              "def resolve_url(name: str) -> str:\n"
              "    from modules import faceforge_runtime as _ffr\n"
              "    _publisher = _ffr.publisher_url(name)\n"
              "    if _publisher:\n"
              "        return _publisher\n",
              "model_downloader: publisher URLs")
    p.replace("modules/model_downloader.py",
              "        if _download(name, resolve_url(name), target, expected_size(name)):\n",
              "        from modules import faceforge_runtime as _ffr\n"
              "        if not _ffr.download_allowed(name):\n"
              "            return None\n"
              "        _ffr.license_notice(name, resolve_url(name))\n"
              "        if _download(name, resolve_url(name), target, expected_size(name)):\n",
              "model_downloader: licence notice + blocked mirrors")
    p.replace("modules/model_downloader.py",
              "    members = [n for n in MODEL_SIZES if n.startswith(f\"{name}/\")]\n",
              "    # Faceforge: when the pack is absent, let insightface fetch it from its\n"
              "    # publisher (github.com/deepinsight/insightface releases v0.7).\n"
              "    _ff_dest = os.path.join(os.path.expanduser(\"~\"), \".insightface\", \"models\", name)\n"
              "    if not os.path.isdir(_ff_dest):\n"
              "        from modules import faceforge_runtime as _ffr\n"
              "        _ffr.license_notice(name, \"https://github.com/deepinsight/insightface/releases/download/v0.7/\" + name + \".zip\")\n"
              "        return True\n"
              "    members = [n for n in MODEL_SIZES if n.startswith(f\"{name}/\")]\n",
              "model_downloader: insightface pack from publisher")

    # 6. OpenVINO device selection. Verified: "AUTO:GPU,NPU,CPU" fails to
    #    initialise when listed devices are missing and ORT silently uses CPU.
    p.replace("modules/platform_info.py",
              "    {\"device_type\": \"AUTO:GPU,NPU,CPU\"},",
              "    {\"device_type\": __import__(\"os\").environ.get(\"FACEFORGE_OV_DEVICE\", \"AUTO\")},",
              "platform_info: OpenVINO AUTO device")

    # 7. The NSFW filter needs keras + a deep-learning backend we don't ship.
    p.replace("modules/ui.py",
              "    from numpy import ndarray\n    from modules.predicter import predict_frame, predict_image, predict_video\n",
              "    from numpy import ndarray\n"
              "    try:\n"
              "        from modules.predicter import predict_frame, predict_image, predict_video\n"
              "    except Exception:\n"
              "        update_status(\"NSFW filter is not included in this build; skipping.\")\n"
              "        return False\n",
              "ui: NSFW filter optional")


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    root = pathlib.Path(args[0] if args else ".").resolve()
    check_only = "--check" in argv
    # Always dry-run first, so a mismatch never leaves upstream half-patched.
    dry = Patcher(root, check_only=True)
    if not check_only:
        print("Checking patches...")
    apply_all(dry)
    if dry.failures or check_only:
        p = dry
    else:
        print("\nApplying patches...")
        p = Patcher(root)
        apply_all(p)
    if p.failures:
        print("\nFaceforge patching FAILED - upstream changed:")
        for f in p.failures:
            print("  - " + f)
        return 1
    verb = "would apply" if p.check_only else "applied"
    print(f"\nFaceforge patches OK: {len(p.applied)} {verb}, {len(p.skipped)} already present.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
