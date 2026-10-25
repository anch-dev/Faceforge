"""Where the default models come from, and fetching them on the user's machine.

Faceforge never redistributes weights. These are the publishers' own files;
sizes and SHA-256 were recorded from the official releases on 2026-10-24.
The iOS app uses the same catalog (ios/FaceforgeCore ModelCatalog.swift).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

INSIGHTFACE_LICENSE = ("InsightFace models are released for non-commercial research purposes only "
                       "(https://github.com/deepinsight/insightface#license).")


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int
    sha256: str
    license: str


INSWAPPER = Asset(
    "inswapper_128.onnx",
    "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx",
    554253681, "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af", INSIGHTFACE_LICENSE)

BUFFALO_L = Asset(
    "buffalo_l.zip",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    288621354, "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", INSIGHTFACE_LICENSE)

# Members extracted from buffalo_l.zip (the rest of the pack is not needed).
BUFFALO_MEMBERS = {
    "det_10g.onnx": (16923827, "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
    "w600k_r50.onnx": (174383860, "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
}


class DownloadError(RuntimeError):
    pass


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: str, size: int, sha256: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) == size and sha256_file(path) == sha256


def download(asset: Asset, dest: str, progress: Optional[Callable[[int, int], None]] = None,
             opener: Callable = urllib.request.urlopen) -> str:
    """Download `asset` to `dest` atomically, checking size and SHA-256."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(os.path.abspath(dest)))
    h = hashlib.sha256()
    got = 0
    try:
        with os.fdopen(fd, "wb") as out, opener(asset.url, timeout=60) as resp:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                h.update(block)
                got += len(block)
                if progress:
                    progress(got, asset.size)
        if got != asset.size or h.hexdigest() != asset.sha256:
            raise DownloadError(f"{asset.name}: got {got} bytes sha256 {h.hexdigest()[:12]}..., "
                                f"expected {asset.size} bytes sha256 {asset.sha256[:12]}...")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def _print_progress(done: int, total: int) -> None:
    pct = 100.0 * done / total if total else 0
    sys.stdout.write(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


def ensure_models(models_dir: str, accept_license: bool, printer: Callable[[str], None] = print,
                  opener: Callable = urllib.request.urlopen) -> Dict[str, str]:
    """Return paths of inswapper / det_10g / w600k_r50 in `models_dir`, downloading
    from InsightFace's releases if missing. Downloading requires accept_license."""
    paths = {
        "swapper": os.path.join(models_dir, INSWAPPER.name),
        "detector": os.path.join(models_dir, "buffalo_l", "det_10g.onnx"),
        "recognizer": os.path.join(models_dir, "buffalo_l", "w600k_r50.onnx"),
    }
    need_swapper = not os.path.isfile(paths["swapper"])
    need_pack = not (os.path.isfile(paths["detector"]) and os.path.isfile(paths["recognizer"]))
    if not (need_swapper or need_pack):
        return paths
    if not accept_license:
        raise DownloadError(
            "Models are missing. Faceforge can download them from InsightFace's GitHub releases.\n"
            f"Licence: {INSIGHTFACE_LICENSE}\n"
            "Re-run with --accept-model-license to download, or place the files in "
            f"{models_dir} yourself.")
    printer(f"[models] Licence: {INSIGHTFACE_LICENSE}")
    if need_swapper:
        printer(f"[models] Downloading {INSWAPPER.url}")
        download(INSWAPPER, paths["swapper"], _print_progress, opener)
    if need_pack:
        printer(f"[models] Downloading {BUFFALO_L.url}")
        zip_path = os.path.join(models_dir, BUFFALO_L.name)
        download(BUFFALO_L, zip_path, _print_progress, opener)
        extract_buffalo(zip_path, os.path.join(models_dir, "buffalo_l"))
        os.remove(zip_path)
    return paths


def extract_buffalo(zip_path: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = {os.path.basename(n): n for n in z.namelist()}
        for member, (size, sha) in BUFFALO_MEMBERS.items():
            if member not in names:
                raise DownloadError(f"{member} missing from {zip_path}")
            target = os.path.join(dest_dir, member)
            with z.open(names[member]) as src, open(target + ".part", "wb") as out:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            if not verify(target + ".part", size, sha):
                os.remove(target + ".part")
                raise DownloadError(f"{member} failed size/sha256 check")
            os.replace(target + ".part", target)
