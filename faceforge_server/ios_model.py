"""Make the iPhone's fast (Neural Engine) swap model from YOUR copy of
inswapper_128 and hand it to the Faceforge app over your Wi-Fi.

    python -m faceforge_server.ios_model --accept-model-license

What it makes (from the publisher's fp32 file, on this PC; nothing is uploaded):
  * inswapper_128_fp16.onnx           - always. Half-precision weights, float32
    inputs/outputs. On the iPhone, ONNX Runtime's Core ML backend then builds an
    fp16 ML Program, which the Neural Engine can run (an fp32 one cannot).
  * inswapper_128_fp16.mlpackage.zip  - only where coremltools installs (Linux,
    macOS, WSL; it has no Windows wheels). Native Core ML, usually the fastest.

Then it serves those files on port 47801 until you close the window. In the app:
Models > Fast swap model > Get from PC, with the code printed below. Only
requests that carry that code are answered.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import secrets
import shutil
import sys
import zipfile
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "patches"))
sys.path.insert(0, os.path.join(REPO, "tools"))

DEFAULT_PORT = 47801
FP16_ONNX = "inswapper_128_fp16.onnx"
MLPACKAGE = "inswapper_128_fp16.mlpackage"
MLPACKAGE_ZIP = MLPACKAGE + ".zip"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_dir(src_dir: str, dst_zip: str) -> None:
    """Zip a directory (an .mlpackage) with the directory itself as the top
    entry, written atomically. Weights barely compress, so entries are stored."""
    base = os.path.dirname(os.path.abspath(src_dir))
    tmp = dst_zip + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for root, _dirs, files in os.walk(src_dir):
            for name in sorted(files):
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, base).replace(os.sep, "/"))
    os.replace(tmp, dst_zip)


def make_fp16_onnx(fp32: str, out_dir: str, printer=print) -> Optional[str]:
    from faceforge_runtime import ensure_local_fp16

    dst = os.path.join(out_dir, FP16_ONNX)
    return dst if ensure_local_fp16(fp32, dst, lambda: False, printer) else None


def make_mlpackage(fp32: str, out_dir: str, printer=print) -> Optional[str]:
    """Native Core ML fp16 package, zipped. None if coremltools is unavailable
    (always the case on Windows) or conversion fails."""
    dst_zip = os.path.join(out_dir, MLPACKAGE_ZIP)
    if os.path.exists(dst_zip):
        return dst_zip
    try:
        import coremltools  # noqa: F401
        from convert_coreml import convert
    except Exception as e:  # ImportError, or coremltools' native libs missing
        printer(f"[ios-model] Native Core ML package skipped (coremltools not available here: {e}).")
        printer("[ios-model]   The fp16 ONNX below still runs on the Neural Engine via ONNX Runtime.")
        return None
    pkg = os.path.join(out_dir, MLPACKAGE)
    printer("[ios-model] Converting to a native Core ML package (fp16). Takes a couple of minutes...")
    try:
        if os.path.exists(pkg):
            shutil.rmtree(pkg)
        info = convert(fp32, pkg, [(1, 3, 128, 128), (1, 512)], "fp16", "iOS17")
        printer(f"[ios-model]   PyTorch intermediate vs ONNX Runtime max |diff|: {info['intermediate_vs_ort_max_abs_diff']:.2e}")
        zip_dir(pkg, dst_zip)
        shutil.rmtree(pkg, ignore_errors=True)
        return dst_zip
    except Exception as e:
        printer(f"[ios-model] Core ML conversion failed ({e}); serving the fp16 ONNX only.")
        return None


def build_index(paths: List[str]) -> Dict:
    """What the app downloads: name, size, sha256 per file, best first."""
    files = []
    for p in paths:
        files.append({"name": os.path.basename(p), "size": os.path.getsize(p), "sha256": sha256_file(p),
                      "kind": "mlpackage-zip" if p.endswith(".zip") else "onnx-fp16"})
    files.sort(key=lambda f: 0 if f["kind"] == "mlpackage-zip" else 1)
    return {"version": 1, "files": files}


def resolve_request(path: str, code: str, files: Dict[str, str]) -> Optional[str]:
    """Map a request path to something to send: 'index' or a file path.
    Anything without the exact code, or not one of `files`, is None (404)."""
    parts = [p for p in path.split("?", 1)[0].split("/") if p]
    if len(parts) != 2 or not secrets.compare_digest(parts[0], code):
        return None
    if parts[1] == "index.json":
        return "index"
    return files.get(parts[1])


def make_handler(code: str, files: Dict[str, str], index: Dict, printer=print):
    index_bytes = json.dumps(index).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "FaceforgeModel/1"

        def log_message(self, fmt, *args):  # quiet default access log
            pass

        def do_GET(self):  # noqa: N802
            target = resolve_request(self.path, code, files)
            if target is None:
                self.send_error(404)
                return
            if target == "index":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(index_bytes)))
                self.end_headers()
                self.wfile.write(index_bytes)
                return
            size = os.path.getsize(target)
            printer(f"[ios-model] {self.client_address[0]} is downloading {os.path.basename(target)} "
                    f"({size / 1e6:.0f} MB)...")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            try:
                with open(target, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, 1 << 20)
                printer(f"[ios-model]   sent {os.path.basename(target)}")
            except (BrokenPipeError, ConnectionResetError):
                printer("[ios-model]   download cancelled by the phone")

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=os.path.join(REPO, "models"))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--code", help="download code (default: random 6 digits)")
    ap.add_argument("--no-serve", action="store_true", help="only make the files (in <models-dir>/ios)")
    ap.add_argument("--no-mlpackage", action="store_true", help="skip the native Core ML package")
    ap.add_argument("--accept-model-license", action="store_true",
                    help="allow downloading InsightFace models (non-commercial research licence)")
    args = ap.parse_args(argv)

    from faceforge_eval.models import DownloadError, ensure_models
    try:
        paths = ensure_models(args.models_dir, args.accept_model_license)
    except DownloadError as e:
        print(f"[ios-model] {e}")
        return 2
    out_dir = os.path.join(args.models_dir, "ios")
    os.makedirs(out_dir, exist_ok=True)

    made = []
    onnx16 = make_fp16_onnx(paths["swapper"], out_dir)
    if onnx16:
        made.append(onnx16)
    if not args.no_mlpackage:
        pkg = make_mlpackage(paths["swapper"], out_dir)
        if pkg:
            made.append(pkg)
    if not made:
        print("[ios-model] Nothing to serve (conversion failed).")
        return 1
    print("[ios-model] Ready: " + ", ".join(os.path.basename(p) for p in made) + f"  (in {out_dir})")
    if args.no_serve:
        return 0

    from .server import lan_addresses
    code = args.code or f"{secrets.randbelow(10**6):06d}"
    index = build_index(made)
    files = {os.path.basename(p): p for p in made}
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), make_handler(code, files, index))
    print("=" * 60)
    print(" Faceforge: fast iPhone model is ready to send")
    for a in lan_addresses() or ["(no LAN address found)"]:
        print(f"   PC address:    {a}")
    print(f"   Download code: {code}")
    print(" In the app: Models > Fast swap model > Get from PC.")
    print(" Keep this window open until the phone says it's installed.")
    print("=" * 60, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
