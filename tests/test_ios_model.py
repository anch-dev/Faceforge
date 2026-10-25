"""The PC -> iPhone fast-model hand-off (faceforge_server.ios_model)."""
import hashlib
import http.server
import json
import threading
import urllib.error
import urllib.request
import zipfile

import pytest

from faceforge_server import ios_model as IM


def test_resolve_request_requires_exact_code_and_known_file():
    files = {"inswapper_128_fp16.onnx": "/x/a.onnx"}
    assert IM.resolve_request("/123456/index.json", "123456", files) == "index"
    assert IM.resolve_request("/123456/inswapper_128_fp16.onnx", "123456", files) == "/x/a.onnx"
    assert IM.resolve_request("/123456/inswapper_128_fp16.onnx?x=1", "123456", files) == "/x/a.onnx"
    for bad in ("/000000/index.json", "/index.json", "/123456/../etc/passwd", "/123456/other.onnx",
                "/123456/a/b", "/", "/1234567/index.json"):
        assert IM.resolve_request(bad, "123456", files) is None, bad


def test_zip_dir_keeps_package_as_top_entry(tmp_path):
    pkg = tmp_path / "m.mlpackage"
    (pkg / "Data" / "w").mkdir(parents=True)
    (pkg / "Manifest.json").write_text("{}")
    (pkg / "Data" / "w" / "weight.bin").write_bytes(b"\0" * 10)
    out = tmp_path / "m.mlpackage.zip"
    IM.zip_dir(str(pkg), str(out))
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["m.mlpackage/Data/w/weight.bin", "m.mlpackage/Manifest.json"]
    assert not (tmp_path / "m.mlpackage.zip.part").exists()


def test_index_lists_native_package_first_with_hashes(tmp_path):
    a = tmp_path / IM.FP16_ONNX
    b = tmp_path / IM.MLPACKAGE_ZIP
    a.write_bytes(b"onnx")
    b.write_bytes(b"zip!")
    idx = IM.build_index([str(a), str(b)])
    assert [f["kind"] for f in idx["files"]] == ["mlpackage-zip", "onnx-fp16"]
    assert idx["files"][1] == {"name": IM.FP16_ONNX, "size": 4, "sha256": hashlib.sha256(b"onnx").hexdigest(),
                               "kind": "onnx-fp16"}


def test_http_serves_only_with_code(tmp_path):
    f = tmp_path / IM.FP16_ONNX
    f.write_bytes(b"x" * 3000)
    files = {f.name: str(f)}
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), IM.make_handler("424242", files, IM.build_index([str(f)]),
                                                                              printer=lambda *_: None))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        idx = json.loads(urllib.request.urlopen(f"{base}/424242/index.json", timeout=5).read())
        assert idx["files"][0]["name"] == IM.FP16_ONNX
        assert urllib.request.urlopen(f"{base}/424242/{IM.FP16_ONNX}", timeout=5).read() == b"x" * 3000
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/111111/{IM.FP16_ONNX}", timeout=5)
    finally:
        httpd.shutdown()
        httpd.server_close()
