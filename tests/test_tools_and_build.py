import json
import os
import subprocess
import sys

import pytest
import yaml

from conftest import REPO


def test_bench_reports_actual_provider(tiny_conv):
    r = subprocess.run([sys.executable, str(REPO / "tools" / "bench_models.py"), "--provider", "cpu",
                        "--runs", "3", tiny_conv], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout.strip().splitlines()[-1])
    assert data["active"] == "CPUExecutionProvider" and not data["fell_back_to_cpu"]
    assert data["p50_ms"] > 0


def test_bench_flags_fallback(tiny_conv):
    import onnxruntime as ort
    if "DmlExecutionProvider" in ort.get_available_providers():
        pytest.skip("DirectML present")
    r = subprocess.run([sys.executable, str(REPO / "tools" / "bench_models.py"), "--provider", "dml",
                        "--runs", "2", tiny_conv], capture_output=True, text=True)
    assert r.returncode == 2
    assert json.loads(r.stdout.strip().splitlines()[-1])["fell_back_to_cpu"] is True


@pytest.mark.parametrize("provider,code", [("cpu", 0), ("cuda", 3)])
def test_rthook_selftest(provider, code):
    import onnxruntime as ort
    if provider == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA present")
    env = dict(os.environ, FACEFORGE_SELFTEST=provider)
    r = subprocess.run([sys.executable, str(REPO / "build" / "rthook_faceforge.py")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == code, r.stdout + r.stderr
    assert "[selftest] requested" in r.stdout


def test_convert_cli(fake_inswapper, tmp_path):
    dst = tmp_path / "o.onnx"
    r = subprocess.run([sys.executable, str(REPO / "tools" / "convert_fp16.py"), fake_inswapper, str(dst)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "emap" in r.stdout and dst.exists()


def load_workflow(name):
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text())


def test_build_workflow_ships_no_weights():
    text = (REPO / ".github" / "workflows" / "build-windows.yml").read_text()
    assert "huggingface.co" not in text and "models/" not in text.split("Download FFmpeg")[0].split("Install dependencies")[1]
    assert "Model weights found in the app folder" in text
    wf = load_workflow("build-windows.yml")
    trig = wf.get("on", wf.get(True))
    assert set(trig) == {"workflow_dispatch"}, "builds must be manual"
    assert trig["workflow_dispatch"]["inputs"]["publish"]["default"] is False


def test_build_workflow_pins():
    env = load_workflow("build-windows.yml")["env"]
    for key in ("ORT_GPU", "ORT_DML", "ORT_OPENVINO", "OPENVINO_RUNTIME", "FP16_CONVERTER", "PYINSTALLER"):
        assert "==" in env[key], key


def test_drift_workflow_never_builds():
    text = (REPO / ".github" / "workflows" / "upstream-drift.yml").read_text()
    assert "--check" in text and "pyinstaller" not in text.lower() and "release" not in text.split("jobs:")[1]


def test_spec_is_valid_python():
    compile((REPO / "build" / "faceforge.spec").read_text(), "faceforge.spec", "exec")


def test_launchers_use_crlf_on_checkout():
    attrs = (REPO / ".gitattributes").read_text()
    assert "*.bat text eol=crlf" in attrs
    cuda = (REPO / "build" / "launchers" / "Faceforge-cuda.bat").read_text()
    assert "--execution-provider cuda" in cuda
    assert "__PROVIDER__" in (REPO / "build" / "launchers" / "Faceforge.bat").read_text()
