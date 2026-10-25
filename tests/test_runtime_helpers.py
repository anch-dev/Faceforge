import os
import stat

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import numpy_helper

import faceforge_runtime as ffr


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("FACEFORGE_FP32", raising=False)
    monkeypatch.delenv("FACEFORGE_NO_CUDA_GRAPH", raising=False)
    ffr._noticed.clear()


@pytest.mark.parametrize("providers,torch,expected", [
    (["CPUExecutionProvider"], False, False),
    (["DmlExecutionProvider", "CPUExecutionProvider"], False, True),
    ([("OpenVINOExecutionProvider", {"device_type": "AUTO"})], False, True),
    (["CUDAExecutionProvider"], False, True),
    (["CPUExecutionProvider"], True, True),
])
def test_prefers_fp16(providers, torch, expected):
    assert ffr.prefers_fp16(providers, torch) is expected


def test_fp32_override(monkeypatch):
    monkeypatch.setenv("FACEFORGE_FP32", "1")
    assert ffr.prefers_fp16(["CUDAExecutionProvider"], True) is False


def test_cuda_graphs(monkeypatch):
    assert ffr.cuda_graphs_ok(["CUDAExecutionProvider"])
    assert not ffr.cuda_graphs_ok(["DmlExecutionProvider"])
    monkeypatch.setenv("FACEFORGE_NO_CUDA_GRAPH", "1")
    assert not ffr.cuda_graphs_ok(["CUDAExecutionProvider"])


def test_model_sources():
    assert ffr.publisher_url("inswapper_128.onnx").startswith(
        "https://github.com/deepinsight/insightface/releases/download/")
    assert ffr.publisher_url("GPEN-BFR-256.onnx") is None
    assert ffr.download_allowed("inswapper_128.onnx")
    assert not ffr.download_allowed("inswapper_128_fp16.onnx")


def test_license_notice_once():
    lines = []
    ffr.license_notice("inswapper_128.onnx", "https://x", lines.append)
    ffr.license_notice("inswapper_128.onnx", "https://x", lines.append)
    assert len(lines) == 2  # source line + licence line, printed once
    assert "non-commercial research" in lines[1]


def test_report_session_flags_cpu_fallback(tiny_conv):
    sess = ort.InferenceSession(tiny_conv, providers=["CPUExecutionProvider"])
    out, status = [], []
    msg = ffr.report_session("swap model", sess, ["DmlExecutionProvider", "CPUExecutionProvider"],
                             out.append, status.append)
    assert msg and "GPU NOT IN USE" in msg and status == [msg]
    assert any("running on: CPUExecutionProvider" in o for o in out)


def test_report_session_quiet_when_cpu_requested(tiny_conv):
    sess = ort.InferenceSession(tiny_conv, providers=["CPUExecutionProvider"])
    assert ffr.report_session("x", sess, ["CPUExecutionProvider"], lambda s: None) is None


def test_report_session_unwraps_adapter(tiny_conv):
    class Adapter:
        def __init__(self, s):
            self._underlying = s
    sess = ort.InferenceSession(tiny_conv, providers=["CPUExecutionProvider"])
    assert ffr.session_active_provider(Adapter(sess)) == "CPUExecutionProvider"
    assert ffr.session_active_provider(object()) is None


def test_report_face_analyser(tiny_conv):
    class M:
        def __init__(self):
            self.session = ort.InferenceSession(tiny_conv, providers=["CPUExecutionProvider"])
    analyser = type("A", (), {"models": {"detection": M(), "recognition": M()}})()
    warnings = ffr.report_face_analyser(analyser, ["CUDAExecutionProvider"], lambda s: None)
    assert len(warnings) == 2


def test_convert_fp16_keeps_emap_and_matches(fake_inswapper, tmp_path):
    dst = tmp_path / "fp16.onnx"
    kept = ffr.convert_fp16(fake_inswapper, str(dst))
    assert kept == ["emap"]
    g = onnx.load(str(dst)).graph
    assert numpy_helper.to_array(g.initializer[-1]).dtype == np.float32  # insightface reads this
    assert g.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT  # float32 I/O kept
    assert not list(tmp_path.glob("*.part"))
    assert stat.S_IMODE(os.stat(dst).st_mode) == 0o644
    rng = np.random.default_rng(3)
    feed = {"target": rng.uniform(0, 1, (1, 3, 128, 128)).astype(np.float32),
            "source": rng.standard_normal((1, 512)).astype(np.float32)}
    a = ort.InferenceSession(fake_inswapper, providers=["CPUExecutionProvider"]).run(None, feed)[0]
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    assert np.abs(a - b).max() < 5e-3


def test_ensure_local_fp16_paths(fake_inswapper, tmp_path):
    fp16 = tmp_path / "m_fp16.onnx"
    missing32 = tmp_path / "missing.onnx"
    msgs = []
    # fp32 missing and cannot be obtained -> False, nothing created
    assert not ffr.ensure_local_fp16(str(missing32), str(fp16), lambda: False, msgs.append)
    assert not fp16.exists()
    # fp32 present -> converted
    assert ffr.ensure_local_fp16(fake_inswapper, str(fp16), lambda: pytest.fail("no download"), msgs.append)
    assert fp16.exists()
    # fp16 present -> no work
    assert ffr.ensure_local_fp16("/nonexistent", str(fp16), lambda: pytest.fail("no download"))


def test_ensure_local_fp16_conversion_failure_falls_back(tmp_path):
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    msgs = []
    assert not ffr.ensure_local_fp16(str(bad), str(tmp_path / "out.onnx"), lambda: True, msgs.append)
    assert any("using fp32" in m for m in msgs)
