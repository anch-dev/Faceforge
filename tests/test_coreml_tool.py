"""Core ML conversion tool test. Needs torch + coremltools + onnx2torch, which
are not part of requirements-dev (heavy); skipped when absent."""
import pytest

pytest.importorskip("coremltools")
pytest.importorskip("onnx2torch")
pytest.importorskip("torch")


def test_converts_inswapper_shaped_model(fake_inswapper, tmp_path):
    from convert_coreml import convert

    out = tmp_path / "fake.mlpackage"
    r = convert(fake_inswapper, str(out), [(1, 3, 128, 128), (1, 512)])
    assert out.exists()
    assert r["intermediate_vs_ort_max_abs_diff"] < 1e-4
    assert "conv" in r["mil_ops"]
