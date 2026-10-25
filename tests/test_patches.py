import py_compile
import subprocess
import sys

import pytest

from conftest import REPO

PATCHER = str(REPO / "patches" / "apply_faceforge.py")


def run(*args):
    return subprocess.run([sys.executable, PATCHER, *map(str, args)], capture_output=True, text=True)


def snapshot(root):
    return {p: p.read_bytes() for p in root.rglob("*.py")}


def test_check_mode_writes_nothing(upstream_copy):
    before = snapshot(upstream_copy)
    r = run(upstream_copy, "--check")
    assert r.returncode == 0, r.stdout
    assert "12 would apply" in r.stdout
    assert snapshot(upstream_copy) == before


def test_apply_then_reapply_is_noop(upstream_copy):
    r = run(upstream_copy)
    assert r.returncode == 0, r.stdout
    assert "12 applied" in r.stdout
    after_first = snapshot(upstream_copy)
    r2 = run(upstream_copy)
    assert r2.returncode == 0 and "0 applied, 12 already present" in r2.stdout
    assert snapshot(upstream_copy) == after_first


def test_patched_tree_compiles(upstream_copy):
    assert run(upstream_copy).returncode == 0
    for f in list((upstream_copy / "modules").rglob("*.py")):
        py_compile.compile(str(f), doraise=True)
    assert (upstream_copy / "modules" / "faceforge_runtime.py").exists()


def test_drifted_upstream_fails_without_partial_writes(upstream_copy):
    g = upstream_copy / "modules" / "globals.py"
    g.write_text(g.read_text().replace("DEFAULT_DET_SIZE: int = 640", "DEFAULT_DET_SIZE: int = 512"))
    before = snapshot(upstream_copy)
    r = run(upstream_copy)
    assert r.returncode == 1
    assert "upstream changed" in r.stdout and "globals: default det_size 320" in r.stdout
    # The dry-run gate means no other file was touched.
    assert snapshot(upstream_copy) == before
    assert not (upstream_copy / "modules" / "faceforge_runtime.py").exists()


def test_key_patch_contents(upstream_copy):
    assert run(upstream_copy).returncode == 0
    fs = (upstream_copy / "modules/processors/frame/face_swapper.py").read_text()
    assert "_ffr.prefers_fp16(" in fs and "_ffr.ensure_local_fp16(" in fs
    assert "use_fp16 = _HAS_TORCH_CUDA and" not in fs
    md = (upstream_copy / "modules/model_downloader.py").read_text()
    assert "_ffr.publisher_url(name)" in md and "_ffr.download_allowed(name)" in md
    pi = (upstream_copy / "modules/platform_info.py").read_text()
    assert '{"device_type": "AUTO:GPU,NPU,CPU"}' not in pi  # (upstream's comment may still mention it)
    assert 'FACEFORGE_OV_DEVICE", "AUTO")' in pi


@pytest.mark.parametrize("line", ["repo=hacksider/Deep-Live-Cam", "commit="])
def test_upstream_lock_format(line):
    text = (REPO / "patches" / "UPSTREAM.lock").read_text()
    assert line in text
    sha = [l for l in text.splitlines() if l.startswith("commit=")][0].split("=", 1)[1]
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
