import os
import pathlib
import shutil
import sys

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "patches"))
sys.path.insert(0, str(REPO / "tools"))


def _finish(graph, opset=13):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8  # loadable by every pinned onnxruntime
    onnx.checker.check_model(model)
    return model


@pytest.fixture
def fake_inswapper(tmp_path):
    """Inswapper-shaped model: inputs target (1,3,128,128) + source (1,512),
    output (1,3,128,128) in [0,1], plus an unused 512x512 'emap' initializer
    as the LAST initializer (what insightface reads)."""
    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.standard_normal((3, 3, 3, 3)).astype(np.float32) * 0.1, "w")
    proj = numpy_helper.from_array(rng.standard_normal((512, 3)).astype(np.float32) * 0.01, "proj")
    shape = numpy_helper.from_array(np.array([1, 3, 1, 1], np.int64), "shape")
    emap = numpy_helper.from_array(rng.standard_normal((512, 512)).astype(np.float32), "emap")
    nodes = [
        helper.make_node("Conv", ["target", "w"], ["c"], pads=[1, 1, 1, 1]),
        helper.make_node("MatMul", ["source", "proj"], ["s"]),
        helper.make_node("Reshape", ["s", "shape"], ["s4"]),
        helper.make_node("Add", ["c", "s4"], ["a"]),
        helper.make_node("Sigmoid", ["a"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_inswapper",
        [helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, 3, 128, 128]),
         helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 512])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 128, 128])],
        [w, proj, shape, emap])
    path = tmp_path / "fake_inswapper.onnx"
    onnx.save(_finish(graph), str(path))
    return str(path)


@pytest.fixture
def tiny_conv(tmp_path):
    rng = np.random.default_rng(1)
    w = numpy_helper.from_array(rng.standard_normal((4, 3, 3, 3)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 32, 32])], [w])
    path = tmp_path / "tiny.onnx"
    onnx.save(_finish(graph), str(path))
    return str(path)


@pytest.fixture
def upstream_copy(tmp_path):
    """Fresh copy of the pinned upstream checkout (set FACEFORGE_UPSTREAM)."""
    src = os.environ.get("FACEFORGE_UPSTREAM")
    if not src or not os.path.isdir(src):
        pytest.skip("FACEFORGE_UPSTREAM not set to an upstream Deep-Live-Cam checkout")
    dst = tmp_path / "upstream"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
    return dst


@pytest.fixture
def weights_dir():
    d = os.environ.get("FACEFORGE_WEIGHTS")
    if not d or not os.path.isdir(d):
        pytest.skip("FACEFORGE_WEIGHTS not set (real-model tests need user-obtained weights)")
    return pathlib.Path(d)
