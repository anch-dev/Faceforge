"""Convert a face-pipeline ONNX model to a Core ML ML Program (.mlpackage).

Runs on Linux, Windows or macOS (conversion only; Core ML *prediction* needs
macOS/iOS). Intended for the user to run on their own copy of the weights,
then copy the .mlpackage to the iPhone (Files app); the app compiles it on
device with MLModel.compileModel(at:). Faceforge never ships converted weights.

    pip install torch coremltools onnx2torch onnx onnxruntime
    python tools/convert_coreml.py inswapper_128.onnx inswapper_128.mlpackage \
        --shape 1,3,128,128 --shape 1,512

Path: ONNX -> onnx2torch -> TorchScript trace -> coremltools ML Program
(fp16, iOS 17 target). Before converting, the PyTorch intermediate is checked
against ONNX Runtime on random input and the max difference printed.

Verified in the audit sandbox (Linux, coremltools 9.0, torch 2.14): det_10g,
w600k_r50, 2d106det and inswapper_128 all convert; intermediate vs ORT max
|diff| <= 8e-7. Not yet verified: Core ML prediction parity and on-device speed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings


def parse_shape(s: str):
    return tuple(int(x) for x in s.split(","))


def convert(onnx_path: str, out_path: str, shapes, precision: str = "fp16", target: str = "iOS17"):
    import coremltools as ct
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnx2torch import convert as onnx_to_torch

    tm = onnx_to_torch(onnx.load(onnx_path)).eval()
    example = tuple(torch.randn(*s) for s in shapes)
    with torch.no_grad():
        ref = tm(*example)
        traced = torch.jit.trace(tm, example, check_trace=False)
    ref0 = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    ort_out = sess.run(None, dict(zip(names, [e.numpy() for e in example])))[0]
    diff = float(np.abs(ref0 - ort_out).max())

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s) for n, s in zip(names, shapes)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=getattr(ct.target, target),
    )
    mlmodel.save(out_path)
    ops = {}
    for fn in mlmodel._mil_program.functions.values():
        for op in fn.operations:
            ops[op.op_type] = ops.get(op.op_type, 0) + 1
    return {"intermediate_vs_ort_max_abs_diff": diff, "mil_ops": ops, "output": out_path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--shape", action="append", required=True, type=parse_shape,
                    help="input shape, once per input in model order, e.g. 1,3,128,128")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--target", default="iOS17")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    result = convert(args.onnx, args.out, args.shape, args.precision, args.target)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
