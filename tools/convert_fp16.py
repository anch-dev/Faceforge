"""Convert an fp32 ONNX model to fp16 locally (float32 I/O, unused initializers
such as inswapper's 'emap' kept as-is). Same code the app runs on first launch.

Usage: python tools/convert_fp16.py SRC.onnx DST.onnx

Verified on inswapper_128 (CPU EP): cosine vs fp32 output 0.999994, mean pixel
difference 0.34/255, identical identity-similarity score. GPU numerics unverified.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patches"))

from faceforge_runtime import convert_fp16  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    print("kept fp32:", convert_fp16(sys.argv[1], sys.argv[2]))
