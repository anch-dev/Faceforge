"""Faceforge shared evaluation pipeline.

A small, dependency-light reference implementation of the face pipeline
(detect -> align -> embed -> swap -> composite) on ONNX Runtime + NumPy +
OpenCV. It exists so that:

* Windows backends (CUDA / DirectML / OpenVINO / CPU) are benchmarked with
  identical pre/post-processing, independent of the Deep-Live-Cam UI.
* The iOS Core ML port has a golden reference: same stage interfaces, same
  inputs, comparable outputs.

Stage interfaces (mirrored in Swift on iOS):

    Detector.detect(bgr_frame) -> list[Face]          # bbox, score, 5 landmarks
    Embedder.embed(bgr_frame, face) -> np.ndarray     # (512,) L2-normalised
    Swapper.swap(bgr_frame, face, source_embedding) -> (aligned_bgr, M)
    Compositor.paste(bgr_frame, aligned_bgr, M) -> bgr_frame

Model weights are never bundled; pass paths to files the user obtained from
the publisher under its terms.
"""

from .types import Face  # noqa: F401
