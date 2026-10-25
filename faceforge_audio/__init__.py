"""Faceforge audio module: kept separate from the visual pipeline.

Defines the streaming contract any voice processor must meet and a harness
that measures it. No voice-conversion model is included; see
docs/REPORT.md (Voice) for the candidate evaluation.

Contract (mirrored in Swift on iOS, e.g. inside an AVAudioEngine tap):

    processor.sample_rate                 int, Hz
    processor.block_size                  samples per call (mono float32)
    processor.algorithmic_latency_samples delay the model itself adds
                                          (lookahead, overlap-add, buffering)
    processor.process(block) -> block     same length, float32, in [-1, 1]
    processor.reset()                     clear internal state between streams

End-to-end latency = device input buffer + block_size + algorithmic latency
+ processing time + device output buffer. The harness reports each part it
can measure and leaves device buffers to the platform.
"""
from .interface import Delay, Passthrough, VoiceProcessor  # noqa: F401
from .measure import measure_latency_samples, stream_benchmark  # noqa: F401
