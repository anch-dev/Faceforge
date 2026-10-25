import time

import numpy as np
import pytest

from faceforge_audio import Delay, Passthrough, VoiceProcessor, measure_latency_samples, stream_benchmark


def test_reference_processors_meet_contract():
    for p in (Passthrough(), Delay(240)):
        assert isinstance(p, VoiceProcessor)
        out = p.process(np.zeros(p.block_size, np.float32))
        assert out.dtype == np.float32 and out.shape == (p.block_size,)


def test_contract_rejects_wrong_blocks():
    p = Passthrough(block_size=480)
    with pytest.raises(ValueError):
        p.process(np.zeros(479, np.float32))
    with pytest.raises(ValueError):
        p.process(np.zeros(480, np.float64))


def test_passthrough_is_exact():
    p = Passthrough()
    x = np.random.default_rng(0).standard_normal(p.block_size).astype(np.float32)
    assert np.array_equal(p.process(x), x)


@pytest.mark.parametrize("delay", [0, 1, 240, 961, 4800])
def test_measured_latency_matches_declared(delay):
    proc = Delay(delay)
    assert measure_latency_samples(proc) == proc.algorithmic_latency_samples


def test_reset_clears_state():
    proc = Delay(100, block_size=100)
    proc.process(np.ones(100, np.float32))
    proc.reset()
    assert not proc.process(np.zeros(100, np.float32)).any()


def test_benchmark_detects_underruns():
    class Slow(Passthrough):
        def process(self, block):
            time.sleep(self.block_size / self.sample_rate * 1.5)
            return super().process(block)
    fast = stream_benchmark(Passthrough(), seconds=0.5)
    slow = stream_benchmark(Slow(block_size=4800), seconds=0.5)
    assert fast["underrun_blocks"] == 0 and fast["real_time_factor"] < 1
    assert slow["underrun_blocks"] == slow["blocks"] and slow["real_time_factor"] > 1
    assert fast["pipeline_latency_ms"] >= fast["block_ms"]
