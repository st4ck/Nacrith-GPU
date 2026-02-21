"""Tests for parallel neural compressor."""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parallel import ParallelNeuralCompressor


@pytest.mark.slow
def test_roundtrip_small():
    pc = ParallelNeuralCompressor(n_workers=2, verbose=False)
    text = "Hello, world! This is a test of the parallel compressor."
    compressed = pc.compress(text)
    assert compressed[:4] == b"NC05"
    decompressed = pc.decompress(compressed)
    assert decompressed == text


@pytest.mark.slow
def test_roundtrip_medium():
    pc = ParallelNeuralCompressor(n_workers=4, verbose=False)
    text = (
        "The quick brown fox jumps over the lazy dog. " * 20
        + "\nNatural language processing has seen remarkable advances. " * 20
    )
    compressed = pc.compress(text)
    decompressed = pc.decompress(compressed)
    assert decompressed == text


@pytest.mark.slow
def test_roundtrip_empty():
    pc = ParallelNeuralCompressor(n_workers=2, verbose=False)
    compressed = pc.compress("")
    decompressed = pc.decompress(compressed)
    assert decompressed == ""


@pytest.mark.slow
def test_roundtrip_alice29():
    alice_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "benchmark", "alice29.txt",
    )
    if not os.path.isfile(alice_path):
        pytest.skip("alice29.txt not found")
    with open(alice_path, "r") as f:
        text = f.read()
    pc = ParallelNeuralCompressor(n_workers=8, verbose=False)
    compressed = pc.compress(text)
    decompressed = pc.decompress(compressed)
    assert decompressed == text, (
        f"Length mismatch: {len(text)} vs {len(decompressed)}"
    )


@pytest.mark.slow
def test_speedup():
    """Verify parallel is faster than serial for non-trivial text."""
    pc1 = ParallelNeuralCompressor(n_workers=1, verbose=False)
    pc4 = ParallelNeuralCompressor(n_workers=4, verbose=False)

    text = "The quick brown fox jumps over the lazy dog. " * 50

    t0 = time.perf_counter()
    pc1.compress(text)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    pc4.compress(text)
    t_parallel = time.perf_counter() - t0

    assert t_parallel < t_serial, (
        f"Parallel ({t_parallel:.2f}s) not faster than serial ({t_serial:.2f}s)"
    )
