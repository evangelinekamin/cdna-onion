"""The GPU baseline: provenance is exact, and it conforms on a real GPU when one is present."""

import shutil
from pathlib import Path

import pytest

from onionrig.rig import check as check_mod
from onionrig.rig import runner

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "entries" / "baseline"
CUDA_BUILD = BASELINE / "build" / "grind-cuda"


@pytest.mark.parametrize("name", ["fe25519.cuh", "ge25519.cuh"])
def test_arithmetic_is_byte_for_byte_upstream(name):
    assert (BASELINE / "src" / name).read_bytes() == (BASELINE / "upstream" / name).read_bytes()


def test_upstream_commit_is_pinned():
    commit = (BASELINE / "upstream" / "COMMIT").read_text().split()
    assert commit[0] == "24e5874" and commit[1].startswith("24e5874")


@pytest.mark.integration
@pytest.mark.skipif(not CUDA_BUILD.exists() or shutil.which("nvidia-smi") is None,
                    reason="needs `make -C entries/baseline cuda` and an NVIDIA GPU")
def test_baseline_conforms_on_the_local_gpu():
    results = check_mod.run_checks(runner.EntrySpec((str(CUDA_BUILD),)), ROOT / "vectors")
    failures = [f"{r.name}: {r.detail}" for r in results if not r.ok]
    assert check_mod.passed(results), failures
