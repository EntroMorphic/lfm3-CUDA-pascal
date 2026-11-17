"""Host counterexamples for fixed soak resource thresholds; no GPU/listener."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
SPEC = importlib.util.spec_from_file_location("soak_gate", Path(__file__).resolve().parents[1] / "tools/check_lfm3_soak.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def sample():
    return dict(server=dict(rss_bytes=1000000, descriptors=8), worker=dict(rss_bytes=2000000, descriptors=6))


@pytest.mark.parametrize("role,metric", [("server", "rss_bytes"), ("worker", "rss_bytes"),
                                         ("server", "descriptors"), ("worker", "descriptors")])
def test_fixed_resource_thresholds_reject_growth(role, metric):
    baseline, current = sample(), sample()
    current[role][metric] += 32 * 1024 * 1024 + 1 if metric == "rss_bytes" else 3
    with pytest.raises(AssertionError):
        gate.resource_gate(baseline, [sample(), current])


def test_resource_gate_accepts_exact_budget_and_rejects_empty():
    baseline, current = sample(), sample()
    for role in ("server", "worker"):
        current[role]["rss_bytes"] += 32 * 1024 * 1024
        current[role]["descriptors"] += 2
    gate.resource_gate(baseline, [current])
    with pytest.raises(AssertionError):
        gate.resource_gate(baseline, [])


def test_real_linux_process_resources_are_measured():
    current = gate.process_sample(os.getpid())
    assert current["rss_bytes"] > 0 and current["descriptors"] > 0
