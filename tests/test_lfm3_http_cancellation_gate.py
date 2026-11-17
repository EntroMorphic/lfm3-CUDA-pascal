"""Strict partial HTTP cancellation assertions; no listener or GPU."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
SPEC = importlib.util.spec_from_file_location("http_cancel_gate", Path(__file__).resolve().parents[1] / "tools/check_lfm3_http_cancellation.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize("changes", [{}, {"cancelled":False}, {"finish_reason":"length"},
    {"cache_valid":True}, {"consumed_tokens":1}, {"consumed_tokens":False},
    {"generated_ids":[]}, {"generated_ids":[1]}, {"generated_ids":[1,3]},
    {"generated_ids":[True,2]}, {"generated_ids":list(range(1,33))}])
def test_partial_result_cannot_hide_invalid_terminal_state(changes):
    response = dict(cancelled=True, finish_reason="cancelled", cache_valid=False,
                    consumed_tokens=0, generated_ids=[1,2])
    response.update(changes)
    if changes:
        with pytest.raises(AssertionError):
            gate.validate_cancelled(response, list(range(1,41)))
    else:
        gate.validate_cancelled(response, list(range(1,41)))
