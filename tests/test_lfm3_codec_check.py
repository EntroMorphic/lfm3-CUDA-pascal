import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lfm3_codec_check as cc  # noqa: E402


def test_snr_db():
    ref = np.ones(100)
    assert cc.snr_db(ref, ref) == float("inf")
    assert abs(cc.snr_db(ref * 1.1, ref) - 20.0) < 1e-9


def _case(tmp, name, whole, streams, hf_db):
    a, n = tmp / "anchor", tmp / "native"
    a.mkdir(exist_ok=True), n.mkdir(exist_ok=True)
    ref = np.sin(np.arange(1920) / 7.0)
    np.save(a / f"{name}.fp64.npy", ref)
    np.save(a / f"{name}.fp32.npy", ref.astype(np.float32))
    whole = (ref + whole).astype(np.float32)
    whole.tofile(n / f"{name}.whole.f32")
    for k, s in zip((12, 1), streams):
        (whole + s).astype(np.float32).tofile(n / f"{name}.chunk{k}.f32")
    v = {"snr_db": hf_db}
    return {name: {"vs_fp64": {"fp32": {"snr_db": 100.0}}, "vs_fp32": {"bf16": v, "bf16-eager": v, "bf16-stream": v}}}


def test_report_bar_and_stream_identity(tmp_path):
    results = {}
    results.update(_case(tmp_path, "good", 1e-6, (0, 0), 30.0))
    results.update(_case(tmp_path, "noisy", 0.1, (0, 0), 30.0))   # ~17 dB < bar
    results.update(_case(tmp_path, "drift", 1e-6, (0, 1e-3), 30.0))  # stream1 not bitwise
    (tmp_path / "anchor" / "anchor.json").write_text(json.dumps({"results": results}))
    out = tmp_path / "a2.json"
    assert not cc.report(tmp_path / "anchor", tmp_path / "native", out, ["good", "noisy", "drift"])
    r = json.loads(out.read_text())["results"]
    assert r["good"]["a2_pass"] and not r["noisy"]["a2_pass"] and not r["drift"]["a2_pass"]
    assert r["drift"]["stream12_bitwise_equal_whole"] and not r["drift"]["stream1_bitwise_equal_whole"]
