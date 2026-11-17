"""run_native_gates.sh refuses unsafe argument combinations before it touches any build (G4 tiers)."""
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/thor/run_native_gates.sh"


@pytest.mark.parametrize("env,message", [
    (dict(TIERS="pin", KEEP_GOING="1", PIN_DIR="/x"), "TIERS=pin requires KEEP_GOING=0"),
    (dict(TIERS="pin"), "TIERS needs PIN_DIR"),
    (dict(TIERS="bogus", PIN_DIR="/x"), "TIERS must be pin or golden"),
    (dict(KEEP_GOING="2"), "KEEP_GOING must be 0 or 1"),
])
def test_unsafe_combinations_are_refused(env, message):
    result = subprocess.run(["bash", str(SCRIPT), "no-such-build", "no-such-tag"], capture_output=True, text=True,
                            env={**{k: v for k, v in os.environ.items() if k not in ("TIERS", "KEEP_GOING", "PIN_DIR")}, **env})
    assert result.returncode == 2 and message in result.stdout, (result.returncode, result.stdout, result.stderr)


def fake_build(tmp_path, cublas, gemm):
    """A build directory whose binaries a stub readelf/nm describe: `cublas` and `gemm` are counts out of 3."""
    build = tmp_path / "root" / "b"
    build.mkdir(parents=True)
    bins = ["liquid-lfm3", "liquid-lfm3-worker", "liquid-lfm3-server"]
    for i, name in enumerate(bins + ["text-sampling-oracle"]):
        path = build / name
        path.write_text(f"{'cublas' if i < cublas else ''} {'gemm' if i < gemm else ''}")
        path.chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "readelf").write_text('#!/bin/sh\ngrep -q cublas "$2" && echo " (NEEDED) Shared library: [libcublas.so.13]"; echo " (NEEDED) [libc.so.6]"\n')
    (stubs / "nm").write_text('#!/bin/sh\ngrep -q gemm "$2" && echo "0 T lfm::gemm::run"; echo "0 T main"\n')
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    return tmp_path / "root", stubs


@pytest.mark.parametrize("cublas,gemm,tiers,refusal", [
    (0, 3, "", "is a native-GEMM build: set TIERS"),
    (3, 0, "golden", "links cuBLAS: TIERS is for native-GEMM builds"),
    (1, 3, "golden", "cannot tell the engine kind"),
    (0, 2, "golden", "cannot tell the engine kind"),
    (0, 3, "golden", None),   # native with tiers: accepted (fails later, outside the test, on Thor-only paths)
    (3, 0, "", None),         # cuBLAS without tiers: accepted
])
def test_engine_kind_is_judged_from_all_three_binaries(tmp_path, cublas, gemm, tiers, refusal):
    root, stubs = fake_build(tmp_path, cublas, gemm)
    env = {k: v for k, v in os.environ.items() if k not in ("TIERS", "KEEP_GOING", "PIN_DIR")}
    env.update(LFM3_BUILD_ROOT=str(root), PATH=f"{stubs}:{env['PATH']}", TIERS=tiers, PIN_DIR="/x")
    result = subprocess.run(["bash", str(SCRIPT), "b", "t"], capture_output=True, text=True, env=env, timeout=60)
    if refusal:
        assert result.returncode == 2 and refusal in result.stdout, result.stdout
    else:
        assert "TIERS" not in result.stdout and "engine kind" not in result.stdout, result.stdout
        assert (root / "t-gates/mode.json").exists()  # got past the check
