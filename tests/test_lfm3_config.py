import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def config():
    return dict(model_type="lfm3", hidden_size=2048, intermediate_size=7680,
                vocab_size=128000, num_hidden_layers=4, num_attention_heads=16,
                num_key_value_heads=4, head_dim=128, conv_L_cache=7, norm_eps=1e-5,
                max_position_embeddings=128000, tie_word_embeddings=True,
                use_sandwich_norm=True, conv_bias=False, sliding_window=256,
                layer_types=["conv", "full_attention", "sliding_attention", "conv"],
                rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5},
                                 "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}},
                eos_token_id=124900)


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    binary = tmp_path_factory.mktemp("lfm3") / "probe"
    subprocess.run(["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/probes/lfm3_config_probe.cpp"), "-o", str(binary)], check=True)
    return binary


def run(probe, tmp_path, value, execute=False):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return subprocess.run([str(probe), str(path)] + (["--execute"] if execute else []),
                          capture_output=True, text=True)


def test_standalone_geometry_and_layer_traits(probe, tmp_path):
    result = run(probe, tmp_path, config())
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["hidden"] == 2048 and data["mlp"] == 7680
    assert data["conv_width"] == 7 and data["prefix"] == "model."
    assert data["gate"] == data["sandwich_norm"] == 1
    assert data["schedule"][1] == dict(kind=1, theta=1e6, rotary_pairs=32, window=0)
    assert data["schedule"][2] == dict(kind=2, theta=1e4, rotary_pairs=64, window=256)


def test_nested_omni_and_no_lfm2_width_adjustment(probe, tmp_path):
    c = config()
    c.update(block_auto_adjust_ff_dim=True, block_multiple_of=256)
    result = run(probe, tmp_path, dict(model_type="lfm3_omni", text_config=c))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["mlp"] == 7680
    assert data["prefix"] == "model.language_model."
    assert data["context"] == 32768


def test_execution_is_explicitly_blocked(probe, tmp_path):
    result = run(probe, tmp_path, config(), execute=True)
    assert result.returncode == 1
    assert "native LFM3 execution is not implemented" in result.stderr


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 0), ("intermediate_size", 0), ("num_attention_heads", 0),
    ("num_key_value_heads", 3), ("head_dim", 64), ("sliding_window", 0),
    ("layer_types", ["conv"]), ("layer_types", ["conv", "mamba", "conv", "conv"]),
    ("conv_bias", True), ("eos_token_id", 128000), ("norm_eps", 0),
])
def test_invalid_config(probe, tmp_path, field, value):
    c = config()
    c[field] = value
    assert run(probe, tmp_path, c).returncode == 1


@pytest.mark.parametrize("field,value", [("partial_rotary_factor", -0.1),
                                         ("partial_rotary_factor", 1.1),
                                         ("rope_theta", 0), ("rope_type", "yarn")])
def test_invalid_rope(probe, tmp_path, field, value):
    c = copy.deepcopy(config())
    c["rope_parameters"]["full_attention"][field] = value
    assert run(probe, tmp_path, c).returncode == 1


def test_actual_omni_checkpoint(probe, tmp_path):
    path = ROOT.parent / "models/LFM3-Omni-2B/config.json"
    if not path.exists():
        pytest.skip("local checkpoint not available")
    result = run(probe, tmp_path, json.loads(path.read_text()))
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["layers"] == 24
    assert [sum(x["kind"] == kind for x in data["schedule"]) for kind in range(3)] == [13, 6, 5]


def test_each_checkpoint_keeps_its_own_context_limit(tmp_path):
    # The engine ceiling is 262,144 (text_serving_limits.hpp); each checkpoint's card limit is ModelConfig's:
    # LFM3-Omni-2B 32K (its card, though its backbone config allows 128,000), LFM3-1.7B 256K.
    models = Path(__file__).resolve().parents[2] / "models"
    omni, base = models / "LFM3-Omni-2B/config.json", models / "LFM3-1.7B/config.json"
    if not (omni.exists() and base.exists()):
        pytest.skip("checkpoint configs not present")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    source = tmp_path / "limits.cpp"
    source.write_text(r'''
#include <lfm/model_config.hpp>
#include <cstdlib>
#include <iostream>
int main(int argc, char** argv) {
  try { auto c = lfm::ModelConfig::load(argv[1], std::atoi(argv[2])); std::cout << c.shape.context << "\n"; return 0; }
  catch (const std::exception& e) { std::cout << "rejected\n"; return 0; }
}
''')
    exe = tmp_path / "limits"
    subprocess.run([compiler, "-std=c++20", "-I", str(ROOT / "native/include"), str(source), "-o", str(exe)], check=True)
    run = lambda cfg, cap: subprocess.run([str(exe), str(cfg), str(cap)], capture_output=True, text=True,  # noqa: E731
                                          check=True).stdout.strip()
    assert run(omni, 32768) == "32768" and run(omni, 32769) == "rejected"
    assert run(base, 262144) == "262144" and run(base, 262145) == "rejected"
