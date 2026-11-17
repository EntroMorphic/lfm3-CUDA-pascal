"""CPU-only tests for architecture metadata; reject malformed configs before CUDA."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ required")
    binary = tmp_path_factory.mktemp("config") / "probe"
    subprocess.run(["g++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "native/include"),
                    str(ROOT / "native/probes/config_probe.cpp"), "-o", str(binary)], check=True)
    return binary


def config():
    return dict(model_type="lfm2", hidden_size=2048, intermediate_size=12288,
                vocab_size=65536, num_hidden_layers=4, num_attention_heads=32,
                num_key_value_heads=8, conv_L_cache=3, norm_eps=1e-5,
                rope_theta=1e6, max_position_embeddings=128000,
                block_use_swiglu=True, block_auto_adjust_ff_dim=True,
                block_multiple_of=256, block_ffn_dim_multiplier=1.0,
                conv_bias=False, tie_embedding=True,
                layer_types=["conv", "full_attention", "conv", "full_attention"],
                eos_token_id=7)


def run(probe, tmp_path, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)
    return subprocess.run([str(probe), str(path)], capture_output=True, text=True)


def test_dimensions_and_layer_schedule_are_not_model_specific(probe, tmp_path):
    result = run(probe, tmp_path, config())
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2048 8192 32 8 4 2"
    c = config()
    c.update(hidden_size=1024, intermediate_size=6656, num_attention_heads=16)
    c["rope_parameters"] = {"rope_type": "default", "rope_theta": c.pop("rope_theta")}
    result = run(probe, tmp_path, c)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1024 4608 16 8 4 2"


def test_version_aliases_and_unicode_metadata(probe, tmp_path):
    c = config()
    c.update(block_ff_dim=12288, intermediate_size=1, layer_types=None,
             full_attn_idxs=[1, 3], description="液体 🧪")
    result = run(probe, tmp_path, c)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2048 8192 32 8 4 2"
    c["block_ffn_dim_multiplier"] = None
    result = run(probe, tmp_path, c)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2048 8192 32 8 4 2"


def test_conv_only_and_all_attention_schedules(probe, tmp_path):
    c = config()
    c["layer_types"] = ["conv"] * 4
    assert run(probe, tmp_path, c).stdout.strip().endswith("4 0")
    c.pop("layer_types")
    assert run(probe, tmp_path, c).stdout.strip().endswith("4 4")


def moe_config():
    c = config()
    c.update(model_type="lfm2_moe", intermediate_size=7168, num_experts=32,
             num_experts_per_tok=4, moe_intermediate_size=1792, num_dense_layers=2,
             norm_topk_prob=True, use_expert_bias=True, routed_scaling_factor=1.0)
    return c


def test_moe_dense_width_does_not_use_dense_lfm2_adjustment(probe, tmp_path):
    result = run(probe, tmp_path, moe_config())
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2048 7168 32 8 4 2"


def test_flagship_geometry_uses_metadata_without_recompilation(probe, tmp_path):
    c = moe_config()
    c.update(num_hidden_layers=40, intermediate_size=11776, moe_intermediate_size=1536,
             num_experts=64, layer_types=["full_attention" if i in (2, 6, 10, 14, 18, 22, 26, 30, 34, 38) else "conv" for i in range(40)],
             rope_parameters=dict(rope_theta=10000000.0, rope_type="default"))
    result = run(probe, tmp_path, c)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2048 11776 32 8 40 10"


@pytest.mark.parametrize("hidden,width,layers,heads,attention", [(1024, 2560, 14, 16, 6),
                                                               (2048, 10752, 30, 32, 8)])
def test_remaining_dense_geometries_do_not_readjust_explicit_ff_width(probe, tmp_path, hidden, width, layers, heads, attention):
    c = config()
    c.update(hidden_size=hidden, intermediate_size=width, num_hidden_layers=layers,
             num_attention_heads=heads, block_auto_adjust_ff_dim=False,
             layer_types=["full_attention"] * attention + ["conv"] * (layers - attention))
    result = run(probe, tmp_path, c)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{hidden} {width} {heads} 8 {layers} {attention}"


@pytest.mark.parametrize("field,value", [("num_experts", 0), ("num_experts_per_tok", 33),
                                        ("num_experts_per_tok", 0), ("num_dense_layers", 5),
                                        ("moe_intermediate_size", 0), ("routed_scaling_factor", -1)])
def test_invalid_moe_metadata(probe, tmp_path, field, value):
    c = moe_config()
    c[field] = value
    assert run(probe, tmp_path, c).returncode == 1


@pytest.mark.parametrize("field,value", [
    ("hidden_size", 0), ("hidden_size", 2049), ("hidden_size", 2**40),
    ("hidden_size", 1.5), ("hidden_size", True), ("num_attention_heads", 0),
    ("num_key_value_heads", 7), ("layer_types", ["conv"]),
    ("layer_types", ["conv", "moe", "conv", "conv"]),
    ("model_type", "lfm2_moe"), ("conv_bias", True), ("conv_L_cache", 0),
    ("norm_eps", 0), ("rope_theta", -1), ("eos_token_id", 65536),
    ("block_multiple_of", 0), ("intermediate_size", 0),
    ("conv_dim", 1024), ("head_dim", 128), ("partial_rotary_factor", 0.5),
    ("rope_parameters", {"rope_type": "yarn", "rope_theta": 1e6}),
])
def test_reject_invalid_or_unsupported_architecture(probe, tmp_path, field, value):
    c = config()
    c[field] = value
    assert run(probe, tmp_path, c).returncode == 1


@pytest.mark.parametrize("text", ['{}garbage', '{"a":1,"a":2}', '{"a":01}',
                                   '{"a":1.}', '{"a":1e}', '{"a":true,}',
                                   '{"a":NaN}', '{"a":Infinity}', '{"a":'])
def test_reject_malformed_json(probe, tmp_path, text):
    assert run(probe, tmp_path, text).returncode == 1
