import importlib.util
import json
from pathlib import Path
import struct

import pytest

from test_lfm3_config import config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("checkpoint_inventory", ROOT / "tools/inspect_checkpoint.py")
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def fixture(tmp_path, mutation=None):
    c = config()
    c.update(hidden_size=8, intermediate_size=16, vocab_size=32,
             num_attention_heads=2, num_key_value_heads=1, head_dim=4, eos_token_id=1)
    root = dict(model_type="lfm3_omni", text_config=c)
    _, shapes = inventory.text_schema(root)
    offset = 0
    header = {}
    for name, shape in shapes.items():
        import math
        size = 2 * math.prod(shape)
        header[name] = dict(dtype="BF16", shape=shape, data_offsets=[offset, offset + size])
        offset += size
    if mutation:
        mutation(header)
    encoded = json.dumps(header).encode()
    (tmp_path / "config.json").write_text(json.dumps(root))
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return tmp_path


def test_valid_header(tmp_path):
    result = inventory.inspect(fixture(tmp_path))
    assert result["validated_text_tensors"] > 0
    assert result["other_tensors"] == 0
    assert result["native_lfm3_execution_ready"] is False


@pytest.mark.parametrize("mutation,error", [
    (lambda h: h.pop(next(iter(h))), "missing text tensor"),
    (lambda h: h[next(iter(h))].update(dtype="F32"), "unexpected dtype/shape"),
    (lambda h: h[next(iter(h))].update(shape=[1]), "unexpected dtype/shape"),
    (lambda h: h[next(iter(h))].update(data_offsets=[0, 1]), "invalid data offsets"),
    (lambda h: h.update({"model.language_model.extra.weight": {}}), "unexpected text tensors"),
])
def test_invalid_headers(tmp_path, mutation, error):
    with pytest.raises(ValueError, match=error):
        inventory.inspect(fixture(tmp_path, mutation))


def test_overlapping_ranges(tmp_path):
    def mutate(header):
        keys = list(header)
        size = header[keys[1]]["data_offsets"][1] - header[keys[1]]["data_offsets"][0]
        header[keys[1]]["data_offsets"] = [0, size]
    with pytest.raises(ValueError, match="overlapping"):
        inventory.inspect(fixture(tmp_path, mutate))


def test_truncated_header(tmp_path):
    path = fixture(tmp_path)
    (path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        inventory.inspect(path)


def test_actual_checkpoint_inventory():
    path = ROOT.parent / "models/LFM3-Omni-2B"
    if not (path / "model.safetensors").exists():  # CI has the small files only (.github/workflows/host-tests.yml)
        pytest.skip("local checkpoint weights not available")
    assert inventory.inspect(path)["validated_text_tensors"] == 286
