import json
import struct

import pytest

from test_checkpoint_inventory import fixture, inventory
from test_lfm3_config import config, probe, run


@pytest.mark.parametrize("field,value", [
    ("hidden_size", True), ("hidden_size", 0), ("intermediate_size", 0),
    ("head_dim", 64), ("num_attention_heads", 0), ("num_key_value_heads", 3),
    ("conv_bias", True), ("conv_bias", 0), ("use_sandwich_norm", 1),
    ("norm_eps", 1e300), ("norm_eps", 1e-300), ("norm_eps", 10**400), ("sliding_window", 0),
    ("eos_token_id", True), ("eos_token_id", []),
])
def test_python_and_native_config_reject_same_bad_metadata(probe, tmp_path, field, value):
    c = config()
    c[field] = value
    assert run(probe, tmp_path, c).returncode == 1
    with pytest.raises(ValueError):
        inventory.text_schema(c)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e9999}'])
def test_strict_json_rejects_ambiguous_or_nonfinite(text):
    with pytest.raises(ValueError):
        inventory.strict_json(text)


def test_utf16_header_not_accepted_as_utf8(tmp_path):
    path = fixture(tmp_path)
    p = path / "model.safetensors"
    raw = p.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    encoded = raw[8:8 + length].decode().encode("utf-16")
    p.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length:])
    with pytest.raises(ValueError):
        inventory.inspect(path)


@pytest.mark.parametrize("root", [None, [], 1, {"model_type": []}, {"model_type": "lfm3_omni", "text_config": []}])
def test_bad_root_is_value_error(root):
    with pytest.raises(ValueError):
        inventory.text_schema(root)


def test_trailing_payload_rejected(tmp_path):
    path = fixture(tmp_path)
    with (path / "model.safetensors").open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(ValueError, match="trailing"):
        inventory.inspect(path)


@pytest.mark.parametrize("extra", [
    {"dtype": "BF16", "shape": [1], "data_offsets": [False, 2]},
    {"dtype": "unknown", "shape": [1], "data_offsets": [0, 2]},
    {"dtype": "BF16", "shape": [-1], "data_offsets": [0, 2]},
    {"dtype": "BF16", "shape": [1.0], "data_offsets": [0, 2]},
    {"dtype": "BF16", "shape": [1], "data_offsets": [-2, 0]},
    {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2], "extra": True},
])
def test_all_modalities_receive_strict_structural_validation(tmp_path, extra):
    with pytest.raises(ValueError):
        inventory.inspect(fixture(tmp_path, lambda h: h.update({"model.audio.fake": extra})))


def test_payload_gap_rejected(tmp_path):
    path = fixture(tmp_path)
    p = path / "model.safetensors"
    raw = p.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    for tensor in header.values():
        tensor["data_offsets"] = [n + 2 for n in tensor["data_offsets"]]
    encoded = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"xx" + raw[8 + length:])
    with pytest.raises(ValueError, match="hole"):
        inventory.inspect(path)


def test_valid_scalar_empty_tensor_and_metadata(tmp_path):
    path = fixture(tmp_path)
    p = path / "model.safetensors"
    raw = p.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    offset = len(raw) - 8 - length
    header["model.audio.scalar"] = dict(dtype="F32", shape=[], data_offsets=[offset, offset + 4])
    header["model.audio.empty"] = dict(dtype="F32", shape=[0, 8], data_offsets=[offset + 4, offset + 4])
    header["__metadata__"] = {"test": "valid mixed-modality control"}
    encoded = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length:] + bytes(4))
    assert inventory.inspect(path)["other_tensors"] == 2
