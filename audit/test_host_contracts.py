"""Adversarial acceptance regressions: all must pass after remediation.

Run with PYTHONPATH=tests python -m pytest audit/test_host_contracts.py -q.
These preserve the five originally failing review cases without weakening assertions.
"""
import json
import struct

import pytest

from test_checkpoint_inventory import fixture, inventory


def test_reject_fractional_type_offsets(tmp_path):
    def mutate(header):
        for tensor in header.values():
            tensor["data_offsets"] = [float(x) for x in tensor["data_offsets"]]
    with pytest.raises(ValueError):
        inventory.inspect(fixture(tmp_path, mutate))


def test_reject_boolean_shape_dimensions(tmp_path):
    def mutate(header):
        name = next(name for name in header if name.endswith("conv.conv.weight"))
        header[name]["shape"][1] = True
    with pytest.raises(ValueError):
        inventory.inspect(fixture(tmp_path, mutate))


def test_reject_other_modality_overlapping_text(tmp_path):
    def mutate(header):
        source = header[next(iter(header))]
        header["model.audio_encoder.fake.weight"] = dict(source)
    with pytest.raises(ValueError):
        inventory.inspect(fixture(tmp_path, mutate))


def test_reject_duplicate_header_keys(tmp_path):
    path = fixture(tmp_path)
    source = (path / "model.safetensors").read_bytes()
    length = struct.unpack("<Q", source[:8])[0]
    header = json.loads(source[8:8 + length])
    name = next(iter(header))
    duplicate = json.dumps(name) + ":" + json.dumps(header[name])
    encoded = ("{" + duplicate + "," + source[8:8 + length].decode()[1:]).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + source[8 + length:])
    with pytest.raises(ValueError):
        inventory.inspect(path)


def test_reject_bias_config_without_bias_tensors(tmp_path):
    path = fixture(tmp_path)
    config = json.loads((path / "config.json").read_text())
    config["text_config"]["conv_bias"] = True
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        inventory.inspect(path)
