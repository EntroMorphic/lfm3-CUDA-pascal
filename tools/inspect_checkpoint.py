"""Validate LFM3 text tensor names/shapes without loading weights or importing torch."""
import argparse
import json
import math
from pathlib import Path
import struct


def strict_json(text):
    if isinstance(text, bytes):
        # json.loads(bytes) otherwise accepts UTF-16/32, unlike safetensors' UTF-8 contract.
        text = text.decode("utf-8", errors="strict")
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant, parse_float=finite_float)


def integer(value, name, minimum=1, maximum=2**31 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid integer: {name}")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"invalid boolean: {name}")
    return value


def positive_number(value, name):
    if type(value) not in (int, float) or value <= 0:
        raise ValueError(f"invalid positive number: {name}")
    try:
        if not math.isfinite(value):
            raise ValueError(f"invalid positive number: {name}")
    except OverflowError as exc:
        raise ValueError(f"numeric overflow: {name}") from exc
    return value


def positive_float32(value, name):
    value = positive_number(value, name)
    try:
        converted = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name} overflows float32") from exc
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} is not a positive finite float32")
    return converted


def validate_config(root):
    """Mirror the native metadata contract; execution is still independently guarded."""
    if not isinstance(root, dict) or root.get("model_type") not in ("lfm3", "lfm3_omni"):
        raise ValueError("expected lfm3 or lfm3_omni checkpoint")
    omni = root["model_type"] == "lfm3_omni"
    config = root.get("text_config") if omni else root
    if not isinstance(config, dict) or config.get("model_type") != "lfm3":
        raise ValueError("expected lfm3 text configuration")
    try:
        for name, maximum in (("hidden_size", 1048576), ("intermediate_size", 1048576),
                              ("vocab_size", 16777216), ("num_hidden_layers", 4096),
                              ("conv_L_cache", 1024), ("num_attention_heads", 1048576),
                              ("num_key_value_heads", 1048576)):
            integer(config[name], name, maximum=maximum)
        hidden, heads, kv = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
        if hidden % heads or heads % kv:
            raise ValueError("incompatible attention head geometry")
        dim = integer(config.get("head_dim", hidden // heads), "head_dim", maximum=1024)
        if dim % 2 or dim != hidden // heads:
            raise ValueError("unsupported head dimension")
        positive_float32(config["norm_eps"], "norm_eps")
        integer(config["max_position_embeddings"], "max_position_embeddings")
        boolean(config["tie_word_embeddings"], "tie_word_embeddings")
        boolean(config["use_sandwich_norm"], "use_sandwich_norm")
        if boolean(config.get("conv_bias", False), "conv_bias"):
            raise ValueError("convolution bias is not implemented")
        if config.get("rope_scaling") is not None:
            raise ValueError("rope_scaling is not implemented")
        if omni and "tie_word_embeddings" in root:
            if boolean(root["tie_word_embeddings"], "tie_word_embeddings") != config["tie_word_embeddings"]:
                raise ValueError("conflicting embedding tying configuration")
        kinds = config["layer_types"]
        if not isinstance(kinds, list) or len(kinds) != config["num_hidden_layers"]:
            raise ValueError("layer schedule length mismatch")
        for kind in kinds:
            if kind not in ("conv", "full_attention", "sliding_attention"):
                raise ValueError(f"unsupported layer kind: {kind}")
            if kind == "conv":
                continue
            rope = config["rope_parameters"][kind]
            if rope["rope_type"] != "default":
                raise ValueError("unsupported RoPE type")
            positive_float32(rope["rope_theta"], "rope_theta")
            factor = positive_number(rope.get("partial_rotary_factor", 1.0), "partial_rotary_factor")
            if factor > 1:
                raise ValueError("partial_rotary_factor exceeds one")
            if kind == "sliding_attention":
                integer(config["sliding_window"], "sliding_window")
        eos = config["eos_token_id"]
        eos = eos if isinstance(eos, list) else [eos]
        if not eos:
            raise ValueError("empty EOS list")
        for token in eos:
            integer(token, "eos_token_id", minimum=0, maximum=config["vocab_size"] - 1)
        context = min(32768, config["max_position_embeddings"])
        attention_layers = sum(kind != "conv" for kind in kinds)
        if attention_layers * kv * context * dim > 2**31 - 1:
            raise ValueError("cache stride exceeds kernel index range")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed LFM3 configuration: {exc}") from exc
    return config


def text_schema(root):
    config = validate_config(root)
    omni = root["model_type"] == "lfm3_omni"
    prefix = "model.language_model." if omni else "model."
    hidden = config["hidden_size"]
    width = config["intermediate_size"]
    heads = config["num_attention_heads"]
    dim = config.get("head_dim", hidden // heads)
    kv = config["num_key_value_heads"] * dim
    expected = {
        prefix + "embed_tokens.weight": [config["vocab_size"], hidden],
        prefix + "embedding_norm.weight": [hidden],
    }
    if not config["tie_word_embeddings"]:
        expected["lm_head.weight"] = [config["vocab_size"], hidden]
    if len(config["layer_types"]) != config["num_hidden_layers"]:
        raise ValueError("layer schedule length mismatch")
    for index, kind in enumerate(config["layer_types"]):
        layer = prefix + f"layers.{index}."
        shapes = {"operator_norm.weight": [hidden], "ffn_norm.weight": [hidden],
                  "feed_forward.w1.weight": [width, hidden],
                  "feed_forward.w3.weight": [width, hidden],
                  "feed_forward.w2.weight": [hidden, width]}
        if config["use_sandwich_norm"]:
            shapes.update({"post_operator_norm.weight": [hidden], "post_ffn_norm.weight": [hidden]})
        if kind == "conv":
            shapes.update({"conv.conv.weight": [hidden, 1, config["conv_L_cache"]],
                           "conv.in_proj.weight": [3 * hidden, hidden],
                           "conv.out_proj.weight": [hidden, hidden]})
        elif kind in {"full_attention", "sliding_attention"}:
            shapes.update({"self_attn.q_proj.weight": [heads * dim, hidden],
                           "self_attn.k_proj.weight": [kv, hidden],
                           "self_attn.v_proj.weight": [kv, hidden],
                           "self_attn.o_proj.weight": [hidden, heads * dim],
                           "self_attn.q_norm.weight": [dim], "self_attn.k_norm.weight": [dim],
                           "self_attn.gate_proj.weight": [heads, hidden]})
        else:
            raise ValueError(f"unsupported layer kind: {kind}")
        expected.update({layer + name: shape for name, shape in shapes.items()})
    return prefix, expected


def inspect(directory):
    directory = Path(directory)
    config_path = directory / "config.json"
    if config_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("model configuration too large")
    root = strict_json(config_path.read_text())
    prefix, expected = text_schema(root)
    path = directory / "model.safetensors"
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length > 32 * 1024 * 1024 or length > size - 8:
            raise ValueError("invalid safetensors header size")
        encoded = stream.read(length)
        if not encoded.startswith(b"{"):
            raise ValueError("safetensors header must start with an object")
        header = strict_json(encoded)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    for name, shape in expected.items():
        if name not in header:
            raise ValueError(f"missing text tensor: {name}")
        tensor = header[name]
        if not isinstance(tensor, dict) or tensor.get("dtype") != "BF16" or tensor.get("shape") != shape:
            raise ValueError(f"unexpected dtype/shape: {name}")
    extra = sorted(name for name in header if name.startswith(prefix) and name not in expected)
    if extra:
        raise ValueError(f"unexpected text tensors: {extra}")
    # Validate every tensor, not only selected text tensors. No cross-modality
    # aliases, holes or trailing bytes are permitted in the serialized payload.
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
                   "U32": 4, "I32": 4, "U64": 8, "I64": 8,
                   "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                   "F8_E4M3": 1, "F8_E5M2": 1}
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid tensor metadata: {name}")
        dtype, shape, offsets = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if not isinstance(dtype, str) or dtype not in dtype_bytes or not isinstance(shape, list):
            raise ValueError(f"invalid dtype/shape: {name}")
        if any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"invalid shape dimension: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError(f"invalid data offsets: {name}")
        begin, end = offsets
        if not (0 <= begin <= end <= size - 8 - length) or end - begin != dtype_bytes[dtype] * math.prod(shape):
            raise ValueError(f"invalid data offsets: {name}")
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin < cursor:
            raise ValueError("overlapping tensor data")
        if begin > cursor:
            raise ValueError("hole in safetensors payload")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("trailing safetensors payload bytes")
    byte_count = 0
    for name, shape in expected.items():
        if name not in header:
            raise ValueError(f"missing text tensor: {name}")
        tensor = header[name]
        if tensor["dtype"] != "BF16" or tensor["shape"] != shape:
            raise ValueError(f"unexpected dtype/shape: {name}")
        begin, end = tensor["data_offsets"]
        required = 2 * math.prod(shape)
        if not (0 <= begin <= end <= size - 8 - length) or end - begin != required:
            raise ValueError(f"invalid data offsets: {name}")
        byte_count += required
    extra = sorted(name for name in header if name.startswith(prefix) and name not in expected)
    if extra:
        raise ValueError(f"unexpected text tensors: {extra}")
    return {"checkpoint": str(directory.resolve()), "architecture": root["model_type"],
            "text_tensor_prefix": prefix, "validated_text_tensors": len(expected),
            "text_weight_bytes": byte_count,
            "other_tensors": len(set(header) - set(expected) - {"__metadata__"}),
            "native_lfm3_execution_ready": False,
            "validation_scope": "strict whole-file header/layout and text config/names/shapes/BF16 dtype; no payload hashing or numerical parity test"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect(args.checkpoint), indent=2) + "\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")
