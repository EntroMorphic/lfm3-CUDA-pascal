"""Local tokenizer-only frontend -> experimental native CUDA -> decoded text.

No Transformers language model, AutoProcessor, or audio codec is loaded.
Native chat-tokenizer parity is not claimed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

try:
    from lfm3_text_frontend import load_frontend, validate_messages
except ModuleNotFoundError:
    from tools.lfm3_text_frontend import load_frontend, validate_messages


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 65536
MAX_JSON_BYTES = 1024 * 1024
MODALITY_KEYS = (
    "audio_token_id", "audio_out_token_id", "image_token_id", "video_token_id",
    "mixed_start_token_id", "audio_start_token_id",
)


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def token_ids(value, name, vocabulary, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{name}: expected bounded token list")
    return [integer(token, name, 0, vocabulary - 1) for token in value]


def read_json(path, *, object_required=True):
    # Reject response symlinks/FIFOs and oversized files before reading into RAM.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise ValueError(f"{path}: expected regular JSON file <= {MAX_JSON_BYTES} bytes")
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON file too large")
    # Do not rely on Python-version-dependent parser recursion limits. Scan the
    # UTF-8 text outside strings before asking the JSON decoder to allocate it.
    quoted = escaped = False
    depth = 0
    for char in raw.decode("utf-8"):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 64:
                raise ValueError("JSON nesting exceeds supported depth")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds supported depth") from exc
    if object_required and not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def policy(config):
    omni = config.get("model_type") == "lfm3_omni"
    text = config.get("text_config") if omni else config
    if not isinstance(text, dict) or text.get("model_type") != "lfm3":
        raise ValueError("checkpoint must be LFM3 or LFM3-Omni")
    vocabulary = integer(text.get("vocab_size"), "vocab_size", 1, 2**31 - 1)
    eos = text.get("eos_token_id")
    stop = [] if eos is None else eos if isinstance(eos, list) else [eos]
    suppress = []
    if omni:
        suppress = [config.get(key) for key in MODALITY_KEYS]
        stop = stop + [config.get("text_end_token_id")]
    suppress = token_ids(suppress, "modality token", vocabulary, 6)
    stop = token_ids(stop, "stop token", vocabulary, vocabulary)
    if set(stop) & set(suppress) or len(set(suppress)) == vocabulary:
        raise ValueError("conflicting text generation policy")
    return vocabulary, sorted(set(suppress)), sorted(set(stop))


SAMPLING_DEFAULTS = dict(do_sample=False, temperature=1.0, top_k=0, top_p=1.0,
                         min_p=0.0, repetition_penalty=1.0, seed=0)


def sampling_options(options, vocabulary):
    result = dict(SAMPLING_DEFAULTS)
    if options is not None:
        if set(options) - set(result):
            raise ValueError("unknown sampling option")
        result.update(options)
    if type(result["do_sample"]) is not bool:
        raise ValueError("do_sample must be boolean")
    integer(result["top_k"], "top_k", 0, vocabulary)
    integer(result["seed"], "seed", 0, 2**31 - 1)
    for key in ("temperature", "top_p", "min_p", "repetition_penalty"):
        value = result[key]
        if (type(value) not in (int, float) or not -sys.float_info.max <= value <= sys.float_info.max
                or not math.isfinite(value) or value < 0 or (key != "min_p" and value == 0)
                or (key in ("top_p", "min_p") and value > 1)):
            raise ValueError(f"invalid {key}")
    return result


def make_request(frontend, prompt, system, config, capacity, count, *, messages=None, sampling=None):
    integer(capacity, "capacity", 1, 32768)
    integer(count, "max_new_tokens", 1, 32768)
    if messages is None:
        if not prompt.strip() or len((prompt + system).encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError("prompt must be nonblank; combined prompt/system limit is 65536 UTF-8 bytes")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    messages = validate_messages(messages)
    vocabulary, suppress, stop = policy(config)
    sampling = sampling_options(sampling, vocabulary)
    ids = frontend.encode_messages(messages, capacity, count)
    ids = token_ids(ids, "input_ids", vocabulary, capacity)
    if not ids or len(ids) + count > capacity:
        raise ValueError("prompt plus generation exceeds native cache capacity; no truncation performed")
    if set(ids) & set(suppress):
        raise ValueError("multimodal placeholder input is unsupported in text-only mode")
    request = dict(input_ids=ids, max_new_tokens=count, capacity=capacity,
                   suppress_tokens=suppress, stop_tokens=stop, **sampling)
    if capacity > 4096:
        request["experimental_long_context"] = True
    return request


def validate_response(response, request, vocabulary):
    if response.get("backend") != "native-cuda-experimental":
        raise ValueError("unexpected native backend")
    if any(key in response for key in ("sampling_policy", "sampled_policy", "sampling")):
        raise ValueError("unsupported native sampling metadata")
    for key in ("do_sample", "seed"):
        if key not in response or type(response[key]) is not type(request[key]) or response[key] != request[key]:
            raise ValueError(f"native {key} does not match request")
    echoed = token_ids(response.get("input_ids"), "response input_ids", vocabulary, request["capacity"])
    if echoed != request["input_ids"]:
        raise ValueError("native input_ids do not match request")
    generated = token_ids(response.get("generated_ids"), "generated_ids", vocabulary, request["max_new_tokens"])
    if not generated or set(generated) & set(request["suppress_tokens"]):
        raise ValueError("empty generation or forbidden modality token in native response")
    stops = set(request["stop_tokens"])
    if any(token in stops for token in generated[:-1]):
        raise ValueError("native response continues after stop token")
    if len(generated) < request["max_new_tokens"] and generated[-1] not in stops:
        raise ValueError("native response ended early without stop token")
    for key in ("prefill_seconds", "decode_seconds"):
        value = response.get(key)
        if (type(value) not in (int, float) or value < 0 or value > sys.float_info.max
                or not math.isfinite(value)):
            raise ValueError(f"invalid {key}")
    return generated[:-1] if generated[-1] in stops else generated


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def write_json(path, value):
    payload = json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)


def publish_success(output, metadata, text):
    # Finish serialization before opening any success artifact. Stage both files
    # before exclusive hard-link publication (never overwrite an existing path).
    payloads = {"response.json": json_bytes(metadata), "text.txt": (text + "\n").encode("utf-8")}
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix=".success-", dir=output) as staging:
            for name, payload in payloads.items():
                (Path(staging) / name).write_bytes(payload)
            for name in payloads:
                destination = output / name
                os.link(Path(staging) / name, destination)
                published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink()
        raise


def run(args, frontend=None):
    checkpoint = args.checkpoint.resolve(strict=True)
    executable = args.executable.resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("native executable is not an executable file")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 86400:
        raise ValueError("timeout must be finite and in (0, 86400] seconds")
    # Never reuse a directory, even if empty, or follow an output-directory symlink.
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output directory already exists; choose a fresh path")
    config = read_json(checkpoint / "config.json")
    integer(args.capacity, "capacity", 1, 32768)
    integer(args.max_new_tokens, "max_new_tokens", 1, 32768)
    if args.max_new_tokens >= args.capacity:
        raise ValueError("capacity must fit a nonempty prompt plus max_new_tokens")
    sampling = sampling_options({key: getattr(args, key, default)
                                 for key, default in SAMPLING_DEFAULTS.items()}, policy(config)[0])
    messages_path = getattr(args, "messages", None)
    messages = None
    prompt = args.prompt
    if messages_path is not None:
        if prompt is not None or args.system != "Respond with text.":
            raise ValueError("--messages cannot be combined with --prompt or a custom --system")
        messages = validate_messages(read_json(messages_path, object_required=False))
    elif prompt is None:
        raw = sys.stdin.buffer.read(MAX_TEXT_BYTES + 1)
        if len(raw) > MAX_TEXT_BYTES:
            raise ValueError("stdin prompt exceeds 65536 bytes")
        prompt = raw.decode("utf-8")
    if messages is None and (not prompt.strip() or len((prompt + args.system).encode("utf-8")) > MAX_TEXT_BYTES):
        raise ValueError("blank or oversized prompt/system text")
    policy(config)
    if frontend is None:
        frontend = load_frontend(checkpoint)
    request = make_request(frontend, prompt, args.system, config, args.capacity, args.max_new_tokens,
                           messages=messages, sampling=sampling)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = args.output.resolve()
    write_json(output / "request.json", request)
    # No shell and no unbounded captured subprocess logs. Failure artifacts remain
    # for diagnosis, but no decoded success report is written on any failure.
    subprocess.run([str(executable), "--experimental", str(checkpoint),
                    str(output / "request.json"), str(output / "output.json")],
                   check=True, timeout=args.timeout, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    response = read_json(output / "output.json")
    ids = validate_response(response, request, policy(config)[0])
    text = frontend.decode(ids, skip_special_tokens=True)
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("invalid or oversized decoded text")
    metadata = dict(
        backend="native-cuda-experimental", preprocessing="official-local-tokenizer-only",
        native_tokenizer_parity=False, generated_ids=response["generated_ids"], text=text,
        finish_reason="stop" if len(ids) < len(response["generated_ids"]) else "length",
        requested_sampling=sampling, torch_rng_parity=False,
    )
    for key in ("do_sample", "seed"):
        metadata[key] = response[key]
    publish_success(output, metadata, text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="trusted local checkpoint directory")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/cmake/liquid-lfm3")
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory (must not exist)")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt", help="user text; if omitted, read bounded UTF-8 stdin")
    inputs.add_argument("--messages", type=Path, help="JSON array of complete text conversation history")
    parser.add_argument("--system", default="Respond with text.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        text = run(args)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"lfm3_chat: {exc}", file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
