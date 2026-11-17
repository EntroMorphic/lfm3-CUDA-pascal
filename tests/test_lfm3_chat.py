"""Host-only launcher contracts: fake frontend and executable, no GPU/model."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools/lfm3_chat.py"
sys.path.insert(0, str(TOOL.parent))
SPEC = importlib.util.spec_from_file_location("lfm3_chat", TOOL)
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


class Processor:
    def __init__(self, ids=None):
        self.ids = [1, 2] if ids is None else ids
        self.tokenizer = self
        self.decoded = None

    def encode_messages(self, messages, capacity, count):
        self.messages = messages
        return self.ids

    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=True)
        self.decoded = ids
        return "Hello native!"


@pytest.fixture
def config():
    result = dict(model_type="lfm3_omni", text_config=dict(model_type="lfm3", vocab_size=100, eos_token_id=9),
                  text_end_token_id=10)
    result.update(zip(chat.MODALITY_KEYS, range(11, 17)))
    return result


@pytest.fixture
def args(tmp_path, config):
    checkpoint = tmp_path / "model with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config))
    executable = tmp_path / "native executable"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "assert len(sys.argv) == 5 and sys.argv[1] == '--experimental'\n"
        "request = json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
        "result = dict(backend='native-cuda-experimental', input_ids=request['input_ids'],\n"
        "              generated_ids=[3, 10], do_sample=request['do_sample'], seed=request['seed'],\n"
        "              prefill_seconds=0.1, decode_seconds=0.2)\n"
        "pathlib.Path(sys.argv[4]).write_text(json.dumps(result))\n"
    )
    executable.chmod(0o700)
    return argparse.Namespace(checkpoint=checkpoint, executable=executable, output=tmp_path / "new output",
                              prompt="Hello", system="Respond with text.", max_new_tokens=4,
                              capacity=8, timeout=10)


def test_native_subprocess_and_decoding(args):
    processor = Processor()
    assert chat.run(args, processor) == "Hello native!"
    assert processor.decoded == [3]  # Text-end is validated then removed, not displayed.
    request = json.loads((args.output / "request.json").read_text())
    assert request == dict(input_ids=[1, 2], capacity=8, max_new_tokens=4,
                           suppress_tokens=list(range(11, 17)), stop_tokens=[9, 10],
                           **chat.SAMPLING_DEFAULTS)
    assert processor.messages == [{"role": "system", "content": "Respond with text."},
                                  {"role": "user", "content": "Hello"}]
    result = json.loads((args.output / "response.json").read_text())
    assert result["native_tokenizer_parity"] is False
    assert result["finish_reason"] == "stop"
    assert result["generated_ids"] == [3, 10]
    assert (args.output / "text.txt").read_text() == "Hello native!\n"


@pytest.mark.parametrize("ids", [[], [True], [-1], [100], [11], [1] * 9, [1] * 6])
def test_bad_or_overflowing_prompt_tokens(args, ids):
    with pytest.raises(ValueError):
        chat.run(args, Processor(ids))
    assert not args.output.exists()


@pytest.mark.parametrize("field,value", [
    ("capacity", 0), ("capacity", 32769), ("capacity", True),
    ("max_new_tokens", 0), ("max_new_tokens", -1), ("max_new_tokens", 32769),
    ("timeout", float("nan")), ("timeout", float("inf")), ("timeout", 0),
    ("prompt", " "), ("prompt", "é" * 32769), ("system", "x" * 65536),
])
def test_preflight_limits(args, field, value):
    setattr(args, field, value)
    with pytest.raises(ValueError):
        chat.run(args, Processor())
    assert not args.output.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling"])
def test_existing_output_never_overwritten(args, kind):
    if kind == "directory":
        args.output.mkdir()
        (args.output / "keep").write_text("user work")
    elif kind == "file":
        args.output.write_text("user work")
    else:
        args.output.symlink_to(args.checkpoint if kind == "symlink" else args.output.parent / "missing")
    with pytest.raises(ValueError, match="already exists"):
        chat.run(args, Processor())
    if kind == "directory":
        assert (args.output / "keep").read_text() == "user work"


@pytest.mark.parametrize("mode", ["missing", "nonexecutable", "exit", "timeout", "no_output"])
def test_executable_failures(args, mode):
    if mode == "missing":
        args.executable.unlink()
    elif mode == "nonexecutable":
        args.executable.chmod(0o600)
    else:
        body = {"exit": "raise SystemExit(7)", "timeout": "import time; time.sleep(5)",
                "no_output": "pass"}[mode]
        args.executable.write_text(f"#!{sys.executable}\n{body}\n")
        if mode == "timeout":
            args.timeout = 0.05
    with pytest.raises((OSError, ValueError, subprocess.SubprocessError)):
        chat.run(args, Processor())
    assert not (args.output / "response.json").exists()
    assert not (args.output / "text.txt").exists()


def response(ids):
    return dict(backend="native-cuda-experimental", input_ids=[1, 2], generated_ids=ids,
                prefill_seconds=0.1, decode_seconds=0.0, do_sample=False, seed=0)


@pytest.mark.parametrize("ids", [[], [True], [-1], [100], [11, 10], [9, 3], [10, 10], [3], [3] * 5])
def test_reject_bad_generated_tokens(config, ids):
    request = chat.make_request(Processor(), "Hello", "Respond with text.", config, 8, 4)
    with pytest.raises(ValueError):
        chat.validate_response(response(ids), request, 100)


@pytest.mark.parametrize("field,value", [
    ("backend", "Transformers"), ("input_ids", [2]), ("input_ids", [True, 2]),
    ("do_sample", True), ("do_sample", 0), ("seed", 1), ("seed", False),
    ("prefill_seconds", -1), ("decode_seconds", float("nan")),
    ("decode_seconds", float("inf")), ("decode_seconds", True), ("decode_seconds", 10**1000),
])
def test_reject_bad_response_metadata(config, field, value):
    request = chat.make_request(Processor(), "Hello", "Respond with text.", config, 8, 4)
    value_response = response([3, 10])
    value_response[field] = value
    with pytest.raises(ValueError):
        chat.validate_response(value_response, request, 100)


@pytest.mark.parametrize("ids,expected", [([9], []), ([10], []), ([3, 3, 3, 3], [3] * 4)])
def test_valid_stop_and_length(config, ids, expected):
    request = chat.make_request(Processor(), "Hello", "Respond with text.", config, 8, 4)
    assert chat.validate_response(response(ids), request, 100) == expected


@pytest.mark.parametrize("raw", [b"[]", b"{", b'{"a":1,"a":2}', b'{"a":NaN}',
                                  b'{"a":Infinity}', b" " * (chat.MAX_JSON_BYTES + 1), b"\xff"])
def test_malformed_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, UnicodeError)):
        chat.read_json(path)


def test_response_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_text("{}")
    path = tmp_path / "output.json"
    path.symlink_to(target)
    with pytest.raises(OSError):
        chat.read_json(path)


def test_nested_json_has_controlled_error(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text('{"a":' + '[' * 2000 + '0' + ']' * 2000 + '}')
    with pytest.raises(ValueError, match="nesting"):
        chat.read_json(path)


@pytest.mark.parametrize("payload", ["{}", "[]", "not json", '{"generated_ids":[11]}'])
def test_malformed_executable_response_has_no_success_artifact(args, payload):
    args.executable.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\n"
        f"pathlib.Path(sys.argv[4]).write_text({payload!r})\n"
    )
    with pytest.raises(ValueError):
        chat.run(args, Processor())
    assert not (args.output / "response.json").exists()
    assert not (args.output / "text.txt").exists()


def test_stdin_limit_checked_before_processor(args, monkeypatch):
    import io

    args.prompt = None
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"x" * (chat.MAX_TEXT_BYTES + 1))))
    with pytest.raises(ValueError, match="stdin"):
        chat.run(args, Processor())
    assert not args.output.exists()


def test_standalone_lfm3_policy():
    assert chat.policy(dict(model_type="lfm3", vocab_size=100, eos_token_id=[9, 10])) == (100, [], [9, 10])


def test_exact_cache_boundary(config):
    request = chat.make_request(Processor([1] * 4), "Hello", "Respond with text.", config, 8, 4)
    assert len(request["input_ids"]) + request["max_new_tokens"] == 8
    with pytest.raises(ValueError, match="capacity"):
        chat.make_request(Processor([1] * 5), "Hello", "Respond with text.", config, 8, 4)


def test_cli_help_has_no_transformers_dependency():
    result = subprocess.run([sys.executable, str(TOOL), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Native chat-tokenizer parity is not claimed" in " ".join(result.stdout.split())


def test_cli_error_has_no_traceback(args):
    result = subprocess.run([sys.executable, str(TOOL), "--checkpoint", str(args.checkpoint),
                             "--output", str(args.output), "--executable", "/nonexistent"],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "lfm3_chat:" in result.stderr and "Traceback" not in result.stderr


def test_complete_cli_uses_only_local_tokenizer(args, tmp_path):
    # Fake module deliberately has no model classes: any model loading fails.
    module_dir = tmp_path / "python modules"
    module_dir.mkdir()
    (module_dir / "transformers.py").write_text(
        "class TokenizersBackend:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, checkpoint, **kwargs):\n"
        "        assert kwargs == dict(local_files_only=True)\n"
        "        obj = cls(); obj.tokenizer = obj; return obj\n"
        "    def apply_chat_template(self, messages, **kwargs):\n"
        "        assert messages[-1]['content'] == 'Hello'\n"
        "        assert kwargs['enable_thinking'] is False\n"
        "        assert kwargs['preserve_thinking'] is False\n"
        "        return 'rendered'\n"
        "    def __call__(self, text, **kwargs):\n"
        "        assert kwargs == dict(add_special_tokens=False, truncation=False, padding=False)\n"
        "        return dict(input_ids=[1, 2])\n"
        "    def __len__(self): return 100\n"
        "    def decode(self, ids, **kwargs):\n"
        "        assert ids == [3]\n"
        "        return 'Mock CLI text (not inference evidence)'\n"
    )
    (args.checkpoint / "tokenizer_config.json").write_text('{"tokenizer_class":"TokenizersBackend"}')
    (args.checkpoint / "chat_template.jinja").write_text("official template")
    result = subprocess.run([
        sys.executable, str(TOOL), "--checkpoint", str(args.checkpoint),
        "--executable", str(args.executable), "--output", str(args.output),
        "--max-new-tokens", "4", "--capacity", "8",
    ], input="Hello", capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(module_dir)))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "Mock CLI text (not inference evidence)\n"
    assert (args.output / "response.json").is_file()


@pytest.mark.parametrize("field,value", [
    ("do_sample", 1), ("temperature", 0), ("temperature", float("nan")),
    ("temperature", True), ("top_k", -1), ("top_k", 101), ("top_k", 1.5),
    ("top_p", 0), ("top_p", 1.01), ("min_p", -0.1), ("min_p", 1.01),
    ("repetition_penalty", 0), ("repetition_penalty", float("inf")),
    ("seed", -1), ("seed", 2**31), ("seed", True),
])
def test_sampling_preflight_before_loading_or_process(args, monkeypatch, field, value):
    setattr(args, field, value)
    monkeypatch.setattr(chat, "load_frontend", lambda _: pytest.fail("loaded frontend"))
    monkeypatch.setattr(chat.subprocess, "run", lambda *a, **kw: pytest.fail("started native"))
    with pytest.raises(ValueError):
        chat.run(args)
    assert not args.output.exists()


def test_sampling_request_and_native_metadata(args):
    options = dict(do_sample=True, temperature=.7, top_k=20, top_p=.9, min_p=.05,
                   repetition_penalty=1.1, seed=2**31 - 1)
    for key, value in options.items():
        setattr(args, key, value)
    chat.run(args, Processor())
    request = json.loads((args.output / "request.json").read_text())
    result = json.loads((args.output / "response.json").read_text())
    assert all(request[key] == value for key, value in options.items())
    assert result["requested_sampling"] == options
    assert result["do_sample"] is True and result["seed"] == options["seed"]
    assert result["torch_rng_parity"] is False


@pytest.mark.parametrize("capacity,count,input_count", [(4096, 1, 4095), (32768, 1, 32767),
                                                          (32768, 32766, 2)])
def test_long_context_strict_boundary(config, capacity, count, input_count):
    request = chat.make_request(Processor([1] * input_count), "Hello", "", config, capacity, count)
    assert request.get("experimental_long_context", False) == (capacity > 4096)
    with pytest.raises(ValueError):
        chat.make_request(Processor([1] * (input_count + 1)), "Hello", "", config, capacity, count)


def test_messages_history(args):
    history = [{"role": "system", "content": "Brief."}, {"role": "user", "content": "First"},
               {"role": "assistant", "content": "Answer"}, {"role": "user", "content": "Next"}]
    args.messages = args.checkpoint / "messages.json"
    args.messages.write_text(json.dumps(history))
    args.prompt = None
    frontend = Processor()
    chat.run(args, frontend)
    assert frontend.messages == history


@pytest.mark.parametrize("history", [{}, [], [{"role": "user", "content": "<|image|>"}],
    [{"role": "user", "content": [{"type": "image", "url": "x"}]}],
    [{"role": "assistant", "content": "x", "tool_calls": []}]])
def test_invalid_messages_rejected_before_loading(args, monkeypatch, history):
    args.messages = args.checkpoint / "messages.json"
    args.messages.write_text(json.dumps(history))
    args.prompt = None
    monkeypatch.setattr(chat, "load_frontend", lambda _: pytest.fail("loaded frontend"))
    with pytest.raises(ValueError):
        chat.run(args)
    assert not args.output.exists()


def test_messages_prompt_conflict(args):
    args.messages = args.checkpoint / "missing.json"
    with pytest.raises(ValueError, match="cannot be combined"):
        chat.run(args, Processor())


def assert_failure_artifacts(output):
    assert (output / "request.json").is_file()
    assert (output / "output.json").is_file()
    assert not (output / "response.json").exists()
    assert not (output / "text.txt").exists()
    assert not list(output.glob(".success-*"))


@pytest.mark.parametrize("field", ["do_sample", "seed"])
def test_missing_native_sampling_echo_rejected(args, field):
    payload = response([3, 10])
    del payload[field]
    args.executable.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\n"
        f"pathlib.Path(sys.argv[4]).write_text({json.dumps(payload)!r})\n")
    with pytest.raises(ValueError, match=field):
        chat.run(args, Processor())
    assert_failure_artifacts(args.output)


@pytest.mark.parametrize("field", ["sampling_policy", "sampled_policy", "sampling"])
@pytest.mark.parametrize("value", [{"do_sample": True, "seed": 123}, "\ud800", None])
def test_unsupported_native_sampling_metadata_rejected(args, field, value):
    payload = response([3, 10])
    payload[field] = value
    args.executable.write_text(
        f"#!{sys.executable}\nimport pathlib, sys\n"
        f"pathlib.Path(sys.argv[4]).write_text({json.dumps(payload)!r})\n")
    with pytest.raises(ValueError, match="unsupported native sampling metadata"):
        chat.run(args, Processor())
    assert_failure_artifacts(args.output)


@pytest.mark.parametrize("decoded", ["\ud800", b"not text", "x" * (chat.MAX_JSON_BYTES + 1)],
                         ids=["surrogate", "non-string", "oversized"])
def test_invalid_decoder_text_has_no_success_artifacts(args, monkeypatch, decoded):
    frontend = Processor()
    monkeypatch.setattr(frontend, "decode", lambda *a, **kw: decoded)
    with pytest.raises((ValueError, UnicodeError)):
        chat.run(args, frontend)
    assert_failure_artifacts(args.output)


@pytest.mark.parametrize("name", ["response.json", "text.txt"])
def test_staging_write_failure_has_no_success_artifacts(args, monkeypatch, name):
    original = Path.write_bytes

    def fail_write(path, payload):
        if path.name == name and path.parent.name.startswith(".success-"):
            with path.open("wb") as stream:
                stream.write(payload[:1])
            raise OSError("injected write failure")
        return original(path, payload)

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(OSError, match="injected write failure"):
        chat.run(args, Processor())
    assert_failure_artifacts(args.output)


@pytest.mark.parametrize("name", ["response.json", "text.txt"])
def test_publication_failure_rolls_back_success_artifacts(args, monkeypatch, name):
    original = os.link

    def fail_link(source, destination):
        if destination.name == name:
            raise OSError("injected publication failure")
        return original(source, destination)

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected publication failure"):
        chat.run(args, Processor())
    assert_failure_artifacts(args.output)


@pytest.mark.parametrize("value", [{"text": "\ud800"}, {"value": float("nan")}])
def test_write_json_serializes_before_opening(tmp_path, value):
    destination = tmp_path / "response.json"
    with pytest.raises((ValueError, UnicodeError)):
        chat.write_json(destination, value)
    assert not destination.exists()


@pytest.mark.parametrize("name", ["response.json", "text.txt"])
def test_success_publication_never_overwrites_existing_file(tmp_path, name):
    destination = tmp_path / name
    destination.write_text("user work")
    with pytest.raises(FileExistsError):
        chat.publish_success(tmp_path, {"text": "hello"}, "hello")
    assert destination.read_text() == "user work"
    other = "text.txt" if name == "response.json" else "response.json"
    assert not (tmp_path / other).exists()
    assert not list(tmp_path.glob(".success-*"))
