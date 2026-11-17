"""Tokenizer-only LFM3 text frontend; no processor, codec, model, or CUDA import.

32K means exactly 32768 *combined* prompt and requested output tokens (the
model-card serving limit), not the architecture's 128000 position setting.
The checkpoint Jinja template, including BOS and empty-think headers, is used
verbatim. No hand-built chat formatting, truncation, or final-token discount.
"""
import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import re


MAX_CONTEXT_TOKENS = 32768
# A separate generous input bound prevents unbounded work before tokenization.
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_MESSAGES = 4096
RESERVED = re.compile(r"<\|[^<>]*\|>|CONTINUE_FINAL_MESSAGE_TAG ")


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if RESERVED.search(value):
        raise ValueError(f"{name}: reserved chat/multimodal placeholder is not text input")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name}: invalid Unicode") from exc
    if size > MAX_TEXT_BYTES:
        raise ValueError(f"{name}: text byte limit exceeded")
    return size


def validate_messages(messages):
    """Return a defensive copy of the supported, non-batched text conversation.

    Plain tool responses (including same-id consecutive response merging) work.
    Tool definitions, calls, deferred references and additional_tools are not
    supported; rejecting them is preferable to silently dropping template data.
    Only an initial system message is accepted. Final assistant messages are
    completed turns, never continuation requests.
    """
    if not isinstance(messages, (list, tuple)) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError(f"messages must contain 1..{MAX_MESSAGES} messages")
    result = []
    total_bytes = 0
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError("each message must be a mapping")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError("role must be system, user, assistant, or tool")
        if role == "system" and index != 0:
            raise ValueError("system is supported only as the first message")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed.update(("thinking", "reasoning", "reasoning_content"))
        if role == "tool":
            allowed.add("tool_call_id")
        if set(message) - allowed:
            raise ValueError("unsupported message fields (tool calls/definitions are not supported)")
        if "content" not in message:
            raise ValueError("every message requires text content")
        copied = dict(message)
        content = message["content"]
        if isinstance(content, str):
            total_bytes += _text(content, "content")
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, Mapping) or set(part) != {"type", "text"} or part["type"] != "text":
                    raise ValueError("content parts must be text-only {type: text, text: string}")
                total_bytes += _text(part["text"], "content part")
                parts.append(dict(part))
            # The checkpoint template concatenates parts without separators.
            # Individually harmless fragments can form a reserved delimiter.
            _text("".join(part["text"] for part in parts), "joined content")
            copied["content"] = parts
        else:
            raise ValueError("content must be a string or a list of text parts; multimodal input is unsupported")
        reasoning = [key for key in ("thinking", "reasoning", "reasoning_content") if key in message]
        if len(reasoning) > 1:
            raise ValueError("use only one assistant reasoning field")
        for key in reasoning:
            total_bytes += _text(message[key], key)
        if "tool_call_id" in message:
            total_bytes += _text(message["tool_call_id"], "tool_call_id")
            if not message["tool_call_id"]:
                raise ValueError("tool_call_id must not be empty")
        if total_bytes > MAX_TEXT_BYTES:
            raise ValueError("conversation text byte limit exceeded")
        result.append(copied)
    return result


class TextFrontend:
    def __init__(self, tokenizer, chat_template):
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ValueError("checkpoint chat_template.jinja must not be empty")
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def render_messages(self, messages, *, tools=None):
        if tools is not None:
            raise ValueError("tool definitions are not supported by this text frontend")
        messages = validate_messages(messages)
        return self.tokenizer.apply_chat_template(
            messages, chat_template=self.chat_template, tokenize=False,
            add_generation_prompt=True, enable_thinking=False, preserve_thinking=False,
        )

    def encode_messages(self, messages, capacity, max_new_tokens, *, tools=None):
        """Return flat input IDs, reserving every requested output token.

        ``capacity`` is the native serving allocation and cannot exceed 32768.
        ``len(input_ids) + max_new_tokens <= capacity`` is mandatory, even when
        the last generated token would not need another forward pass.
        """
        _integer(capacity, "capacity", 1, MAX_CONTEXT_TOKENS)
        _integer(max_new_tokens, "max_new_tokens", 1, MAX_CONTEXT_TOKENS)
        if max_new_tokens >= capacity:
            raise ValueError("capacity must fit a nonempty prompt plus max_new_tokens")
        rendered = self.render_messages(messages, tools=tools)
        # ProcessorMixin also suppresses added BOS when the template starts with
        # bos_token. This template already renders BOS; never insert it twice.
        encoded = self.tokenizer(rendered, add_special_tokens=False, truncation=False,
                                 padding=False)
        ids = encoded["input_ids"]
        if not isinstance(ids, list) or not ids:
            raise ValueError("tokenizer must return a nonempty flat list of input_ids")
        vocabulary = len(self.tokenizer)
        ids = [_integer(token, "input token", 0, vocabulary - 1) for token in ids]
        if len(ids) + max_new_tokens > capacity:
            raise ValueError(f"prompt ({len(ids)}) + output ({max_new_tokens}) exceeds capacity ({capacity})")
        return ids

    def decode(self, ids, *, skip_special_tokens=True):
        if not isinstance(ids, (list, tuple)) or len(ids) > MAX_CONTEXT_TOKENS:
            raise ValueError("decode requires a bounded flat token list")
        if type(skip_special_tokens) is not bool:
            raise ValueError("skip_special_tokens must be boolean")
        vocabulary = len(self.tokenizer)
        checked = [_integer(token, "decode token", 0, vocabulary - 1) for token in ids]
        return self.tokenizer.decode(checked, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=False)


def load_frontend(checkpoint):
    """Load only the official local tokenizer, without executing checkpoint code."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    template = (checkpoint / "chat_template.jinja").read_text(encoding="utf-8")
    # Lazy import keeps module import and fake-tokenizer host tests torch-free.
    import transformers

    config = json.loads((checkpoint / "tokenizer_config.json").read_text(encoding="utf-8"))
    if config.get("tokenizer_class") not in ("TokenizersBackend", "PreTrainedTokenizerFast"):
        raise ValueError("expected the checkpoint's official generic fast tokenizer")
    # Transformers 5 renamed its generic fast tokenizer to TokenizersBackend.
    # Transformers 4's official class reads the same tokenizer.json verbatim.
    tokenizer_class = getattr(transformers, "TokenizersBackend", None)
    kwargs = {}
    if tokenizer_class is None:
        tokenizer_class = transformers.PreTrainedTokenizerFast
        if config.get("extra_special_tokens") == []:
            # v4 represents an empty set of named extra tokens as a mapping.
            kwargs["extra_special_tokens"] = {}
    tokenizer = tokenizer_class.from_pretrained(str(checkpoint), local_files_only=True, **kwargs)
    return TextFrontend(tokenizer, template)


PARITY_CASES = [
    [{"role": "user", "content": "Hello"}],
    [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
    [{"role": "system", "content": "Answer briefly.\nUnicode ✓"},
     {"role": "user", "content": "café 中文 👩🏽‍💻 e\u0301\n\t"}],
    [{"role": "user", "content": "First"},
     {"role": "assistant", "content": "<think>old private text</think>  answer "},
     {"role": "user", "content": [{"type": "text", "text": "Next"}]}],
    [{"role": "user", "content": "hi"},
     {"role": "assistant", "content": "answer", "thinking": ""}],
    [{"role": "tool", "content": "résultat", "tool_call_id": "a"},
     {"role": "tool", "content": "second", "tool_call_id": "a"}],
]


def check_official_parity(checkpoint):
    """Opt-in CPU check; official processor may load its bundled audio codec."""
    frontend = load_frontend(checkpoint)
    # This is intentionally NOT part of the serving/load_frontend path.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=True,
    )
    for index, messages in enumerate(PARITY_CASES):
        messages = validate_messages(messages)
        rendered = frontend.render_messages(messages)
        official_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, preserve_thinking=False,
        )
        # Processor APIs may return a batch even for one conversation.
        if isinstance(official_text, list) and len(official_text) == 1:
            official_text = official_text[0]
        if rendered.encode("utf-8") != official_text.encode("utf-8"):
            raise AssertionError(f"case {index}: rendered UTF-8 bytes differ")
        official_ids = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            enable_thinking=False, preserve_thinking=False,
        )["input_ids"]
        if official_ids and isinstance(official_ids[0], list):
            official_ids = official_ids[0]
        actual = frontend.encode_messages(messages, MAX_CONTEXT_TOKENS, 1)
        if actual != official_ids:
            raise AssertionError(f"case {index}: token IDs differ")
    return {"cases": len(PARITY_CASES), "utf8_bytes_equal": True, "token_ids_equal": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--check-official-parity", action="store_true", required=True,
                        help="opt in to loading the official processor/codec for parity only")
    args = parser.parse_args()
    print(json.dumps(check_official_parity(args.checkpoint), sort_keys=True))
