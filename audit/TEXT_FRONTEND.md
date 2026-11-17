# Tokenizer-only text frontend

`tools/lfm3_text_frontend.py` provides:

```python
frontend = load_frontend(checkpoint)
input_ids = frontend.encode_messages(messages, capacity=32768, max_new_tokens=256)
text = frontend.decode(generated_ids)
```

`load_frontend` loads only the official generic Transformers tokenizer and the
checkpoint's `chat_template.jinja`. It does not import the checkpoint's processor
or execute checkpoint Python. Imports are lazy for host-only tests. Transformers
5 uses `TokenizersBackend`; Transformers 4 uses its earlier official name,
`PreTrainedTokenizerFast`, and an empty mapping instead of an empty list for
`extra_special_tokens`. Both read the unchanged checkpoint `tokenizer.json`.

## Contract

- 32K means **32768 combined prompt and requested output tokens**, per the
  model-card serving limit. The architecture's 128000 position value and the
  tokenizer's effectively infinite `model_max_length` do not raise this limit.
- Require `len(input_ids) + max_new_tokens <= capacity <= 32768`. No truncation,
  padding, or last-token optimization. The requested output is positive.
- Apply the unchanged checkpoint template, with `add_generation_prompt=True`,
  `enable_thinking=False`, `preserve_thinking=False`. It emits BOS and the
  assistant header `<|im_start|>assistant\n<think></think>` itself. Encoding uses
  `add_special_tokens=False` to avoid duplicate BOS.
- Accept user/assistant/tool and an optional first system message, including
  empty strings and lists of strictly typed text parts. Preserve whitespace and
  Unicode; reject invalid Unicode. The template handles prior assistant thinking
  and consecutive plain tool responses with matching `tool_call_id`.
- Assistant endings are completed turns, not continuation requests. One explicit
  assistant thinking/reasoning field is supported; ambiguous aliases are rejected.
- Reject unknown fields, tool definitions/calls, additional/deferred tools, all
  nontext content parts, reserved `<|...|>` placeholders, and the template's
  continuation sentinel. These are deliberate scope constraints, not silent
  approximations of official tool or multimodal behavior.
  Text-part lists are also validated after concatenation, exactly as rendered by
  the template: splitting a delimiter or continuation sentinel across parts
  cannot bypass validation or inject a new role.
- Plain tool responses are supported for importing textual histories; this is
  **not** a tool-execution frontend. No schemas or call generation are supported.
- Input text has a separate 4 MiB aggregate UTF-8 safety bound and at most 4096
  messages. Decode validates flat vocabulary-bounded IDs and disables cleanup.

## Validation

Host CPU tests (no processor, model, CUDA, or SSH):

```bash
python -m pytest -q tests/test_lfm3_text_frontend.py
```

Official processor comparison, to run in the prepared Thor environment:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tools/lfm3_text_frontend.py \
  --checkpoint ../models/LFM3-Omni-2B --check-official-parity
```

Only this opt-in comparison loads `AutoProcessor` with trusted local checkpoint
code (which may load the audio codec on CPU). It checks exact rendered UTF-8
bytes and token IDs for six cases: basic text, empty system/user, system plus
non-ASCII text, multi-turn historical reasoning, explicit empty thinking, and
same-ID tool response merging. Any mismatch fails; no parity claim is made on a
failed or unavailable comparison.

**Thor verification:** the six official processor comparisons passed with
`CUDA_VISIBLE_DEVICES=` and offline mode: `token_ids_equal: true` and
`utf8_bytes_equal: true`. The comparison itself loads the processor/codec only as
an oracle; normal tokenizer-only frontend loading does not.

The local workstation CPU attempt loaded the tokenizer successfully, but official processor
initialization failed before comparison with `FileNotFoundError` for cached
`configuration_lfm3_omni.py` in the Hugging Face dynamic-module cache. This is
not a successful processor parity result. No SSH/GPU validation was attempted.
The local actual-tokenizer tests independently check exact headers, Unicode
round-trip, historical reasoning stripping, tool merge rendering, and direct
tokenizer-template token equality for all six cases.

Review-remediation host verification: **52 frontend tests passed** separately,
including every two-fragment split of role-injection delimiters, a multimodal
placeholder and the continuation sentinel (with an intervening empty part).
Benign Unicode text parts remain preserved. The launcher suite independently
passed **121 tests**; the combined host suite passed **449 tests**. No new
SSH/GPU or official-processor validation was performed for these fixes.
