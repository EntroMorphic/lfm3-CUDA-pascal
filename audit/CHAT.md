# Experimental text CLI

`tools/lfm3_chat.py` uses `load_frontend` from `tools/lfm3_text_frontend.py`:
the official local fast tokenizer and checkpoint Jinja template, without
AutoProcessor, audio codec, model loading, or checkpoint Python execution.
All inference runs in the custom CUDA executable. This is a Python-fronted
native engine, **not a fully native tokenizer**; native tokenizer parity remains
unestablished. Default single-turn formatting remains system `Respond with text.`
plus the user prompt. The template disables thinking and preserves no old thinking.

From `liquid-CUDA`, with the checkpoint tokenizer dependencies installed:

```bash
python3 tools/lfm3_chat.py \
  --checkpoint ../models/LFM3-Omni-2B \
  --executable build/cmake/liquid-lfm3 \
  --prompt 'Say hello in one short sentence.' \
  --max-new-tokens 32 --capacity 512 \
  --output ../tmp/chat-smoke-001
```

The **GPU smoke passed on Thor**, producing `Hello! How can I help you today?`
through native CUDA inference. Its artifacts are in project-root
`tmp/chat-smoke-001/`. The output path
must not exist. For interactive typing, omit `--prompt`, type UTF-8 text on stdin,
then EOF. The native interface is exactly:

```text
liquid-lfm3 --experimental MODEL_DIRECTORY request.json output.json
```

The final frozen runtime also completed the original reflective-recursion prompt
through this launcher. Its actual 86-token response and raw request/output are
saved locally in project-root `tmp/lfm3-native-final/`.

Artifacts are `request.json`, untouched native `output.json`, decoded
`response.json`, and `text.txt`. Stdout is decoded text. `response.json` explicitly
records official preprocessing and `native_tokenizer_parity: false`. Copy actual
GPU outputs back to the local project root `tmp/` when running remotely; never
present mocked host-test text as inference evidence.

## Multi-turn history and sampling

Use `--messages history.json` instead of `--prompt`/stdin for a JSON array:

```json
[
  {"role": "system", "content": "Answer briefly."},
  {"role": "user", "content": "What is CUDA?"},
  {"role": "assistant", "content": "A GPU computing platform."},
  {"role": "user", "content": "Who develops it?"}
]
```

The supplied history is complete; no default system message is inserted.
A custom `--system` is rejected with `--messages`. Every invocation prefills
the entire history: there is **no persistent server/session cache**. The shared
frontend validates roles, text-only content, reasoning fields and plain tool
responses. Tool calls/definitions, reserved chat placeholders and multimodal
content are rejected rather than silently dropped.

Greedy remains default. Sampling controls are `--do-sample` (or
`--no-do-sample`), `--temperature` (default 1), `--top-k` (0 disables),
`--top-p` (default 1), `--min-p` (default 0), `--repetition-penalty` (default 1),
and `--seed` (default 0, integer 0..2147483647). Temperature and penalty must be
finite and positive; top-p is in (0,1], min-p in [0,1], and top-k is an integer
0..vocabulary size. All are validated before launching native inference.
Repetition history contains generated IDs only, not prompt tokens. Requests carry
all controls; decoded metadata records requested settings and native `do_sample`
and `seed`. Both native fields are mandatory and must match the request with exact
types (boolean and integer). Optional nested `sampling_policy`, `sampled_policy`,
and `sampling` fields are unsupported and rejected, never forwarded. No PyTorch
RNG/identical sampled-sequence parity is claimed.

## Bounds and failure policy

- Prompt plus system text: 65,536 UTF-8 bytes, nonblank user prompt. Stdin is read
  with a byte bound. No implicit input truncation.
- Capacity defaults consciously to 4,096 to avoid allocating a 32K cache for
  ordinary short requests. Set `--capacity 32768` for the full serving limit.
  Capacity and generation count are positive integers up to 32,768; the complete
  `len(input_ids) + max_new_tokens <= capacity <= 32768` budget is enforced,
  **without a final-token discount**. Requests with capacity above 4,096 carry
  `experimental_long_context: true` as required by the current native runtime.
  The 32K SDPA numerical gate passed (max-abs 0.1875, relative RMS 0.00313;
  8192 prefill, 32767+1 decode, 32768 prefill); this is not a launcher GPU test.
  IDs must be real integers, not bools, and in vocabulary.
- Omni placeholder/mixed/audio-start IDs are suppressed explicitly in requests;
  EOS and text-end stop explicitly. Placeholder inputs are rejected. Responses
  with suppressed IDs, tokens after stop, unexplained early termination, wrong
  echoed input, wrong backend, invalid IDs or invalid timings fail before decoding.
  A final stop token remains in raw evidence but is removed before text decoding.
- Existing files, directories and symlinks are never reused as the output
  directory. It is created exclusively with mode 0700. No overwrite switch exists.
  Failed runs retain their fresh request/raw output; they cannot produce decoded
  success artifacts. A retry must use another fresh directory.
  JSON is fully serialized with strict finite numbers and encoded as UTF-8 before
  opening its destination. Both success files are staged before exclusive
  hard-link publication; caught write/publication failures roll back any newly
  published success file and remove staging files, retaining request/raw output.
  This is not a two-file atomic transaction against process kill or power loss.
- Native invocation has no shell, a finite timeout (default 300 seconds), checked
  exit status, and discarded stdout/stderr rather than unbounded captured logs.
  To diagnose native errors, rerun the recorded request directly with a **new**
  raw-output path. Executable and checkpoint code must be trusted: this is not a
  sandbox against a malicious executable or checkpoint Python.
- JSON inputs/responses (including the history file) are bounded to 1 MiB and must be regular nonsymlink files.
  Duplicate keys, nonfinite constants, excess nesting, malformed JSON and invalid
  response schema fail closed. Decoded text is also bounded to 1 MiB.

## Host validation

```bash
python3 -m pytest tests/test_lfm3_chat.py -q
PYTHONPATH=tests python3 -m pytest tests audit/test_host_contracts.py -q
```

Tests use a fake frontend and an actual fake subprocess, with no weights or GPU.
Host verification after review remediation: frontend **52 passed**, launcher
**121 passed** separately; combined suite **449 passed in 31.66s**.
They exercise the exact argv contract, text-end/EOS and length stopping, malformed
responses, numeric/byte/token/cache limits, modality leakage, executable failures,
timeouts, symlink rejection and output overwrite protection. Regression coverage
also includes missing sampling echoes, contradictory/surrogate nested metadata,
invalid decoder Unicode, serialization errors, partial staging writes, and
publication failure rollback. These are host checks, not new GPU evidence.

The serving path uses `local_files_only=True` with the generic fast tokenizer;
it no longer needs the Omni processor's dynamic-module/audio-codec dependencies.
The standalone frontend's optional official-processor parity check is separate
from serving and may load the codec. This launcher does not invoke that check.
