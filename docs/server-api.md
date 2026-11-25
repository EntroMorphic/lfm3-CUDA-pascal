# Server API

`liquid-lfm3-server --experimental MODEL_DIR [--port N] [--token-file FILE] [--speech-rows N] [--total-capacity N] [...]`

**On the GTX 1080 (this port).** The two servers the build scripts make differ:

| Build | Text | `/speech` (speech out, audio in) | Images |
|---|---|---|---|
| `tools/pascal/build_serve.sh` (`-DLFM3_TEXT_ONLY=1`) | yes | 501 `speech_unsupported` | 501 |
| `tools/pascal/build_speech.sh` (`-DLFM3_TEXT_ONLY=0 -DLFM3_VISION=1`) | yes | yes | yes, in `/speech` ([images-v1](../validation/images-v1/)) |

Memory decides three defaults (validation/speech-v1 rule 12, peak at most 7,740 MiB of the card's 8 GiB):

- `--speech-rows` (1–16) defaults to **1**: one `/speech` generates at a time and the rest wait in the queue.
- `--total-capacity`, the tokens all text sessions may hold together, defaults to **32,768** without speech and, when
  the server serves speech, to **29,696** at one speech row, 2,048 less for each further row, never below 1,024 (8
  rows: 15,360; 15 and 16 rows: 1,024). An explicit `--total-capacity` is the operator's: larger values with speech were not
  measured to fit, and running out of device memory fails the server closed (503 to everything; restart it).
- Speech is served only with `--engine-timeout` at least 300 s, and not with `LFM3_COMPACT_WINDOWS=1` or
  `LFM3_LOSSLESS=1` in the environment (measurement switches the speech engine cannot run with); otherwise `/speech`
  gets 501 and the defaults are those without speech.
- `--image-cache-mb` defaults to **32** (the Thor engine's 512): the engine's image cache is device memory (below).

- Loopback only. Once listening it prints one line, `{"ready":true,"port":N,"total_capacity":N,"speech_rows":N}`
  (`--port 0` picks a free port; `speech_rows` 0: speech not served).
- Every `POST` needs `Authorization: Bearer TOKEN`, `Content-Type: application/json` and one `Content-Length`. The
  token comes from `--token-file`, `LFM3_SERVER_TOKEN`, or `~/.config/liquid-lfm3/token` (created on first start);
  `tools/lfm3_token.py` reads the same place for clients.
- Bodies are at most 4 MiB, parsed strictly: duplicate keys, non-finite numbers, invalid UTF-8, nesting over 64 and
  unknown fields are refused with a 400 naming the problem.
- At most `--handlers` (default 8) connections are served at once. One more is answered `429 connection_limit` at
  once, without its request being read (this port; the Thor engine closes it without a reply, which a client sees as
  a reset); retry it. Past 64 such refusals at once, connections are closed.
- `SIGTERM` stops admission, cancels running generations at their next token, and exits 0.

The authoritative contract is the comment above each handler in `native/include/lfm/lfm3_http.hpp`.

## Text: sessions

| Endpoint | Body | Effect |
|---|---|---|
| `POST /sessions` | `{"capacity": 4096}` | Creates a session with its own KV cache (2 to the total capacity, which all sessions share); returns `session_id` |
| `POST /generate` | `{"session_id", "messages": [...], "max_new_tokens", "sampling": {...}}` | Generates on the session, reusing its cached prefix whenever the prompt extends it: the new prompt tokens in one pass, each computed as a prefill of the whole prompt computes it ([continuation-v1](../validation/continuation-v1/); the parity build, as the Thor engine: at most 1,024 new tokens, one at a time, more prefilled in full) |
| `POST /generate_batch` | `{"requests": [A, B]}` | Two sessions decoded together on the GPU; at most 2,048 `max_new_tokens` a lane (the batch is one engine call, bounded by the engine timeout; 501 `batching_unavailable` with `--engine-timeout` under 300 s) |
| `POST /sessions/ID/reset` | `{}` | Clears the session's cache |
| `POST /sessions/ID/close` | `{}` | Releases the session |
| `POST /sessions/ID/cancel` | `{}` | Cancels a running generation at its next token (202) |

`messages` follows the chat-completions shape (`system`, `user`, `assistant`, `tool` roles; string or text-part
content). Replies carry `text`, `generated_ids`, `finish_reason` (`stop` or `length`) and cache accounting.

## Speech: `POST /speech`

One request, no session; concurrent requests share the engine's speech batch (`--speech-rows`, default 1 on the
GTX 1080, 8 on the Thor engine), each reply byte-identical to the same request alone.

```json
{
  "messages": [{"role": "user", "content": [
      {"type": "audio", "audio": "<base64 16 kHz mono WAV>"},
      {"type": "text", "text": "Please answer briefly."}]}],
  "mode": "interleaved",
  "max_new_tokens": 512,
  "seed": 0,
  "stream": false
}
```

| Field | Values |
|---|---|
| `mode` | `audio` (speech only), `interleaved` (text and speech, the model's voice-chat mode), `text` |
| `max_new_tokens` | 1–4095, per chunk |
| `seed` | 0–2³¹−1 |
| `split` | audio mode, text input: read long text sentence by sentence (default true there) |
| `normalize` | audio mode: speak numbers and abbreviations as words (default true with no system prompt or one naming American or British English) |
| `stream` | true: newline-delimited JSON events as the engine generates |

User messages may hold up to 64 audio parts (base64 WAV, 16 kHz mono, 16-bit PCM or 32-bit float) and up to 16 image
parts, `{"type": "image", "image": "<base64 JPEG or PNG>"}`, in any order with text. For spoken replies to spoken input
use `interleaved`.

Images: a request with an image part may be up to 16 MiB (others keep 4 MiB); an image may decode to at most 64
megapixels; it is resized as the model's processor does (at most about 2 megapixels reach the model), and a full-size
image takes 2,048 of the request's 4,096 tokens. A bad image is refused before the engine sees the request, with its
own error: `invalid_image_base64`, `image_corrupt`, `image_truncated`, `image_unsupported_format` (only JPEG and PNG),
`image_unsupported_jpeg`, `image_unsupported_png`, `image_aspect_ratio` (over 200:1), `image_too_large` (413),
`too_many_images`. Without a vision encoder for the checkpoint, image parts get 501 `image_input_unsupported`.
Images are decoded two at a time across all requests; a request with images waits for its turn rather than being
refused 429 `frontend_busy` as a text request meeting a busy front end is (504 `frontend_deadline` if its deadline
passes while waiting; a request whose client has gone stops waiting, 499 `client_gone`).
An image sent again (the same bytes, the same base64 text) is not decoded or encoded again while it stays in the
server's image caches (`--image-cache-mb N`, default 32 MiB on the GTX 1080 for each of the front end's and the
engine's caches (the reference's 512: the engine's is device memory, [images-v1](../validation/images-v1/) rule 7),
least recently used first; `0` disables them). The replies are identical either way; only their speed differs, which
tells a client whether the image was sent recently by anyone (the Thor engine's SECURITY.md discusses this; it is not
part of this repository).

Reply: `text`, `audio_wav_base64` (24 kHz 16-bit mono WAV), `audio_seconds`, `frames`, `chunks`, `seed`, `cancelled`,
`truncated`, `prefill_seconds`, `generate_seconds`.

Streamed (`"stream": true`), the reply is a sequence of JSON lines:

| Event | Fields |
|---|---|
| `audio` | `chunk`, `pcm_base64` (24 kHz mono 16-bit LE); in order, exactly the unstreamed WAV's samples |
| `text` | `chunk`, `text` (the next piece of decoded text) |
| `chunk` | `chunk`, `frames`, `ended_by_eos` — a chunk has ended |
| `done` | the unstreamed reply's fields except the WAV |
| `error` | `error` code |

A client that disconnects is cancelled at the engine's next step.
