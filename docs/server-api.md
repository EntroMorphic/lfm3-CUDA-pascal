# Server API

`liquid-lfm3-server --experimental MODEL_DIR [--port N] [--token-file FILE] [--speech-rows N] [...]`

- Loopback only. Once listening it prints one line, `{"ready":true,"port":N}` (`--port 0` picks a free port).
- Every `POST` needs `Authorization: Bearer TOKEN`, `Content-Type: application/json` and one `Content-Length`. The
  token comes from `--token-file`, `LFM3_SERVER_TOKEN`, or `~/.config/liquid-lfm3/token` (created on first start);
  `tools/lfm3_token.py` reads the same place for clients.
- Bodies are at most 4 MiB, parsed strictly: duplicate keys, non-finite numbers, invalid UTF-8, nesting over 64 and
  unknown fields are refused with a 400 naming the problem.
- `SIGTERM` stops admission, cancels running generations at their next token, and exits 0.

The authoritative contract is the comment above each handler in `native/include/lfm/lfm3_http.hpp`.

## Text: sessions

| Endpoint | Body | Effect |
|---|---|---|
| `POST /sessions` | `{"capacity": 32768}` | Creates a session with its own KV cache; returns `session_id` |
| `POST /generate` | `{"session_id", "messages": [...], "max_new_tokens", "sampling": {...}}` | Generates on the session, reusing its cached prefix |
| `POST /generate_batch` | `{"requests": [A, B]}` | Two sessions decoded together on the GPU |
| `POST /sessions/ID/reset` | `{}` | Clears the session's cache |
| `POST /sessions/ID/close` | `{}` | Releases the session |
| `POST /sessions/ID/cancel` | `{}` | Cancels a running generation at its next token (202) |

`messages` follows the chat-completions shape (`system`, `user`, `assistant`, `tool` roles; string or text-part
content). Replies carry `text`, `generated_ids`, `finish_reason` (`stop` or `length`) and cache accounting.

## Speech: `POST /speech`

One request, no session; concurrent requests share the engine's speech batch (`--speech-rows`, default 8), each reply
byte-identical to the same request alone.

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
server's image caches (`--image-cache-mb N`, default 512 MiB for each of the front end's and the engine's caches,
least recently used first; `0` disables them). The replies are identical either way; only their speed differs, which
tells a client whether the image was sent recently by anyone (see SECURITY.md).

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
