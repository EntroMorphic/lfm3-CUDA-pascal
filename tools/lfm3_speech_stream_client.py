#!/usr/bin/env python3
"""A minimal client for streamed POST /speech (validation/speech-stream-v1): reads the reply's events as they arrive,
prints the text as it comes, writes the audio to OUT.wav, and with --play pipes it to a player as it arrives
(default `aplay -q -f S16_LE -r 24000 -c 1`). Prints when the first audio arrived. Standard library only.
The network is read as fast as the server sends and the player is fed from its own thread, so a reply never waits on
playback (the server closes a stream whose reader stalls for its socket timeout). A stream that ends without a done
line is a failure.

  lfm3_speech_stream_client.py PORT OUT.wav (--text "Read this." | --wav QUESTION_16K.wav)
      [--mode audio|interleaved|text] [--voice female|male] [--seed N] [--max-new-tokens N] [--play [COMMAND]]
--text reads the text aloud (audio mode); --wav asks a spoken question, answered in interleaved text and speech.
"""
import argparse
import base64
import json
import os
import queue
import shlex
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3_token import read_token  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("port", type=int)
    a.add_argument("out")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--wav")
    a.add_argument("--mode")
    a.add_argument("--voice", default="female")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--max-new-tokens", type=int, default=1024)
    a.add_argument("--play", nargs="?", const="aplay -q -f S16_LE -r 24000 -c 1")
    a.add_argument("--token-file", help="the server's token file (default: LFM3_SERVER_TOKEN, else ~/.config/liquid-lfm3/token)")
    o = a.parse_args()
    token = read_token(o.token_file)  # LFM3_SERVER_TOKEN, else the server's default token file
    if o.text is not None:
        messages = [{"role": "system", "content": f"Perform TTS in American English using a {o.voice} voice."},
                    {"role": "user", "content": o.text}]
        mode = o.mode or "audio"
    else:
        audio = base64.b64encode(open(o.wav, "rb").read()).decode()
        messages = [{"role": "system", "content": "Respond with interleaved text and audio."},
                    {"role": "user", "content": [{"type": "audio", "audio": audio}]}]
        mode = o.mode or "interleaved"
    payload = json.dumps(dict(messages=messages, mode=mode, max_new_tokens=o.max_new_tokens, seed=o.seed,
                              stream=True)).encode()
    head = (f"POST /speech HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\nAuthorization: Bearer {token}\r\n\r\n").encode()
    player = subprocess.Popen(shlex.split(o.play), stdin=subprocess.PIPE) if o.play else None
    pieces = queue.Queue()
    player_error = []

    def feed():  # the player, at its own pace; a player that dies is a failure, reported at the end
        try:
            while (pcm := pieces.get()) is not None:
                player.stdin.write(pcm)
                player.stdin.flush()
        except Exception as e:  # noqa: BLE001
            player_error.append(e)
    feeder = threading.Thread(target=feed, daemon=True) if player else None
    if feeder:
        feeder.start()
    out = wave.open(o.out, "wb")
    out.setnchannels(1), out.setsampwidth(2), out.setframerate(24_000)
    s = socket.create_connection(("127.0.0.1", o.port), timeout=600)
    t0 = time.monotonic()
    s.sendall(head + payload)
    data, header, first, done = b"", None, None, None
    while True:
        block = s.recv(1 << 16)
        data += block
        if header is None:
            if b"\r\n\r\n" not in data and block:
                continue
            header, _, data = data.partition(b"\r\n\r\n")
            if b"x-ndjson" not in header:
                while block:
                    block = s.recv(1 << 16)
                    data += block
                status = header.split(b"\r\n")[0].decode()
                sys.exit(f"refused: {status} {data.decode()}")
        while b"\n" in data:
            line, _, data = data.partition(b"\n")
            e = json.loads(line)
            if e["event"] == "audio":
                pcm = base64.b64decode(e["pcm_base64"])
                if first is None:
                    first = time.monotonic() - t0
                    print(f"[first audio after {first:.2f} s]", file=sys.stderr, flush=True)
                out.writeframes(pcm)
                if player:
                    pieces.put(pcm)
            elif e["event"] == "text":
                print(e["text"], end="", flush=True)
            elif e["event"] in ("done", "error"):
                done = e
        if not block:
            break
    s.close()
    out.close()
    if player:
        pieces.put(None)
        feeder.join()
        try:
            player.stdin.close()
        except OSError as e:
            player_error.append(e)
        if player.wait() != 0 or player_error:
            sys.exit(f"the player failed (exit {player.returncode}): {player_error[:1]}")
    print(file=sys.stderr)
    if not done or done["event"] == "error":
        sys.exit(f"stream ended without a reply: {done}")
    print(f"[{done.get('audio_seconds', 0):.2f} s of audio in {time.monotonic() - t0:.2f} s; wrote {o.out}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
