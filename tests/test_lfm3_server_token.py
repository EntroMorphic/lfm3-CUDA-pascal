"""The server's bearer token without setup (lfm3_http.hpp startup_token; validation/audio-assurance-v1, Tripp: "make the
token easier"): with no --token-file and no LFM3_SERVER_TOKEN, the server reads ~/.config/liquid-lfm3/token
($XDG_CONFIG_HOME/liquid-lfm3/token), creating it on first start (mode 0600, in a mode-0700 directory) with a random
token; clients find it through tools/lfm3_token.py. Run on the harness (the real HTTP layer and token code) in every
build variant. --token-file and LFM3_SERVER_TOKEN keep their precedence; an unsafe file or directory is refused."""
import json
import os
import secrets
import stat
import subprocess
import sys

import pytest

from test_lfm3_http_native import CHECKPOINT, ROOT, harness_binary  # noqa: F401  (fixture)

sys.path.insert(0, str(ROOT / "tools"))
import lfm3_token  # noqa: E402


def start(binary, config, *options, token=None):
    env = {k: v for k, v in os.environ.items() if k != "LFM3_SERVER_TOKEN"}
    env.update(XDG_CONFIG_HOME=str(config), TSAN_OPTIONS="halt_on_error=1")
    if token:
        env["LFM3_SERVER_TOKEN"] = token
    prefix = ["setarch", os.uname().machine, "-R"] if binary.name.endswith("-tsan") else []
    p = subprocess.Popen([*prefix, str(binary), str(CHECKPOINT), *map(str, options)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    line = p.stdout.readline()
    if not line:
        p.wait(60)
        return p, None, p.stderr.read().decode()
    return p, json.loads(line)["port"], ""


def stop(p):
    p.terminate()
    p.wait(60)
    return p.stderr.read().decode()


def status(port, token):
    import http.client
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", "/sessions", body=json.dumps({"capacity": 512}),
              headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    return c.getresponse().status


def test_first_start_creates_the_token_and_later_starts_reuse_it(harness_binary, tmp_path, monkeypatch):
    path = tmp_path / "liquid-lfm3" / "token"
    p, port, err = start(harness_binary, tmp_path)
    assert port, err
    note = stop(p)
    assert f"created the bearer token file {path}" in note
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    token = path.read_text().rstrip("\n")
    assert len(token) >= 32 and token not in note  # the path is logged, never the token
    monkeypatch.delenv("LFM3_SERVER_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert lfm3_token.default_token_path() == path and lfm3_token.read_token() == token
    p, port, err = start(harness_binary, tmp_path)
    try:
        assert port, err
        assert status(port, token) == 200 and status(port, "x" * 43) == 401
    finally:
        note = stop(p)
    assert f"bearer token from {path}" in note and path.read_text().rstrip("\n") == token  # reused, unchanged


def test_explicit_tokens_take_precedence_and_leave_no_file(harness_binary, tmp_path):
    env_token = secrets.token_urlsafe(32)
    p, port, err = start(harness_binary, tmp_path, token=env_token)
    try:
        assert port, err
        assert status(port, env_token) == 200
    finally:
        stop(p)
    file_token = secrets.token_urlsafe(32)
    f = tmp_path / "mine"
    f.write_text(file_token + "\n")
    f.chmod(0o600)
    p, port, err = start(harness_binary, tmp_path, "--token-file", f, token=env_token)
    try:
        assert port, err
        assert status(port, file_token) == 200 and status(port, env_token) == 401
    finally:
        stop(p)
    assert not (tmp_path / "liquid-lfm3").exists()


@pytest.mark.parametrize("unsafe", ["file-mode", "symlink", "dir-writable", "dir-symlink"])
def test_an_unsafe_token_file_or_directory_is_refused(harness_binary, tmp_path, unsafe):
    d = tmp_path / "liquid-lfm3"
    real = tmp_path / "elsewhere"
    real.mkdir(mode=0o700)
    (real / "token").write_text(secrets.token_urlsafe(32) + "\n")
    (real / "token").chmod(0o600)
    if unsafe == "dir-symlink":
        d.symlink_to(real)
    else:
        d.mkdir(mode=0o700)
        if unsafe == "file-mode":
            (d / "token").write_text(secrets.token_urlsafe(32) + "\n")
            (d / "token").chmod(0o644)
        elif unsafe == "symlink":
            (d / "token").symlink_to(real / "token")
        else:
            d.chmod(0o777)
    p, port, err = start(harness_binary, tmp_path)
    if port:
        stop(p)
    assert port is None and p.returncode != 0, unsafe
