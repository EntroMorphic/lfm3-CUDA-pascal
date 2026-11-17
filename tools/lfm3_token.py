"""The bearer token for liquid-lfm3-server, for clients: LFM3_SERVER_TOKEN if set, else the server's default token file,
$XDG_CONFIG_HOME/liquid-lfm3/token or ~/.config/liquid-lfm3/token (lfm3_http.hpp default_token_path; the server
creates it on its first start). The file must be an owned regular mode-0600 file, as the server requires."""
import os
import stat
from pathlib import Path


def default_token_path():
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg.startswith("/") else Path.home() / ".config"
    return base / "liquid-lfm3" / "token"


def read_token(path=None):
    env = os.environ.get("LFM3_SERVER_TOKEN")
    if env and path is None:
        return env
    p = Path(path) if path else default_token_path()
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
            raise PermissionError(f"{p} must be an owned regular mode-0600 file")
        return os.read(fd, 258).decode("ascii").rstrip("\n")
    finally:
        os.close(fd)
