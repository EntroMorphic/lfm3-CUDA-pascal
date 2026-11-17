"""Read the live token count the sampling oracle used (it counts with the engine's own C++
lfm::tokenizer_token_count, so gates never re-implement the tokenizer rules in Python)."""
import re
from pathlib import Path


def oracle_live_count(log_path):
    text = Path(log_path).read_text()
    found = re.findall(r"live_token_count (\d+)\s*$", text, flags=re.M)
    if len(found) != 1:
        raise AssertionError(f"oracle log {log_path} does not report exactly one live_token_count")
    return int(found[0])
