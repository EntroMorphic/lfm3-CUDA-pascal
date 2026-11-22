"""Refuse a commit or a push that would add private data to the repository's history.

Once a commit is pushed, deleting a file does not remove it: the old commit keeps it. So this checks each commit
before it is made (pre-commit, commit-msg) and each commit before it leaves this machine (pre-push), not the tree.
It looks at the lines a change adds, the paths it touches and its commit message, for:
  - the patterns in tools/pascal/private-patterns (git ignores it, so the list itself is never committed): one
    Python regular expression a line, `#` for comments;
  - the values in tools/pascal/target.env (the target machine's addresses; not a value target.env.example
    also gives, such as the container's name);
  - the SHA-256 of every listening-set clip under outputs/ (a clip's hash in the history tells a listener which
    side is which). A hit names the file, not the clip.

    git config core.hooksPath tools/git-hooks      # install (once per clone)

    python3 tools/git-hooks/private_scan.py staged            # what pre-commit runs
    python3 tools/git-hooks/private_scan.py message FILE      # what commit-msg runs
    python3 tools/git-hooks/private_scan.py commits REV...    # what pre-push runs (git rev-list arguments)
"""
import hashlib
import re
import subprocess
import sys
from pathlib import Path


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, errors="replace").stdout


ROOT = Path(git("rev-parse", "--show-toplevel").strip())


def load_checks():
    checks = []  # (compiled pattern, label shown on a hit, whether the matched text may be shown)
    patterns = ROOT / "tools/pascal/private-patterns"
    if patterns.exists():
        for line in patterns.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                checks.append((re.compile(line.strip()), "a private pattern", True))
    else:
        print("private_scan: tools/pascal/private-patterns is missing; checking the other sources only", file=sys.stderr)
    def env_values(path):
        for line in path.read_text().splitlines() if path.exists() else []:
            key, _, value = line.partition("=")
            value = value.split("#")[0].strip().strip("'\"")
            if key.strip() and not key.lstrip().startswith("#") and len(value) >= 4 and not value.isdigit():
                yield key.strip(), value
    # a value the committed example also gives (the container's name, the workspace) is public by design
    published = {v for _, v in env_values(ROOT / "tools/pascal/target.env.example")}
    for key, value in env_values(ROOT / "tools/pascal/target.env"):
        if value not in published:
            checks.append((re.compile(re.escape(value)), f"target.env's {key}", True))
    for wav in sorted((ROOT / "outputs").glob("listening-*/**/*.wav")):
        digest = hashlib.sha256(wav.read_bytes()).hexdigest()
        checks.append((re.compile(digest, re.IGNORECASE), "a listening-set clip's hash", False))
    return checks


def scan(lines, checks):
    """lines: (where, text). Returns the hits as printable strings."""
    hits = []
    for where, text in lines:
        for pattern, label, show in checks:
            m = pattern.search(text)
            if m:
                hits.append(f"  {where}: {label}" + (f" ({m.group(0)!r})" if show else ""))
    return hits


def diff_lines(diff, prefix=""):
    """The added lines and touched paths of `git diff`/`git log -p` output, each with where it is."""
    path, commit = "?", prefix
    for line in diff.splitlines():
        if line.startswith("\x00COMMIT "):
            commit = line.split()[1][:12] + " "
        elif line.startswith("+++ ") or line.startswith("--- "):
            name = line[4:]
            if name != "/dev/null":
                path = name[2:] if name[:2] in ("a/", "b/") else name
                yield f"{commit}{path} (path)", path
        elif line.startswith("rename to ") or line.startswith("rename from "):
            yield f"{commit}{line} (path)", line
        elif line.startswith("+") and not line.startswith("+++"):
            yield f"{commit}{path}", line[1:]


def main(argv):
    if not argv or argv[0] not in ("staged", "message", "commits"):
        sys.exit(__doc__)
    checks = load_checks()
    diff_args = ["-U0", "--no-color", "--no-ext-diff", "--text", "-M"]
    if argv[0] == "staged":
        lines = list(diff_lines(git("diff", "--cached", *diff_args)))
    elif argv[0] == "message":
        lines = [("commit message", l) for l in Path(argv[1]).read_text(errors="replace").splitlines()
                 if not l.startswith("#")]
    else:
        revs = argv[1:]
        lines = list(diff_lines(git("log", "-p", *diff_args, "--format=%x00COMMIT %H", *revs)))
        for block in git("log", "--format=%H%x00%B%x01", *revs).split("\x01"):
            if "\x00" in block:
                sha, body = block.strip("\n").split("\x00", 1)
                lines += [(f"{sha[:12]} commit message", l) for l in body.splitlines()]
    hits = scan(lines, checks)
    if hits:
        print(f"private_scan ({argv[0]}): refused, {len(hits)} hit(s):", file=sys.stderr)
        print("\n".join(hits[:200]) + ("\n  ..." if len(hits) > 200 else ""), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
