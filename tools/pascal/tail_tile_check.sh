#!/usr/bin/env bash
# Runs in the container: a position's score must not depend on where the scorer's tiles of 256 rows end.
#   tools/pascal/tail_tile_check.sh TAG OUT_NAME
# Scores the same 8 token sequences cut to 260 and to 520 tokens. In the 260-token windows rows 256-258 are a last
# tile of 3 rows; in the 520-token windows they sit inside a tile of 256. The first 259 NLL values of every window
# must be byte-identical between the two (a row of a prefill does not depend on later rows). Exit 0 only then.
set -uo pipefail
tag=${1:?usage: tail_tile_check.sh TAG OUT_NAME}; name=${2:?}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); out=/workspace/build/$name
test ! -e "$out" || { echo "exists: $out"; exit 2; }
mkdir -p "$out/windows"
python3 - /workspace/data/ppl-windows/windows-4096.i32 "$out/windows" <<'PY' || exit 1
import sys
from array import array
a = array("i"); a.frombytes(open(sys.argv[1], "rb").read())
for length in (260, 520):
    w = array("i")
    for i in range(8): w.extend(a[i * 4096:i * 4096 + length])
    open(f"{sys.argv[2]}/windows-{length}.i32", "wb").write(w.tobytes())
PY
WINDOWS_DIR=$out/windows "$repo/tools/pascal/score.sh" "$tag" "$name" 260 520 || { echo "TAIL TILE: scoring FAILED"; exit 1; }
python3 - "$out" <<'PY'
import sys
from array import array
def load(length):
    x = array("f"); x.frombytes(open(f"{sys.argv[1]}/nll-pascal-{length}.f32", "rb").read())
    assert len(x) == 8 * (length - 1), f"{length}: {len(x)} values"
    return [x[i * (length - 1):(i + 1) * (length - 1)] for i in range(8)]
short, long_ = load(260), load(520)
head = sum(s[c] != l[c] for s, l in zip(short, long_) for c in range(256))
tail = sum(s[c] != l[c] for s, l in zip(short, long_) for c in range(256, 259))
print(f"8 windows: rows 0-255 differ in {head} of {8 * 256} values; rows 256-258 (the 260-token windows' last tile) in {tail} of 24")
print("TAIL TILE", "PASS" if head == 0 and tail == 0 else "FAIL"); sys.exit(0 if head == 0 and tail == 0 else 1)
PY
