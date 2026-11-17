#!/usr/bin/env bash
# Thor's power state, for the record of every speed measurement (no sudo needed). Writes one JSON object (valid JSON
# always: built by Python; a value that cannot be read is null, never 0). Exits 1 if the power mode is not MAXN (the
# pinned mode for measurements since 2026-11-09). "clocks_locked" is true only when the GPU's minimum and maximum
# frequencies were both read and are equal (jetson_clocks); it is recorded, not required.
# Usage: power_state.sh OUT.json [START.json]   with START.json: also exit 1 if anything but time and the current
#        clock (the mode, governors, frequency ranges, clocks_locked) differs from that earlier record. Deliberately
#        strict: a range lowered by thermal throttling during the round also fails it (its speed is not comparable).
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: power_state.sh OUT.json [START.json]" >&2; exit 2; }
mode=$(nvpmodel -q 2>/dev/null | sed -n 's/^NV Power Mode: //p' | head -1)
python3 - "$mode" "$@" <<'EOF'
import json, sys, time
from pathlib import Path
mode, out, start = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
def read(path, num=True):
    try:
        v = Path(path).read_text().strip()
        return int(v) if num else v
    except (OSError, ValueError):
        return None
g, c = "/sys/class/devfreq/gpu-gpc-0", "/sys/devices/system/cpu/cpu0/cpufreq"
s = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), mode=mode or None,
         gpu_governor=read(f"{g}/governor", False), gpu_min_hz=read(f"{g}/min_freq"), gpu_max_hz=read(f"{g}/max_freq"),
         gpu_cur_hz=read(f"{g}/cur_freq"), cpu_governor=read(f"{c}/scaling_governor", False),
         cpu_min_khz=read(f"{c}/scaling_min_freq"), cpu_max_khz=read(f"{c}/scaling_max_freq"))
s["clocks_locked"] = s["gpu_min_hz"] is not None and s["gpu_min_hz"] == s["gpu_max_hz"]
Path(out).write_text(json.dumps(s) + "\n")
print(json.dumps(s))
if s["mode"] != "MAXN":
    sys.exit(f"power mode is {s['mode']!r}, not MAXN: not measuring speed")
if start:
    before = json.loads(Path(start).read_text())
    changed = {k: (before.get(k), v) for k, v in s.items() if k not in ("time", "gpu_cur_hz") and before.get(k) != v}
    if changed:
        sys.exit(f"power state changed during the round: {changed}")
EOF
