import json, sys
from pathlib import Path
T = {"encode_seconds", "prefill_seconds", "generate_seconds", "lm_step_seconds", "text_pick_seconds", "depth_seconds",
     "sample_host_seconds", "codec_seconds", "real_time_factor", "real_time_factor_end_to_end"}
a, b, stems = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
same = 0
for s in stems:
    fa, fb = sorted(p.name for p in a.glob(s + ".*")), sorted(p.name for p in b.glob(s + ".*"))
    ok = bool(fa) and fa == fb
    for f in fa if ok else []:
        if f.endswith(".json"):
            ja, jb = [{k: v for k, v in json.loads((d / f).read_text()).items() if k not in T} for d in (a, b)]; ok &= ja == jb
        else: ok &= (a / f).read_bytes() == (b / f).read_bytes()
    same += ok
    if not ok: print(f"DIFFERS {s} ({len(fa)} and {len(fb)} files)")
print(f"{same} of {len(stems)} identical ({a} against {b})"); sys.exit(0 if same == len(stems) and stems else 1)
