"""Runs in the container: what a decode step costs, from an Nsight Systems profile exported to SQLite
(nsys profile -t cuda ... ; nsys export -t sqlite). Steps are delimited by the 512,000-byte device-to-host copy
of the logits, or where there is none (the greedy choice on the card) by the greedy_finish kernel. Prints the cold-start timeline, then the steady-state step broken down by kernel and, for the
projection kernel, by weight shape with the bytes of weights each call reads.
    python3 tools/pascal/decode_atoms.py PROFILE.sqlite [STEADY_FROM_STEP=300]
"""
import sqlite3, sys, collections, statistics

db = sqlite3.connect(sys.argv[1]); steady_from = int(sys.argv[2]) if len(sys.argv) > 2 else 300
names = dict(db.execute("select id, value from StringIds"))
logits = [r for r in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes = 512000 order by start")]
if not logits:  # the greedy choice on the card (speed-v1): a step ends with greedy_finish, no logits copy
    logits = [r for r in db.execute("select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName = s.id "
                                    "where s.value = 'greedy_finish' order by k.start")]
kernels = [(s, e, names[n], gx, gy, bx) for s, e, n, gx, gy, bx in db.execute(
    "select start, end, shortName, gridX, gridY, blockX from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
api = [(s, e, names[n]) for s, e, n in db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start")]
t0 = kernels[0][0]
# step i (i >= 1) = (end of logits copy i-1, end of logits copy i]; step 0 is the prefill
bounds = [t0] + [e for _, e in logits]
def step_of(t, lo=[0]):
    while lo[0] + 1 < len(bounds) and t >= bounds[lo[0] + 1]: lo[0] += 1
    return lo[0]
per_step_kernel = collections.defaultdict(float); per_step_count = collections.Counter()
by_kind = collections.defaultdict(list)  # steady state: (name, shape) -> durations
wall = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
if not 1 <= steady_from < len(wall): sys.exit(f"STEADY_FROM_STEP must be 1..{len(wall) - 1}: the profile has {len(wall)} forward calls")
for s, e, name, gx, gy, bx in kernels:
    i = step_of(s)
    if i >= len(wall): continue  # after the last logits copy: teardown, not a step
    per_step_kernel[i] += e - s; per_step_count[i] += 1
    if i >= steady_from: by_kind[(name, gx, gy, bx)].append(e - s)
print(f"{len(logits)} forward calls; prefill wall {wall[0] / 1e6:.1f} ms, {per_step_count[0]} kernels, kernel time {per_step_kernel[0] / 1e6:.1f} ms")
print("\ncold start: decode steps after the prefill (wall ms, GPU kernel ms, the rest)")
for i in list(range(1, 13)) + [16, 24, 32, 48, 64, 96, 128, 200, 300, 500, len(wall) - 1]:
    if i < len(wall): print(f"  step {i:4d} at {(bounds[i] - t0) / 1e9:6.2f} s: wall {wall[i] / 1e6:6.2f}  kernels {per_step_kernel[i] / 1e6:6.2f}  other {(wall[i] - per_step_kernel[i]) / 1e6:5.2f}")
steady = range(steady_from, len(wall)); n = len(steady)
w = statistics.mean(wall[i] for i in steady) / 1e6; k = statistics.mean(per_step_kernel[i] for i in steady) / 1e6
print(f"\nsteady state, steps {steady_from}..{len(wall) - 1}: wall {w:.3f} ms per step (min {min(wall[i] for i in steady) / 1e6:.3f}, max {max(wall[i] for i in steady) / 1e6:.3f}); "
      f"kernels {k:.3f} ms in {per_step_count[steady_from]} launches; not in kernels {w - k:.3f} ms")
# thin_kernel shapes: grid (blocks, batch) with 256 threads; n = columns. warps = ceil(n / 4), blocks = ceil(warps / 8).
SHAPES = {  # blocks -> (label, n, k candidates)
    4000: "lm_head 128000 x 2048", 240: "mlp w1 / w3 7680 x 2048", 192: "conv in_proj 6144 x 2048",
    64: "2048 x K (conv out, q, o: K 2048; mlp w2: K 7680)", 16: "k / v proj 512 x 2048", 1: "gate 16 x 2048"}
print("\nsteady state by kernel (ms per step = mean duration x calls per step):")
rows = []
for (name, gx, gy, bx), d in by_kind.items():
    rows.append((sum(d) / n / 1e6, name, gx, gy, bx, len(d) / n, statistics.mean(d) / 1e3, min(d) / 1e3, max(d) / 1e3))
for ms, name, gx, gy, bx, calls, mean, lo, hi in sorted(rows, reverse=True):
    label = SHAPES.get(gx, "") if name.startswith("thin_kernel") else ""
    print(f"  {ms:7.3f} ms  {calls:6.1f} calls  mean {mean:8.1f} us (min {lo:7.1f}, max {hi:8.1f})  {name[:28]:28s} grid {gx}x{gy} block {bx}  {label}")
# thin 2048-column calls: split by duration into K = 2048 and K = 7680
d64 = sorted(by_kind.get(("thin_kernel", 64, 1, 256), []))
if d64:
    cut = (d64[0] + d64[-1]) / 2; small = [x for x in d64 if x < cut]; large = [x for x in d64 if x >= cut]
    print(f"\n  2048-column projections split by duration: {len(small) / n:.1f} calls per step at {statistics.mean(small) / 1e3:.1f} us, {len(large) / n:.1f} at {statistics.mean(large) / 1e3:.1f} us")
print("\nCUDA API calls in steady state (ms per step):")
acc = collections.defaultdict(float); cnt = collections.Counter()
lo = 0
for s, e, name in api:
    while lo + 1 < len(bounds) and s >= bounds[lo + 1]: lo += 1
    if steady_from <= lo < len(wall): acc[name] += e - s; cnt[name] += 1  # lo == len(wall): teardown
for name, t in sorted(acc.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {t / n / 1e6:7.3f} ms  {cnt[name] / n:6.1f} calls  {name}")
