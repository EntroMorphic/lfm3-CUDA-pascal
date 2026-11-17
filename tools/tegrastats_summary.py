#!/usr/bin/env python3
"""Summarize a tegrastats log for a speed round's record: per rail the median and maximum power (mW), per sensor the
maximum temperature (C), the CPU cores' median and maximum load (%) and the frequencies seen; run as root, also the
memory clock (EMC: load % and MHz, every value seen) and the GPU's clocks (GR3D, MHz per GPC). One JSON object on stdout.
Usage: tegrastats_summary.py tegrastats.log"""
import json, re, statistics, sys

rails, temps, loads, freqs, n = {}, {}, [], set(), 0
emc_load, emc_mhz, gr3d = [], set(), set()
for line in open(sys.argv[1], errors="replace"):
    if "RAM" not in line:
        continue
    n += 1
    for name, now in re.findall(r"(VDD_\w+|VIN\w*) (\d+)mW/\d+mW", line):
        rails.setdefault(name, []).append(int(now))
    for name, t in re.findall(r"(\w+)@(-?[\d.]+)C", line):
        temps.setdefault(name, []).append(float(t))
    e = re.search(r"EMC_FREQ (\d+)%@(\d+)", line)
    if e:
        emc_load.append(int(e.group(1)))
        emc_mhz.add(int(e.group(2)))
    g = re.search(r"GR3D_FREQ (?:\d+%)?@\[([^\]]*)\]", line)
    if g:
        gr3d.update(int(x) for x in g.group(1).split(",") if x.strip().isdigit())
    m = re.search(r"CPU \[([^\]]*)\]", line)
    if m:
        cores = [c for c in m.group(1).split(",") if "@" in c]
        load = [int(c.split("%")[0]) for c in cores]
        if load:
            loads.append(statistics.mean(load))
        freqs.update(int(c.split("@")[1]) for c in cores)
print(json.dumps(dict(
    samples=n,
    rails_mw={k: dict(median=statistics.median(v), max=max(v)) for k, v in rails.items()},
    temps_max_c={k: max(v) for k, v in temps.items()},
    cpu_load_pct=dict(median=statistics.median(loads), max=max(loads)) if loads else None,
    cpu_mhz_seen=sorted(freqs),
    emc=dict(load_pct_median=statistics.median(emc_load), load_pct_max=max(emc_load), mhz_seen=sorted(emc_mhz))
    if emc_load else None,
    gr3d_mhz_seen=sorted(gr3d) or None), indent=1))
