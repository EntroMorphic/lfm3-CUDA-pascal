#!/usr/bin/env python3
"""Break Thor's "host memory in use" (MemTotal - MemAvailable, the soak's figure) into its parts, every PERIOD s.

Run on the Thor host (it sees the container's processes too) beside a soak. Per sample: the whole of /proc/meminfo,
the anonymous RSS of the server, of the soak client and of every other process together (RssAnon + RssShmem from
/proc/PID/status), the server's RssFile, and a residual: in use minus every accounted part (process anon and shmem,
the kernel's unreclaimable slab, page tables, kernel stacks, per-cpu, vmalloc, and the file cache MemAvailable does
not count). On Thor the GPU allocates from system memory and appears in no field, so the residual is mostly GPU and
driver memory. Exits when STOP_FILE exists. Writes one JSON object per line to OUT.
Usage: host_memory_sampler.py OUT STOP_FILE [PERIOD]
"""
import json
import os
import sys
import time
from pathlib import Path

KIB = 1024


def meminfo():
    return {l.split(":")[0]: int(l.split()[1]) * KIB for l in Path("/proc/meminfo").read_text().splitlines()}


def status(pid):
    try:
        text = Path(f"/proc/{pid}/status").read_text()
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    f = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in ("RssAnon", "RssFile", "RssShmem"):
            f[k] = int(v.split()[0]) * KIB
    return cmd, f


def sample(t0):
    m = meminfo()
    server = client = other = 0
    server_file = 0
    n_server = n_client = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        s = status(d)
        if not s or not s[1]:
            continue
        cmd, f = s
        anon = f.get("RssAnon", 0) + f.get("RssShmem", 0)
        if "liquid-lfm3-server" in cmd.split(" ")[0]:
            server += anon
            server_file += f.get("RssFile", 0)
            n_server += 1
        elif "check_lfm3_audio_soak" in cmd:
            client += anon
            n_client += 1
        else:
            other += anon
    used = m["MemTotal"] - m["MemAvailable"]
    # what MemAvailable leaves out of the file cache and reclaimable slab is in "used" too; estimate it as the
    # difference between (MemFree + file cache + reclaimable slab) and MemAvailable
    unavailable_cache = m["MemFree"] + m["Active(file)"] + m["Inactive(file)"] + m["SReclaimable"] - m["MemAvailable"]
    kernel = m["SUnreclaim"] + m["PageTables"] + m["KernelStack"] + m["Percpu"] + m["VmallocUsed"]
    residual = used - server - client - other - kernel - unavailable_cache
    return dict(t=round(time.monotonic() - t0, 1), wall=time.strftime("%H:%M:%S"), used=used, server_anon=server,
                server_file=server_file, client_anon=client, other_anon=other, kernel=kernel,
                unavailable_cache=unavailable_cache, residual=residual, n_server=n_server, n_client=n_client,
                meminfo=m)


def main(out, stop_file, period=30):
    period = float(period)
    t0 = time.monotonic()
    with open(out, "w") as f:
        while not Path(stop_file).exists():
            f.write(json.dumps(sample(t0)) + "\n")
            f.flush()
            time.sleep(period)


if __name__ == "__main__":
    main(*sys.argv[1:])
