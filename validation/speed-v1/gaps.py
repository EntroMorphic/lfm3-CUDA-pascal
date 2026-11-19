import sqlite3, statistics, bisect, sys
db = sqlite3.connect(sys.argv[1])
cp = [r for r in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where bytes = 512000 order by start")]
ks = list(db.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
starts = [k[0] for k in ks]
gaps_copy, gaps_after, inner = [], [], []
for i in range(100, len(cp) - 1):
    s, e = cp[i]
    j = bisect.bisect_left(starts, e); prev = bisect.bisect_left(starts, s) - 1
    gaps_copy.append((s - ks[prev][1]) / 1e3); gaps_after.append((ks[j][0] - e) / 1e3); inner.append((e - s) / 1e3)
a, b = bisect.bisect_left(starts, cp[200][1]), bisect.bisect_left(starts, cp[201][0])
k_in = [(ks[t + 1][0] - ks[t][1]) / 1e3 for t in range(a, b - 1)]
print("last kernel to copy start %.1f us; copy %.1f us; copy end to next kernel %.1f us" % (statistics.median(gaps_copy), statistics.median(inner), statistics.median(gaps_after)))
print("gaps between one step's kernels: %d gaps, total %.1f us, median %.2f us" % (len(k_in), sum(k_in), statistics.median(k_in)))
