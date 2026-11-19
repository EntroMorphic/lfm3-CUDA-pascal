"""Compares the device code of two binaries function by function (cuobjdump -sass output, with each instruction's
address comment removed): the same set of functions with the same instructions, whatever their order in the file.
    python3 validation/speed-v1/sass_compare.py A.sass B.sass [--only-in-a REGEX]
Exit 0 when every function in both has the same code and no function is in only one, except that functions of A
whose (mangled) names match REGEX may be missing from B (a launch removed on purpose). A file with no functions fails.
"""
import re, sys
def functions(path):
    out, name, body = {}, None, []
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if name: out.setdefault(name, []).append("".join(body))
            name, body = m.group(1), []
        elif name: body.append(re.sub(r"/\*[0-9a-f]{4}\*/", "", line))
    if name: out.setdefault(name, []).append("".join(body))
    return {k: sorted(v) for k, v in out.items()}
args = sys.argv[1:]
allowed = None
if "--only-in-a" in args:
    i = args.index("--only-in-a"); allowed = re.compile(args[i + 1]); del args[i:i + 2]
if len(args) != 2: sys.exit(__doc__)
a, b = functions(args[0]), functions(args[1])
if not a or not b: print(f"{len(a)} and {len(b)} functions: no device code read"); sys.exit(1)
only_a, only_b = set(a) - set(b), set(b) - set(a)
removed = {k for k in only_a if allowed and allowed.search(k)}
unexpected = (only_a - removed) | only_b
differ = [k for k in set(a) & set(b) if a[k] != b[k]]
print(f"{len(a)} and {len(b)} functions; removed as expected: {len(removed)}; only in one otherwise: {len(unexpected)}; "
      f"same name, different code: {len(differ)}")
for k in sorted(removed): print("   removed:", k[:160])
for k in sorted(unexpected)[:5] + sorted(differ)[:5]: print("  ", k[:160])
sys.exit(0 if not unexpected and not differ else 1)
