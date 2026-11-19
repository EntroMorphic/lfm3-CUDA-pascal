"""Compares the device code of two binaries function by function (cuobjdump -sass output, with each instruction's
address comment removed): the same set of functions with the same instructions, whatever their order in the file.
    python3 validation/speed-v1/sass_compare.py A.sass B.sass
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
a, b = functions(sys.argv[1]), functions(sys.argv[2])
only = set(a) ^ set(b); differ = [k for k in set(a) & set(b) if a[k] != b[k]]
print(f"{len(a)} and {len(b)} functions; only in one: {len(only)}; same name, different code: {len(differ)}")
for k in list(only)[:5] + differ[:5]: print("  ", k[:120])
sys.exit(0 if not only and not differ else 1)
