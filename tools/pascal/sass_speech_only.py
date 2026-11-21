"""speech-v1 rule 5: the device code of two lfm3-speech binaries (cuobjdump -sass) is the same function for function,
except functions of the language model's GEMM (lfm::gemm, mangled "3lfm4gemm"), which may differ or be in one only.
Functions are compared as validation/speed-v1/sass_compare.py compares them (address comments removed).
    python3 tools/pascal/sass_speech_only.py SHIPPING.sass PARITY.sass
Exit 0 when every function outside lfm::gemm is in both with the same code.
"""
import importlib.util
import sys
from pathlib import Path

if len(sys.argv) != 3: sys.exit(__doc__)
spec = importlib.util.spec_from_file_location("sc", Path(__file__).resolve().parents[2] / "validation/speed-v1/sass_compare.py")
src = Path(spec.origin).read_text()
ns = {}
exec(src.split("args = sys.argv[1:]")[0], ns)  # the functions() reader alone
a, b = ns["functions"](sys.argv[1]), ns["functions"](sys.argv[2])
gemm = lambda k: "3lfm4gemm" in k
if not a or not b: print("no device code read"); sys.exit(1)
only = sorted((set(a) ^ set(b)))
differ = sorted(k for k in set(a) & set(b) if a[k] != b[k])
outside = [k for k in only + differ if not gemm(k)]
same = sum(1 for k in set(a) & set(b) if a[k] == b[k] and not gemm(k))
print(f"{len(a)} and {len(b)} functions; outside lfm::gemm: {same} identical, {len(outside)} not; "
      f"lfm::gemm: {sum(gemm(k) for k in only)} in one only, {sum(gemm(k) for k in differ)} differing")
for k in outside: print("NOT IDENTICAL", k)
sys.exit(0 if not outside else 1)
