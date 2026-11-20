# serve-v1, after the red-team review (tools/pascal/remote.sh detach serve-v1b 'bash validation/serve-v1/job2.sh'):
# paired decode's validate() checks the compact slots before any change; the text-only switch accepts only 0 or 1;
# serve_memory.py judges each refusal's error, samples memory for real and reaches a full queue by simultaneous resets.
# Builds text-v34 and parity-v34. Rules 1 (device code, gate, fixtures), 3-9 on both builds, 10, 11 and 12 again.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; M=$B/sv2-serve; PY=$B/venv-ref/bin/python
[ -e $M ] && { echo "$M exists"; exit 2; }; mkdir -p $M
# The builds the earlier records name, by hash (text-v32 is speed-v1's last build; text-v33 and parity-v33 the rule run's).
for b in text-v32 text-v33 parity-v33; do (cd $B/$b && sha256sum liquid-lfm3 lfm3-score $( [ -e liquid-lfm3-server ] && echo liquid-lfm3-worker liquid-lfm3-server)) | sed "s#  #  $b/#"; done > $M/earlier-builds.sha256; cat $M/earlier-builds.sha256
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_serve.sh parity-v34; echo "build parity-v34 rc=$?"
python3 tools/pascal/compare_pins.py parity-v34; echo "rule 1 gate rc=$?"
tools/pascal/build_serve.sh text-v34; echo "build text-v34 rc=$?"
# Device code: the CLI and scorer unchanged from text-v33; the worker and server too (the fixes are host code).
for f in liquid-lfm3 lfm3-score liquid-lfm3-worker liquid-lfm3-server; do for v in 33 34; do cuobjdump -sass $B/text-v$v/$f > $M/$f.$v.sass; done
  python3 validation/speed-v1/sass_compare.py $M/$f.33.sass $M/$f.34.sass > $M/$f.sass-compare.txt; echo "$f device code against text-v33 rc=$? $(head -1 $M/$f.sass-compare.txt)"; done
sha256sum $M/*.sass > $M/sass.sha256; gzip -9 $M/*.sass
tools/pascal/run_text_fixtures.sh text-v34 2; echo "fixtures rc=$?"
# The switch: 1 builds; every other value but 0 (the Thor build) stops it.
flags="-O3 --fmad=false -std=c++20 -arch=sm_61 -Wno-deprecated-gpu-targets -DLFM3_NATIVE_GEMM -I native/include -I $B/ljt-3.2.0/ljt/include"
for v in 1 2 on 01 "" "1+0" "0*5" "(1)" "-1"; do
  nvcc $flags "-DLFM3_TEXT_ONLY=$v" -c -o $M/switch.o native/src/lfm3_worker.cu > $M/switch.log 2>&1; echo "switch LFM3_TEXT_ONLY='$v' build rc=$?"; done
rm -f $M/switch.o
tools/pascal/serve_gates.sh text-v34 sv2-serve-shipping own; echo "rules 3-9 shipping rc=$?"
tools/pascal/serve_gates.sh parity-v34 sv2-serve-parity pinned; echo "rules 3-9 parity rc=$?"
python3 tools/pascal/paired_decode_check.py $B/text-v34/liquid-lfm3-worker /workspace/models/LFM3-Omni-2B /workspace/data/ppl-windows/windows-32768.i32 $M/paired > $M/paired.log 2>&1; echo "rule 10 rc=$?"; grep -E 'lane|PAIRED' $M/paired.log
$PY tools/pascal/serve_memory.py $B/text-v34/liquid-lfm3-server /workspace/models/LFM3-Omni-2B $B/sv2-serve-shipping/control.output.json $M/memory > $M/memory.log 2>&1; echo "rules 11-12 rc=$?"; grep -E 'rule|SERVE|rror' $M/memory.log
date
