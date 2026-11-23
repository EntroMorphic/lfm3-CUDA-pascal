# images-v1, part 2 (tools/pascal/remote.sh detach img-v1b 'bash validation/images-v1/job2.sh'), after job.sh:
# - rule 3's comparison again on job.sh's outputs. job.sh ran tools/compare_lfm3_vision_stages.py with the container's
#   python3, which has no numpy, so it compared nothing; the encoder's runs (lfm3-vision-check, every stage written)
#   had completed. Here the same comparison and judgment with the venv's Python (no GPU).
# - rule 6's server check again, with --handlers 16 (below).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/img-v1; PY=$B/venv-asr/bin/python
REF=/workspace/data/vision-ref/hf-5.20.2; I=validation/images-v1/inputs/reference
CASES="cherry china flower gray mirror noise rgba small tall two wide"
for c in $CASES; do
  $PY tools/compare_lfm3_vision_stages.py $O/vision/$c $REF/float32/$c $REF/bfloat16/$c --json $O/vision/$c.compare.json > $O/vision/$c.compare.log 2>&1; echo "compare $c rc=$?"
done
python3 - $O/vision $I/thor-fp32-compare $CASES <<'PY2'
import gzip, json, sys
from pathlib import Path
o, thor, cases = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]; ok = True
for c in cases:
    try: j = json.loads((o / f"{c}.compare.json").read_text())["stages"]["image_features"]
    except Exception as e: print(f"rule 3 {c}: NO RESULT ({e!r})"); ok = False; continue
    t = json.loads(gzip.open(thor / f"{c}.compare.json.gz", "rt").read())["stages"]["image_features"]
    good = j["ratio"] <= 1.0 and j["native_min_cos"] >= j["bf16_min_cos"]; ok &= good
    print(f"rule 3 {c}: error {j['native_rel']:.3e} (HF BF16 {j['bf16_rel']:.3e}, ratio {j['ratio']:.4f}; Thor kFp32 {t['ratio']:.4f}), "
          f"min cosine {j['native_min_cos']:.7f} (HF BF16 {j['bf16_min_cos']:.7f}; Thor {t['native_min_cos']:.7f}): {'ok' if good else 'FAIL'}")
print("RULE 3", "PASS" if ok and len(cases) == 11 else "FAIL")
PY2
date

# rule 6 again: job.sh's server check sent the 11 requests at once to a server taking 8 connections (--handlers 8, its
# default; it closes the rest at once), so three were cut off and the check stopped on them. The same check with
# --handlers 16; its sequential part (each case unstreamed and streamed) had passed in job.sh.
M=/workspace/models/LFM3-Omni-2B
$PY tools/check_lfm3_image_server.py $B/text-v40/liquid-lfm3-server $M $O/e2e $O/image-server-h16.json --handlers 16 > $O/image-server-h16.log 2>&1
echo "rule 6 server (--handlers 16) rc=$? $(grep -E '^concurrent' $O/image-server-h16.log | cut -c1-300) $(tail -1 $O/image-server-h16.log)"
pkill -TERM -f liquid-lfm3-server; date

# rule 7 again: job.sh's image configuration sent the clip before the images, when the vision encoder's scratch did not
# yet exist; it is kept once made (development 1: about 452 MiB with the cache, resident after the first full-size
# image), so the worst case is the clip after the images. The configuration now sends the clip, the five images, then
# the clip again, and every configuration keeps its 0.2 s memory series.
LIBRI=/workspace/data/librispeech/wav
IMAGES=1 $PY tools/pascal/speech_memory.py $B/text-v40/liquid-lfm3-server $M 1 $LIBRI $O/memory-1b 7740 29696 > $O/memory-1b.log 2>&1
echo "rule 7 (clip after the images) rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-1b.log
pkill -TERM -f liquid-lfm3-server; date
