#!/usr/bin/env bash
# validation/images-v1 rule 7: compute-sanitizer (memcheck with leak check, initcheck, synccheck, racecheck) on
# lfm3-vision-check: a small image (held-out mm01), two images in one prompt (stage-0 "two"), and a maximum-size image
# (2,097,152 pixels: a 2048x1024 noise image through HF's processor, 8,192 patches). Every run's stage outputs must be
# byte-identical to a plain run's.
# Usage (Thor host): [VISION_TAG=build-tag] vision_sanitizers.sh [TOOLS...]   (default tag vision-v1: images-v1's build;
# outputs in build/TAG/san; the maximum-size case is made once, in build/vision-v1/san/cases/max)
set -uo pipefail
W=${LFM3_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; E=/workspace/liquid-CUDA/build
TAG=${VISION_TAG:-vision-v1}
B=$W/liquid-CUDA/build/$TAG/san; C=$E/$TAG/san; CASES=$W/liquid-CUDA/build/vision-v1/san/cases; CC=$E/vision-v1/san/cases
BIN=$E/$TAG/lfm3-vision-check; M=/workspace/models/LFM3-Omni-2B
SAN=$E/sanitizer-13.5.25/cuda_sanitizer_api-linux-sbsa-13.5.25-archive/compute-sanitizer/compute-sanitizer
EXEC="docker exec -e LD_LIBRARY_PATH=/pinned-cuda:/usr/lib/aarch64-linux-gnu/nvidia lfm3-profile-v1"
TOOLS=${*:-memcheck initcheck synccheck racecheck}
mkdir -p $B
if [ ! -f $CASES/max/pixel_values.npy ]; then
  $EXEC bash -c "mkdir -p $CC/max && cd /workspace && /workspace/venvs/hf-5.20.2/bin/python -I -c \"
import numpy as np
from PIL import Image
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained('models/LFM3-Omni-2B', trust_remote_code=True).image_processor
img = Image.fromarray(np.random.default_rng(7).integers(0, 256, (1024, 2048, 3), dtype=np.uint8), 'RGB')
o = p(images=[img], return_tensors='np')
np.save('$CC/max/pixel_values.npy', o['pixel_values'].astype(np.float32))
np.save('$CC/max/image_grid_thw.npy', o['image_grid_thw'].astype(np.int64))
print('max case', o['pixel_values'].shape, o['image_grid_thw'].tolist())
\" 2>&1 | grep -v -i warn"
fi
declare -A CASE=([small]=$E/vision-ref/heldout/float32/mm01 [two]=$E/vision-ref/hf-5.20.2/float32/two [max]=$CC/max)
sha256sum $W/liquid-CUDA/build/$TAG/lfm3-vision-check > $B/binary.sha256
for c in small two max; do
  [ -f $B/plain-$c/stages.json ] || { $EXEC $BIN $M ${CASE[$c]} $C/plain-$c > $B/plain-$c.log 2>&1; echo "plain $c: exit $?"; }
done
for t in $TOOLS; do for c in small two max; do
  pre="$SAN --tool $t --error-exitcode 99 --print-limit 50"; [ $t = memcheck ] && pre="$pre --leak-check full"
  t0=$(date +%s); $EXEC $pre $BIN $M ${CASE[$c]} $C/$t-$c > $B/$t-$c.log 2>&1; s=$?
  same=identical; for f in $B/plain-$c/*.f32; do cmp -s $f $B/$t-$c/$(basename $f) || same=DIFFER; done
  echo "$t $c: exit $s, $(( $(date +%s) - t0 )) s; $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|LEAK SUMMARY' $B/$t-$c.log | tr '\n' ' '); outputs $same"
done; done
$EXEC chown -R 1000:1000 $C
echo VISION SANITIZERS DONE
