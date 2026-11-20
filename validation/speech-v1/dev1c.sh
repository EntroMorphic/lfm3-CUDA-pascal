# speech-v1 development 1c: dev1.sh's lighthouse runs again, from the split requests (dev1b had run the unsplit request
# of the same name, tts-long-male-sN.req.json, which also exists among the single requests). Same binaries.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev1; M=/workspace/models/LFM3-Omni-2B
for build in parity-s1 text-s1; do
  mkdir -p $O/$build-split
  for s in 1 2 3; do n=tts-long-male-s$s
    /usr/bin/time -f "%e s %M KiB" $B/$build/lfm3-speech $M $O/split/requests/$n.req.json $O/$build-split/$n > $O/$build-split/$n.log 2> $O/$build-split/$n.err; echo "$build $n rc=$? $(tail -1 $O/$build-split/$n.err)"
  done
done
(cd $O && sha256sum *-split/*.wav) > $O/wav-split.sha256; cat $O/wav-split.sha256
date
