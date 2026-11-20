# speech-v1 development 1 (tools/pascal/remote.sh detach speech-dev1 'bash validation/speech-v1/dev1.sh'): builds
# parity-s1 and text-s1 with lfm3-speech; renders the reference's requests (tools/lfm3_speech_hf.py and
# tools/lfm3_speech_chunked_hf.py, "requests": no model loaded); runs both builds on rule 3's clips (seed 1 of the nine
# single-request cases, the two split paragraphs at seeds 1 to 3). The WAVs are compared on the development machine.
# Corrected after its first run (speech-dev1b): the lighthouse runs had taken the unsplit request of the same name;
# dev1c.sh reruns them.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/sp-dev1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-s1; echo "build parity-s1 rc=$?"
tools/pascal/build_speech.sh text-s1; echo "build text-s1 rc=$?"
$PY tools/lfm3_speech_hf.py $M validation/speech-e2e-v1/prompts.json $O/single requests > $O/requests-single.log 2>&1; echo "requests single rc=$?"
$PY tools/lfm3_speech_chunked_hf.py $M validation/speech-e2e-v1/prompts-chunked.json $O/split requests > $O/requests-split.log 2>&1; echo "requests split rc=$?"
ls $O/single/requests $O/split/requests | wc -l
nvidia-smi --query-gpu=memory.used,clocks.mem,clocks.sm,temperature.gpu --format=csv,noheader
for build in parity-s1 text-s1; do
  mkdir -p $O/$build
  for f in $(ls $O/single/requests | grep -- '-s1\.req\.json$' | grep -v tts-long-male) $(ls $O/split/requests | grep -E -- '-s[123]\.req\.json$'); do
    n=${f%.req.json}; r=$O/split/requests/$f; [ -e $r ] || r=$O/single/requests/$f  # the split request where both exist (tts-long-male)
    /usr/bin/time -f "%e s %M KiB" $B/$build/lfm3-speech $M $r $O/$build/$n > $O/$build/$n.log 2> $O/$build/$n.err; echo "$build $n rc=$? $(tail -1 $O/$build/$n.err)"
  done
done
(cd $O && sha256sum */*.wav) > $O/wav.sha256
date
