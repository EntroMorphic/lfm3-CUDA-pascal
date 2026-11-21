# red-team-2 development 1 (tools/pascal/remote.sh detach rt2-dev1 'bash validation/red-team-2/dev1.sh'): the first
# fixes built (text-v36, parity-v36, tools/pascal/build_speech.sh), their device code against text-v35 / parity-v35
# (the speech-v1 rule run's builds), and the measurements that may change a constant: speech-v1 rule 12 with the clip
# at the request limit (D1) at 1, 8 and 16 rows; the /generate_batch bound (B1); the parity build's attention tiles
# above 4,096 tokens of capacity (A1: prompts 129 and 136 leave a last tile of 1 and 8 rows, 137 and 300 do not).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-dev1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }

EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v36 > $O/build-parity-v36.log 2>&1; echo "build parity-v36 rc=$?"
tools/pascal/build_speech.sh text-v36 > $O/build-text-v36.log 2>&1; echo "build text-v36 rc=$?"
grep -h "warnings" $O/build-*.log

# Device code, every binary, against v35
mkdir -p $O/e1
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server; do for k in text parity; do
  cuobjdump -sass $B/$k-v36/$f > $O/e1/a.sass; cuobjdump -sass $B/$k-v35/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.$k.txt; echo "device code $f $k-v36 against $k-v35 rc=$? $(head -1 $O/e1/$f.$k.txt)"
done; done; rm -f $O/e1/*.sass

# A1: the CLI at capacity 4,097, teacher-forced 16 steps, logits traced: v36 against v35, each build
python3 - $O/a1 <<'PY'
import json, sys
from array import array
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir()
a = array("i"); a.frombytes(open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read()); w = a[:32768]
for L in (129, 136, 137, 300):
    r = dict(input_ids=list(w[:L]), teacher_ids=list(w[L:L + 16]), capacity=4097, experimental_long_context=True,
             trace_logits_only=True)
    (out / f"p{L}.request.json").write_text(json.dumps(r))
PY
for k in text parity; do for v in v35 v36; do for L in 129 136 137 300; do
  d=$O/a1/$k-$v-p$L; mkdir -p $d; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r['trace_directory']=sys.argv[2]+'/trace'; json.dump(r,open(sys.argv[2]+'/request.json','w'))" $O/a1/p$L.request.json $d
  $B/$k-$v/liquid-lfm3 --experimental $M $d/request.json $d/native.json > $d/log 2>&1 || echo "A1 $k-$v p$L FAILED"
  (cd $d/trace && sha256sum $(ls *.logits.f32 | sort -n) | sha256sum | cut -c1-16) > $d/logits.sha; rm -rf $d/trace
done; done
for L in 129 136 137 300; do echo "A1 $k p$L: v35 $(cat $O/a1/$k-v35-p$L/logits.sha) v36 $(cat $O/a1/$k-v36-p$L/logits.sha) $(cmp -s $O/a1/$k-v35-p$L/logits.sha $O/a1/$k-v36-p$L/logits.sha && echo same || echo DIFFER)"; done; done

# B1: the batch bound, the worst case measured
$PY tools/pascal/batch_budget.py $B/text-v36/liquid-lfm3-server $M $O/batch > $O/batch.log 2>&1; echo "B1 rc=$?"; tail -2 $O/batch.log; clean

# D1: speech-v1 rule 12 with the full clip, 1 row at the default 30,720, then 8 rows at 16,384 and 16 at 2,048
for rt in 1:30720 8:16384 16:2048; do r=${rt%%:*}; t=${rt##*:}
  $PY tools/pascal/speech_memory.py $B/text-v36/liquid-lfm3-server $M $r $LIBRI $O/memory-$r 7740 $t > $O/memory-$r.log 2>&1
  echo "D1 rows $r total $t rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-$r.log; clean
done
date
