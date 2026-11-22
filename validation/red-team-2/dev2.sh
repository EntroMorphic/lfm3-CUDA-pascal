# red-team-2 development 2 (tools/pascal/remote.sh detach rt2-dev2 'bash validation/red-team-2/dev2.sh'): the second
# round's fixes built (text-v37, parity-v37). The server's speech defaults lowered after development 1 (speech-v1 rule
# 12 with the clip at the request limit peaked at 7,759 MiB at one row and 30,720, and 7,771 MiB at 16 rows and 2,048):
# 29,696 at one row, 2,048 less for each further row, at least 1,024. Here: the ready line at 1, 8 and 16 rows; rule 12
# at those defaults (1: 29,696; 8: 15,360; 16: 1,024); the /generate_batch bound measured (batch_budget.py, its tool
# fixed); A1 in the attention itself: the parity build's attention probabilities for prompts 129 and 136 at capacity
# 4,097 (their last tiles of 1 and 8 rows), v37 against v35, every layer's trace file of that tile.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-dev2; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }

EXTRA_FLAGS=-DLFM_GEMM_TENSOR_CORE_MODEL tools/pascal/build_speech.sh parity-v37 > $O/build-parity-v37.log 2>&1; echo "build parity-v37 rc=$?"
tools/pascal/build_speech.sh text-v37 > $O/build-text-v37.log 2>&1; echo "build text-v37 rc=$?"
grep -h "warnings" $O/build-*.log
mkdir -p $O/e1
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server; do for k in text parity; do
  cuobjdump -sass $B/$k-v37/$f > $O/e1/a.sass; cuobjdump -sass $B/$k-v35/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.$k.txt; echo "device code $f $k-v37 against $k-v35 rc=$? $(head -1 $O/e1/$f.$k.txt)"
done; done; rm -f $O/e1/*.sass

# The ready line with the defaults, at 1, 8 and 16 rows
for r in 1 8 16; do $PY - $B/text-v37/liquid-lfm3-server $M $O $r <<'PY'
import json, os, secrets, signal, subprocess, sys
server, model, out, rows = sys.argv[1:5]; tok = os.path.join(out, f"ready{rows}.token")
fd = os.open(tok, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, (secrets.token_urlsafe(32) + "\n").encode()); os.close(fd)
p = subprocess.Popen([server, "--experimental", model, "--token-file", tok, "--port", "0", "--speech-rows", rows],
                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
line = p.stdout.readline(); os.unlink(tok); p.send_signal(signal.SIGTERM); rc = p.wait(120)
want = {"1": 29696, "8": 15360, "16": 1024}[rows]
r = json.loads(line); print(f"ready line at {rows} rows:", line.strip(), "exit", rc,
      "ok" if (r.get("total_capacity"), r.get("speech_rows"), rc) == (want, int(rows), 0) else "FAIL")
PY
clean; done

# A1: attention probabilities of the last tile, parity v35 and v37, capacity 4,097, prompts 129 and 136
python3 - $O/a1 <<'PY'
import json, sys
from array import array
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir()
a = array("i"); a.frombytes(open("/workspace/data/ppl-windows/windows-32768.i32", "rb").read()); w = a[:32768]
for L in (129, 136):
    (out / f"p{L}.request.json").write_text(json.dumps(dict(input_ids=list(w[:L]), capacity=4097, experimental_long_context=True,
                                                            max_new_tokens=1)))
PY
for v in v35 v37; do for L in 129 136; do
  d=$O/a1/$v-p$L; mkdir -p $d; python3 -c "import json,sys; r=json.load(open(sys.argv[1])); r['trace_directory']=sys.argv[2]+'/trace'; json.dump(r,open(sys.argv[2]+'/request.json','w'))" $O/a1/p$L.request.json $d
  $B/parity-$v/liquid-lfm3 --experimental $M $d/request.json $d/native.json > $d/log 2>&1 || echo "A1 parity-$v p$L FAILED"
  (cd $d/trace && shopt -s nullglob && f=(*probabilities.tile128*) && echo "${#f[@]} files" && sha256sum "${f[@]}") > $d/tile.sha256
  (cd $d/trace && shopt -s nullglob && f=(*.logits.f32) && echo "${#f[@]} logits files" && sha256sum "${f[@]}") > $d/logits.sha256
  rm -rf $d/trace
done; done
for L in 129 136; do python3 - $O/a1/v35-p$L $O/a1/v37-p$L $L <<'PY'
import sys
a, b, L = sys.argv[1:4]
def read(p):
    lines = open(p).read().split("\n"); n = int(lines[0].split()[0]); return n, dict(reversed(l.split()) for l in lines[1:] if l)
(na, ta), (nb, tb) = read(f"{a}/tile.sha256"), read(f"{b}/tile.sha256")
(la, xa), (lb, xb) = read(f"{a}/logits.sha256"), read(f"{b}/logits.sha256")
differ = sum(ta[k] != tb.get(k) for k in ta)
print(f"A1 parity p{L}: last tile's probability traces {na} and {nb} files, {differ} differ between v35 and v37; "
      f"logits {la} and {lb} files, {'identical' if xa == xb else 'DIFFER'}" + ("" if na and na == nb and la and la == lb else ": INCOMPLETE"))
PY
done

# B1: the bound, the worst case measured
$PY tools/pascal/batch_budget.py $B/text-v37/liquid-lfm3-server $M $O/batch > $O/batch.log 2>&1; echo "B1 rc=$?"; tail -2 $O/batch.log; clean

# speech-v1 rule 12 at the new defaults
for rt in 1:29696 8:15360 16:1024; do r=${rt%%:*}; t=${rt##*:}
  $PY tools/pascal/speech_memory.py $B/text-v37/liquid-lfm3-server $M $r $LIBRI $O/memory-$r 7740 $t > $O/memory-$r.log 2>&1
  echo "rule 12 rows $r total $t rc=$?"; grep -E '^rule 12|^SPEECH' $O/memory-$r.log; clean
done
date
