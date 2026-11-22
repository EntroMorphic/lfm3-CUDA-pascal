# red-team-2 verification, part 3 (tools/pascal/remote.sh detach rt2-v1c 'bash validation/red-team-2/job3.sh'): job.sh
# was stopped (owner, 2026-11-21: "Let's optimize for time without sacrificing quality") once the shipping build's
# speech-v1 rule 11 was judged, before the parity build's five checks (about 6.5 hours). Here, in their place and after:
#  - speech-v1 rule 11 on the parity build: audio in run again on parity-v38 (minutes); speech-server, stream and
#    concurrent carried from development 12 (parity-s4) and batch-server from the speech-v1 rule run (parity-v35). The
#    fixes since cannot reach those checks: device code identical (job.sh's E1); their text sessions are of 1,024 tokens
#    (A1 acts above 4,096), single turns (the continuation cap), no /generate_batch (its bound), default engine timeout;
#    /speech runs on the 4,096-token speech runner (one tile) outside the session ledger; the defaults and startup
#    checks are host code the two builds share, run on the final shipping server by job.sh's rule 11.
#  - the steps of job.sh after the stopped loop: prefill-arithmetic-v1's position-1 tables (C10).
#  - job2.sh's continuation steps (R5-1, with round 7's tool).
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/rt2-v1; M=/workspace/models/LFM3-Omni-2B
PY=$B/venv-asr/bin/python; LIBRI=/workspace/data/librispeech/wav
clean() { pkill -TERM -f liquid-lfm3-server; for i in $(seq 120); do pgrep -f liquid-lfm3-server >/dev/null || return 0; sleep 1; done; pkill -KILL -f liquid-lfm3-server; sleep 5; }
[ -d $O ] || { echo "$O missing"; exit 2; }
rm -rf $O/parity-speech-server.json $O/parity-audio-in-work  # anything the stopped loop began for the parity build
S=$B/parity-v38; R=$B/sp-refs-parity-s1
/usr/bin/time -f "%e s" $PY tools/pascal/check_lfm3_audio_in_server.py $S/liquid-lfm3-server $S/lfm3-speech $M $LIBRI $O/parity-audio-in-work $O/parity-audio-in-server.json > $O/parity-audio-in-server.log 2>&1
echo "parity-audio-in-server rc=$? $(grep -E 'PASS|FAIL' $O/parity-audio-in-server.log | tail -1)"; clean
cp $B/sp-dev12/parity-speech-server.json $B/sp-dev12/parity-speech-stream.json $B/sp-dev12/parity-speech-concurrent.json $O/
cp $B/sp-v1/parity-batch-server.json $O/
python3 tools/pascal/judge_speech_server.py $O parity $B/parity-v38,$B/parity-v35,$B/parity-s4 $R $B/parity-s1; echo "speech-v1 rule 11 parity rc=$?"

$PY tools/pascal/analysis/arith_report.py $B/arith-v1 > $O/arith-position1.txt 2>&1; echo "arith_report rc=$?"
cmp $O/arith-position1.txt validation/prefill-arithmetic-v1/arith-position1.txt && echo "arith-position1.txt: equal to the committed one" || echo "arith-position1.txt: DIFFERS"

bash validation/red-team-2/job2.sh 2>&1 | grep -v '^+'
date
