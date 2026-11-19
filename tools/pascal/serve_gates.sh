#!/usr/bin/env bash
# Runs in the container: the reference's serving gates (tools/thor/run_native_gates.sh's sampling, worker,
# grouped-worker, server, cancellation, http-cancellation, http-batch and native-server stages, with the same
# arguments) against build TAG (tools/pascal/build_serve.sh), every stage with LFM_ALLOCATION_METRICS=1 and http-batch
# judging exact engine bytes. serve-v1 rules 3 to 9.
#   tools/pascal/serve_gates.sh TAG OUT CONTROL
# CONTROL=pinned: the reference's control files (validation/serve-v1/inputs, SHA-256 checked as the reference checks
# them); the parity build. CONTROL=own: TAG's CLI output on the pinned control request, recorded first; the shipping build.
# Every stage runs (KEEP_GOING); OUT/status.txt lists each stage's exit status; the exit status is 1 if any failed.
set -uo pipefail
tag=${1:?TAG}; name=${2:?OUT}; control=${3:?CONTROL}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); T=$repo/tools; I=$repo/validation/serve-v1/inputs
B=/workspace/build; N=$B/$tag; out=$B/$name; CK=/workspace/models/LFM3-Omni-2B
test ! -e "$out" || { echo "exists: $out"; exit 2; }; mkdir -p "$out"
for f in liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle; do test -x "$N/$f" || { echo "missing $N/$f"; exit 2; }; done
(cd "$N" && sha256sum liquid-lfm3 liquid-lfm3-worker liquid-lfm3-server text-sampling-oracle) > "$out/binaries.sha256"
for pin in "control.request.json:5651a375a4dd1144c6f49e72a666cf2423ef3062b5502821381a6936ed1c4946" \
           "control.output.json:5d1a4ecbd9be9f3de6d85fa888d640610c32e33caa1729c4f7a1fd8f5fff230b"; do
  f=${pin%%:*}; [ "$(sha256sum "$I/$f" | cut -c1-64)" = "${pin#*:}" ] || { echo "control $f differs from the pinned one"; exit 2; }
done
CTL_REQ=$I/control.request.json
case $control in
  pinned) CTL_RES=$I/control.output.json ;;
  own) CTL_RES=$out/control.output.json
       "$N/liquid-lfm3" --experimental "$CK" "$CTL_REQ" "$CTL_RES" > "$out/control.log" 2>&1 || { echo "control run failed"; exit 1; }
       python3 -c "import json,sys; o=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); g=o['generated_ids']; p=r['generated_ids']; d=next((i for i,(a,b) in enumerate(zip(g,p)) if a!=b),None); print('own control:',len(g),'tokens;', 'identical to the pinned control' if g==p else 'first differs from the pinned control at token %s' % d)" "$CTL_RES" "$I/control.output.json" | tee "$out/control.txt" ;;
  *) echo "CONTROL must be pinned or own"; exit 2 ;;
esac
sha256sum "$CTL_REQ" "$CTL_RES" > "$out/control.sha256"
REQ=$I/generation-request.json; LONG=$I/decode_into32768.request.json; O=$out
export LFM_ALLOCATION_METRICS=1
failed=0; : > "$out/status.txt"
run() { local s=$1; shift; local t0=$(date +%s); python3 "$@" > "$out/$s.log" 2>&1; local rc=$?
  echo "$s rc=$rc $(( $(date +%s) - t0 )) s" | tee -a "$out/status.txt"; [ $rc = 0 ] || failed=1; }
run sampling          $T/check_lfm3_sampling.py --checkpoint $CK --binary $N/liquid-lfm3 --request $REQ --oracle $N/text-sampling-oracle --output $O/sampling
run worker            $T/check_lfm3_worker.py --checkpoint $CK --worker $N/liquid-lfm3-worker --single $N/liquid-lfm3 --request $REQ --long-request $LONG --oracle $N/text-sampling-oracle --output $O/worker
run grouped-worker    $T/check_lfm3_batch_worker.py --checkpoint $CK --worker $N/liquid-lfm3-worker --request $REQ --oracle $N/text-sampling-oracle --long-request $LONG --output $O/grouped-worker
run server            $T/check_lfm3_server.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-result $CTL_RES --concurrency-rounds 3 --output $O/server
run cancellation      $T/check_lfm3_cancellation.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-request $CTL_REQ --control-result $CTL_RES --output $O/cancellation
run http-cancellation $T/check_lfm3_http_cancellation.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-result $CTL_RES --output $O/http-cancellation
run http-batch        $T/check_lfm3_http_batch.py --checkpoint $CK --worker $N/liquid-lfm3-worker --control-result $CTL_RES --output $O/http-batch --require-exact-accounting
run native-server     $T/check_lfm3_native_server.py --checkpoint $CK --server $N/liquid-lfm3-server --worker $N/liquid-lfm3-worker --control-result $CTL_RES --output $O/native-server
[ $failed = 0 ] && echo "SERVE GATES PASS ($tag, control $control)" || echo "SERVE GATES FAIL ($tag, control $control)"
exit $failed
