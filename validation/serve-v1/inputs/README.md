The reference's gate inputs, copied from the reference repository (`validation/` there; read-only from here):

- `generation-request.json`: `forward-bounded-prefill-v1/baseline/generation-request.json` (REQ in `tools/thor/run_native_gates.sh`)
- `decode_into32768.request.json`: `context32k-bounded-prefill-v1/decode_into32768/request.json` (LONG)
- `control.request.json`, `control.output.json`: `frontend-remediation/post-remediation-native/{request,output}.json`
  (CTL_REQ, CTL_RES), whose SHA-256 values are the ones `run_native_gates.sh` pins (5651a375…, 5d1a4ecb…).
