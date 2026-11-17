#!/usr/bin/env bash
# validation/multilingual-speech-v1 rule 0: FLEURS test for the 18 speech languages, on the workstation (it has the
# internet), then copied to Thor. Run only with Tripp's OK (7.74 GB of audio, 8 MB of transcripts; CC-BY-4.0).
# Writes data/fleurs/CONFIG/{test.tsv, audio/test.tar.gz, audio/test/*.wav} and data/fleurs/sha256.txt (every
# downloaded file, recorded before extraction). --tsv-only fetches the transcripts alone (enough for rules 0-3, 5-9).
# Usage: fetch_fleurs.sh [--tsv-only]   then: rsync -a data/fleurs/ "$THOR_HOST:<workspace>/data/fleurs/"
set -euo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd); D=$W/data/fleurs; mkdir -p "$D"
URL=https://huggingface.co/datasets/google/fleurs/resolve/main/data
CONFIGS="ar_eg da_dk de_de el_gr en_us fi_fi fr_fr he_il it_it ja_jp ko_kr nl_nl pl_pl pt_br es_419 sv_se tr_tr cmn_hans_cn"
for c in $CONFIGS; do
  mkdir -p "$D/$c/audio"
  [ -s "$D/$c/test.tsv" ] || curl -fL --retry 3 -o "$D/$c/test.tsv" "$URL/$c/test.tsv"
  if [ "${1:-}" != --tsv-only ]; then
    [ -s "$D/$c/audio/test.tar.gz" ] || curl -fL --retry 3 -o "$D/$c/audio/test.tar.gz" "$URL/$c/audio/test.tar.gz"
  fi
done
(cd "$D" && find . -name test.tsv -o -name test.tar.gz | sort | xargs sha256sum) > "$D/sha256.txt"
if [ "${1:-}" != --tsv-only ]; then
  for c in $CONFIGS; do [ -d "$D/$c/audio/test" ] || tar -xzf "$D/$c/audio/test.tar.gz" -C "$D/$c/audio"; done
fi
echo "FLEURS in $D: $(wc -l < "$D/sha256.txt") files recorded"
