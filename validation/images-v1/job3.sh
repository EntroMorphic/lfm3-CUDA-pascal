# images-v1, part 3 (tools/pascal/remote.sh detach img-v1c 'bash validation/images-v1/job3.sh'): the owner's item 2
# remedied. A connection past --handlers is now answered 429 connection_limit (lfm3_http.hpp Server::refuse_loop)
# instead of being closed without a reply. Here: text-v41 built; its device code against text-v40 (only host code
# changed); rule 6's server check at the server's defaults (8 handlers), the configuration job.sh's check stopped in.
set -x; date; cd /workspace/lfm3-CUDA-pascal; B=/workspace/build; O=$B/img-v1c; M=/workspace/models/LFM3-Omni-2B; PY=$B/venv-asr/bin/python
[ -e $O ] && { echo "$O exists"; exit 2; }; mkdir -p $O/e1
tools/pascal/build_speech.sh text-v41 > $O/build-text-v41.log 2>&1; echo "build text-v41 rc=$?"; grep -h warnings $O/build-text-v41.log
for f in liquid-lfm3 lfm3-score lfm3-speech liquid-lfm3-worker liquid-lfm3-server lfm3-vision-check; do
  cuobjdump -sass $B/text-v41/$f > $O/e1/a.sass; cuobjdump -sass $B/text-v40/$f > $O/e1/b.sass
  python3 validation/speed-v1/sass_compare.py $O/e1/a.sass $O/e1/b.sass > $O/e1/$f.txt; echo "device code $f text-v41 against text-v40 rc=$? $(head -1 $O/e1/$f.txt)"
done; rm -f $O/e1/*.sass
$PY tools/check_lfm3_image_server.py $B/text-v41/liquid-lfm3-server $M $B/img-v1/e2e $O/image-server.json > $O/image-server.log 2>&1
echo "rule 6 server (defaults: 8 handlers) rc=$? $(grep -E '^concurrent' $O/image-server.log | cut -c1-400) $(tail -1 $O/image-server.log)"
pkill -TERM -f liquid-lfm3-server; date
