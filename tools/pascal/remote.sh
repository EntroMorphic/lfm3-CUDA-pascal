#!/usr/bin/env bash
# Sync this repository to the GTX 1080 machine, verify the copy, and run a command in the container.
#
#   tools/pascal/remote.sh sync                 sync and verify only
#   tools/pascal/remote.sh run  'COMMAND'       sync, verify, then run COMMAND in the container
#   tools/pascal/remote.sh exec 'COMMAND'       run COMMAND without syncing
#
# COMMAND runs with bash -l as uid 1000 in /workspace/lfm3-CUDA-pascal (this repository). Next to it:
# /workspace/models (checkpoints), /workspace/data, /workspace/build (build outputs; never synced or deleted).
# The exit status is COMMAND's. A sync whose tree hash differs between the two ends fails before anything runs.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
[ -f "$here/target.env" ] || { echo "remote.sh: tools/pascal/target.env missing (see target.env.example)" >&2; exit 2; }
# shellcheck disable=SC1091
. "$here/target.env"
: "${PASCAL_JUMP:?}" "${PASCAL_JUMP_PORT:?}" "${PASCAL_HOST:?}" "${PASCAL_HOST_PORT:?}" "${PASCAL_CONTAINER:?}" "${PASCAL_HOST_WORKSPACE:?}"
name=lfm3-CUDA-pascal
ssh_opts=(-o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=15)
# What is not part of the synced tree. The same list drives rsync and both tree hashes.
excludes=(.git reference build __pycache__ .pytest_cache '*.pyc' target.env)

on_host() {  # runs "$1" in a shell on the GPU host, stdin passed through
  ssh "${ssh_opts[@]}" -p "$PASCAL_JUMP_PORT" "$PASCAL_JUMP" \
    "ssh ${ssh_opts[*]} -p $PASCAL_HOST_PORT $PASCAL_HOST $(printf '%q' "$1")"
}

# One sha256 over the sorted list of (sha256, path) of every file in the tree.
tree_hash_cmd() {
  local prune=() e
  for e in "${excludes[@]}"; do prune+=(-name "$(printf '%q' "$e")" -o); done
  unset 'prune[${#prune[@]}-1]'
  echo "find . \\( ${prune[*]} \\) -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1"
}

sync() {
  local rs=() e
  for e in "${excludes[@]}"; do rs+=(--exclude "$e"); done
  rsync -a --delete "${rs[@]}" \
    -e "ssh ${ssh_opts[*]} -p $PASCAL_JUMP_PORT $PASCAL_JUMP ssh ${ssh_opts[*]} -p $PASCAL_HOST_PORT -l ${PASCAL_HOST%@*}" \
    "$repo/" "${PASCAL_HOST#*@}:$PASCAL_HOST_WORKSPACE/$name/"
  local cmd local_hash remote_hash
  cmd=$(tree_hash_cmd)
  local_hash=$(cd "$repo" && eval "$cmd")
  remote_hash=$(on_host "mkdir -p $PASCAL_HOST_WORKSPACE/build && cd $PASCAL_HOST_WORKSPACE/$name && $cmd" </dev/null)
  if [ "$local_hash" != "$remote_hash" ] || [ ${#local_hash} -ne 64 ]; then
    echo "remote.sh: SYNC MISMATCH local=$local_hash remote=$remote_hash" >&2; exit 3
  fi
  echo "remote.sh: synced, tree sha256 $local_hash (both ends)" >&2
}

run() {
  local job=".job.$$.$RANDOM.sh" rc=0
  printf '%s\n' "$1" | on_host "cat > $PASCAL_HOST_WORKSPACE/$job"
  on_host "sudo lxc exec $PASCAL_CONTAINER --user 1000 --group 1000 --cwd /workspace/$name --env HOME=/home/ubuntu -- bash -l /workspace/$job </dev/null; rc=\$?; rm -f $PASCAL_HOST_WORKSPACE/$job; exit \$rc" </dev/null || rc=$?
  return $rc
}

case "${1:-}" in
  sync) sync ;;
  run)  [ $# -eq 2 ] || { echo "usage: remote.sh run 'COMMAND'" >&2; exit 2; }; sync; run "$2" ;;
  exec) [ $# -eq 2 ] || { echo "usage: remote.sh exec 'COMMAND'" >&2; exit 2; }; run "$2" ;;
  *) sed -n '2,10p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
