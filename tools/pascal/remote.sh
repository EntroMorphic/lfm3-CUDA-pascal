#!/usr/bin/env bash
# Sync this repository to the GTX 1080 machine, verify the copy, and run a command in the container.
#
#   tools/pascal/remote.sh sync                 sync and verify only
#   tools/pascal/remote.sh run  'COMMAND'       sync, verify, then run COMMAND in the container
#   tools/pascal/remote.sh exec 'COMMAND'       run COMMAND without syncing
#   tools/pascal/remote.sh detach NAME 'COMMAND'  sync, verify, start COMMAND detached; /workspace/build/jobs/NAME.{log,rc}
#
# COMMAND runs with bash -l as uid 1000 in /workspace/lfm3-CUDA-pascal (this repository). Next to it:
# /workspace/models (checkpoints), /workspace/data, /workspace/build (build outputs; never synced or deleted).
# The exit status is COMMAND's. A sync whose tree hash differs between the two ends fails before anything runs.
# Only what git does not ignore is synced (outputs/, the owner's listening sets and keys among what is not), and the
# main tree is not synced while a detached job (a NAME.log without its NAME.rc) is running from it (a job that died
# without its .rc, say at a reboot, blocks syncs until its .rc is written by hand).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
[ -f "$here/target.env" ] || { echo "remote.sh: tools/pascal/target.env missing (see target.env.example)" >&2; exit 2; }
# shellcheck disable=SC1091
. "$here/target.env"
: "${PASCAL_JUMP:?}" "${PASCAL_JUMP_PORT:?}" "${PASCAL_HOST:?}" "${PASCAL_HOST_PORT:?}" "${PASCAL_CONTAINER:?}" "${PASCAL_HOST_WORKSPACE:?}"
# PASCAL_TREE: another directory name on the target, so a tree can be synced and built while a job runs from the main one.
name=${PASCAL_TREE:-lfm3-CUDA-pascal}
name_repo=$name
ssh_opts=(-o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=15)
# What is not part of the synced tree: these paths at the top of the repository, and these names at any depth. The
# same lists drive rsync and both tree hashes. The hash covers the contents of regular files (the tree has no
# symbolic links); it is of the working tree, so the line printed after a sync also says which commit that is.
excludes_top=(.git reference build native/build outputs .vscode .idea tools/pascal/target.env)
excludes_any=(__pycache__ .pytest_cache '*.pyc' '*.swp' .DS_Store '*.safetensors' '*.bf16')

on_host() {  # runs "$1" in a shell on the GPU host, stdin passed through
  ssh "${ssh_opts[@]}" -p "$PASCAL_JUMP_PORT" "$PASCAL_JUMP" \
    "ssh ${ssh_opts[*]} -p $PASCAL_HOST_PORT $PASCAL_HOST $(printf '%q' "$1")"
}

# One sha256 over the sorted list of (sha256, path) of every file in the tree.
tree_hash_cmd() {
  local prune=() e
  for e in "${excludes_top[@]}"; do prune+=(-path "$(printf '%q' "./$e")" -o); done
  for e in "${excludes_any[@]}"; do prune+=(-name "$(printf '%q' "$e")" -o); done
  unset 'prune[${#prune[@]}-1]'
  echo "find . \\( ${prune[*]} \\) -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1"
}

# Every file git ignores must be in the excludes above: refuse otherwise (a new ignored path would be synced).
check_ignored() {  # every ignored file by its own path (red-team-2 R2B-6: --directory also named directories that hold one)
  local all left
  # NUL-separated (-z): paths as they are, never quoted, so the patterns below see every name
  all=$(cd "$repo" && git ls-files -oi --exclude-standard -z | tr '\0' '\n'; exit "${PIPESTATUS[0]}") ||
    { echo "remote.sh: git ls-files failed" >&2; exit 2; }
  left=$(printf '%s\n' "$all" | grep -vE \
    '^(reference|build|native/build|outputs|\.vscode|\.idea)/|^tools/pascal/target\.env$|(^|/)(__pycache__|\.pytest_cache)/|\.(pyc|swp|safetensors|bf16)$|(^|/)\.DS_Store$|^$' || true)
  [ -z "$left" ] || { printf 'remote.sh: git-ignored but not excluded from the sync:\n%s\n' "$left" >&2; exit 2; }
}

# A detached job reads the main tree while it runs: no sync of it until every job has its .rc.
check_no_job() {
  [ "$name" = lfm3-CUDA-pascal ] || return 0
  local running
  running=$(on_host "cd $PASCAL_HOST_WORKSPACE/build/jobs 2>/dev/null || exit 0; for l in *.log; do [ -e \"\$l\" ] || continue; [ -e \"\${l%.log}.rc\" ] || echo \"\${l%.log}\"; done" </dev/null)
  [ -z "$running" ] || { echo "remote.sh: job(s) still running from the main tree, not syncing: $running" >&2; exit 2; }
}

synced=""
sync() {
  local rs=() e cmd local_hash remote_hash mode commit
  check_ignored
  check_no_job
  for e in "${excludes_top[@]}"; do rs+=(--exclude "/$e"); done
  for e in "${excludes_any[@]}"; do rs+=(--exclude "$e"); done
  commit=$(cd "$repo" && git rev-parse --short HEAD 2>/dev/null || echo "no commit")
  [ -z "$(cd "$repo" && git status --porcelain 2>/dev/null)" ] || commit="$commit plus uncommitted changes"
  cmd=$(tree_hash_cmd)
  # First by size and time; if the hashes then differ (a remote file changed with neither), once more by content.
  for mode in -a -ac; do
    rsync "$mode" --delete "${rs[@]}" \
      -e "ssh ${ssh_opts[*]} -p $PASCAL_JUMP_PORT $PASCAL_JUMP ssh ${ssh_opts[*]} -p $PASCAL_HOST_PORT -l ${PASCAL_HOST%@*}" \
      "$repo/" "${PASCAL_HOST#*@}:$PASCAL_HOST_WORKSPACE/$name/"
    local_hash=$(cd "$repo" && eval "$cmd")
    remote_hash=$(on_host "mkdir -p $PASCAL_HOST_WORKSPACE/build && cd $PASCAL_HOST_WORKSPACE/$name && $cmd" </dev/null)
    if [ "$local_hash" = "$remote_hash" ] && [ ${#local_hash} -eq 64 ]; then
      synced="tree sha256 $local_hash; $commit"
      echo "remote.sh: synced, $synced (both ends)" >&2; return 0
    fi
    echo "remote.sh: tree differs after rsync $mode (local=$local_hash remote=$remote_hash)" >&2
  done
  echo "remote.sh: SYNC MISMATCH" >&2; exit 3
}

run() {
  local job=".job.$$.$RANDOM.sh" rc=0
  printf '%s\n' "$1" | on_host "cat > $PASCAL_HOST_WORKSPACE/$job"
  on_host "sudo lxc exec $PASCAL_CONTAINER --user 1000 --group 1000 --cwd /workspace/$name --env HOME=/home/ubuntu -- bash -l /workspace/$job </dev/null; rc=\$?; rm -f $PASCAL_HOST_WORKSPACE/$job; exit \$rc" </dev/null || rc=$?
  return $rc
}

detach() {  # NAME COMMAND: survives this connection; NAME.rc appears when it ends
  local name=$1 jobs=$PASCAL_HOST_WORKSPACE/build/jobs
  [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "remote.sh: job name must be [A-Za-z0-9._-]" >&2; exit 2; }
  on_host "mkdir -p $jobs && test ! -e $jobs/$name.log && test ! -e $jobs/$name.rc" </dev/null || { echo "remote.sh: job $name exists" >&2; exit 2; }
  # The job's first line: the tree it was started on (the sync just done).
  printf '%s\n' "echo $(printf '%q' "remote.sh: $synced")" "$2" | on_host "cat > $jobs/$name.sh"
  on_host "nohup sh -c 'sudo lxc exec $PASCAL_CONTAINER --user 1000 --group 1000 --cwd /workspace/$name_repo --env HOME=/home/ubuntu -- bash -l /workspace/build/jobs/$name.sh </dev/null > $jobs/$name.log 2>&1; echo \$? > $jobs/$name.rc' > /dev/null 2>&1 &" </dev/null
  echo "remote.sh: started $name; log /workspace/build/jobs/$name.log" >&2
}

case "${1:-}" in
  sync) sync ;;
  run)  [ $# -eq 2 ] || { echo "usage: remote.sh run 'COMMAND'" >&2; exit 2; }; sync; run "$2" ;;
  detach) [ $# -eq 3 ] || { echo "usage: remote.sh detach NAME 'COMMAND'" >&2; exit 2; }
          # the job scripts cd to /workspace/lfm3-CUDA-pascal: a job from another tree would run the main tree's tools
          [ "$name" = lfm3-CUDA-pascal ] || { echo "remote.sh: detach runs from the main tree only (PASCAL_TREE unset)" >&2; exit 2; }
          [[ $2 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "remote.sh: job name must be [A-Za-z0-9._-]" >&2; exit 2; }
          sync; detach "$2" "$3" ;;
  exec) [ $# -eq 2 ] || { echo "usage: remote.sh exec 'COMMAND'" >&2; exit 2; }; run "$2" ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
