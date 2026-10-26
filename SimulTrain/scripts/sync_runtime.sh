#!/usr/bin/env bash
# Create a node-local runtime snapshot so torchrun workers never import Python
# source files from Ceph/FUSE concurrently.

a3b_sync_runtime() {
  local src="${1:?usage: a3b_sync_runtime <source_root> [destination]}"
  local dst="${2:-/tmp/simultrain-code}"
  local manifest_file="$dst/.runtime_manifest"
  local source_manifest

  source_manifest="$({
    find "$src/s2s_train" -type f ! -name '*.pyc' -printf 's2s_train/%P %s %T@\n'
    find "$src/vendor/LlamaFactory/src/llamafactory" -type f ! -name '*.pyc' \
      -printf 'vendor/LlamaFactory/src/llamafactory/%P %s %T@\n'
    find "$src/configs" -type f -printf 'configs/%P %s %T@\n'
    find "$src/vendor/LlamaFactory/examples/deepspeed" -maxdepth 1 -type f \
      -printf 'vendor/LlamaFactory/examples/deepspeed/%P %s %T@\n'
    printf 'train.sh %s %s\n' "$(stat -c %s "$src/train.sh")" "$(stat -c %Y "$src/train.sh")"
    printf 'train_merged.sh %s %s\n' "$(stat -c %s "$src/train_merged.sh")" "$(stat -c %Y "$src/train_merged.sh")"
    find "$src/scripts" -maxdepth 1 -type f -name '*.sh' -printf 'scripts/%f %s %T@\n'
  } | sort | sha256sum | awk '{print $1}')"

  if [[ -f "$manifest_file" ]] && [[ "$(cat "$manifest_file")" == "$source_manifest" ]]; then
    return 0
  fi

  local stage archive list
  stage="$(mktemp -d "${dst}.stage.XXXXXX")"
  archive="$(mktemp /tmp/a3b-merged-runtime.XXXXXX.tar)"
  list="$(mktemp /tmp/a3b-merged-runtime.XXXXXX.list)"

  (
    find "$src/s2s_train" -type f ! -name '*.pyc'
    find "$src/vendor/LlamaFactory/src/llamafactory" -type f ! -name '*.pyc'
    find "$src/configs" -type f
    find "$src/vendor/LlamaFactory/examples/deepspeed" -maxdepth 1 -type f
    printf '%s\n' "$src/train.sh" "$src/train_merged.sh"
    find "$src/scripts" -maxdepth 1 -type f -name '*.sh'
  ) | sed "s#^$src/##" | sort -u > "$list"

  tar -C "$src" -cf "$archive" -T "$list"
  tar -C "$stage" -xf "$archive"
  printf '%s\n' "$source_manifest" > "$stage/.runtime_manifest"

  # Destination is an explicit task-specific /tmp path. Replacing it is safe
  # and prevents workers from observing a half-written snapshot.
  if [[ -e "$dst" ]]; then
    mv "$dst" "${dst}.old.$$"
  fi
  mv "$stage" "$dst"
  echo "[runtime] node-local code ready: $dst"
}
