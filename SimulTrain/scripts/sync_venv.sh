#!/usr/bin/env bash
# Optional node-local virtualenv snapshot for shared-filesystem deployments.
# Set HOSTFILE to distribute to listed hosts; unset means this node only.
# SSH_PORT defaults to 22. VENV_REFRESH=1 rebuilds the local snapshot.

s2s_sync_venv() {
  local src="${1:?用法: s2s_sync_venv <venv源目录> [本地目标目录]}"
  local dst="${2:-/tmp/simultrain-venv}"
  local hostfile="${HOSTFILE:-}"
  local port="${SSH_PORT:-22}"
  local ssh_opts=(-p "$port" -o StrictHostKeyChecking=no -o BatchMode=yes
                  -o ConnectTimeout=10)
  local stage="/tmp/.a3b-merged-venv-stage" tarball="/tmp/.a3b-merged-venv.tar"
  local want ips=() todo=() ip

  [[ -d "$src" ]] || { echo "[venv] 源目录不存在：$src" >&2; return 1; }
  [[ "${VENV_REFRESH:-0}" == "1" ]] && rm -rf "$stage" "$tarball" "$dst"

  # 1) 唯一一次从 cephfs 读 venv：落到本机本地盘再打包
  if [[ ! -f "$tarball" ]]; then
    rm -rf "$stage"
    cp -a "$src" "$stage" || { echo "[venv] 从 $src 复制失败" >&2; return 1; }
    tar -C "$stage" -cf "$tarball" . || return 1
  fi
  want="$(tar -tf "$tarball" | wc -l)"

  [[ -f "$hostfile" ]] && mapfile -t ips < <(awk '!/^#/ && NF {print $1}' "$hostfile")

  # 单机（没有 hostfile）：只保证本机这一份
  if [[ ${#ips[@]} -eq 0 ]]; then
    [[ "$(cat "$dst/.sync_done" 2>/dev/null)" == "$want" ]] && return 0
    rm -rf "$dst.part" && mkdir -p "$dst.part" || return 1
    tar -C "$dst.part" -xf "$tarball" || return 1
    echo "$want" > "$dst.part/.sync_done"
    rm -rf "$dst" && mv "$dst.part" "$dst"
    echo "[venv] 已就绪（单机）：$dst"
    return 0
  fi

  # 2) 先探测谁缺。分开探测是为了不给已就绪的节点白推 388 MB，
  #    也避免「远端提前 exit 0、本地 ssh 收到 EPIPE」被误判成失败。
  for ip in "${ips[@]}"; do
    ssh "${ssh_opts[@]}" "$ip" \
      "[ \"\$(cat '$dst/.sync_done' 2>/dev/null)\" = '$want' ]" \
      </dev/null >/dev/null 2>&1 || todo+=("$ip")
  done
  [[ ${#todo[@]} -eq 0 ]] && return 0
  echo "[venv] 分发到 ${#todo[@]}/${#ips[@]} 台本地盘（$dst）"

  # 3) 解到 .part 校验条目数后才改名 —— 半截的传输不会被后续误认成完整副本
  for ip in "${todo[@]}"; do
    (
      ssh "${ssh_opts[@]}" "$ip" "
        rm -rf '$dst.part' && mkdir -p '$dst.part' || exit 1
        tar -C '$dst.part' -xf - || exit 1
        [ \"\$(find '$dst.part' | wc -l)\" = '$want' ] || exit 1
        echo '$want' > '$dst.part/.sync_done'
        rm -rf '$dst' && mv '$dst.part' '$dst'
      " < "$tarball" >/dev/null 2>&1 || echo "[venv] 分发失败：$ip" >&2
    ) &
  done
  wait
}
