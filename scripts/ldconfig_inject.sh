#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# ldconfig_inject.sh
# 将 TensorRT（以及可选 CUDA）库目录永久写入 /etc/ld.so.conf.d/tensorrt.conf，并执行 ldconfig
#
# 用法示例：
#   sudo ./ldconfig_inject.sh /opt/tensorrt/lib
#   sudo ./ldconfig_inject.sh /opt/tensorrt/lib /usr/local/cuda/lib64
#   sudo ./ldconfig_inject.sh --remove
#   ./ldconfig_inject.sh --dry-run /opt/tensorrt/lib
#   ./ldconfig_inject.sh help
#
# 说明：
# - 仅支持 Linux。
# - 需要 root（若不是 root 且非 --dry-run，将自动以 sudo 重新执行）。
# - 幂等：会与已有文件合并并去重。
# - 校验目录存在并（对 TRT）检查是否含 libnvinfer*.so*。
#
set -euo pipefail

CONF="/etc/ld.so.conf.d/tensorrt.conf"

COLOR_INFO="\033[1;34m[INFO]\033[0m"
COLOR_WARN="\033[1;33m[WARN]\033[0m"
COLOR_ERR="\033[1;31m[ERROR]\033[0m"

info(){ printf "%b %s\n" "${COLOR_INFO}" "$*"; }
warn(){ printf "%b %s\n" "${COLOR_WARN}" "$*"; }
err(){  printf "%b %s\n" "${COLOR_ERR}"  "$*" >&2; }

is_linux(){
  [[ "$(uname -s)" == "Linux" ]]
}

print_usage(){
  cat <<'EOF'
用法:
  sudo ./ldconfig_inject.sh /opt/tensorrt/lib
  sudo ./ldconfig_inject.sh /opt/tensorrt/lib /usr/local/cuda/lib64
  sudo ./ldconfig_inject.sh --remove
  ./ldconfig_inject.sh --dry-run /opt/tensorrt/lib

说明:
  - 将给定库目录写入 /etc/ld.so.conf.d/tensorrt.conf（去重合并），然后运行 ldconfig。
  - --dry-run: 仅显示将要写入的内容与操作，不做任何更改，也不提权。
  - --remove : 删除 /etc/ld.so.conf.d/tensorrt.conf 并运行 ldconfig。
  - help / -h / --help: 打印使用方法与用例。

用例:
  1) 注册 TensorRT 库目录：
     sudo ./ldconfig_inject.sh /opt/tensorrt/lib

  2) 同时注册 TensorRT 与 CUDA 库目录：
     sudo ./ldconfig_inject.sh /opt/tensorrt/lib /usr/local/cuda/lib64

  3) 仅演示（不落盘）：
     ./ldconfig_inject.sh --dry-run /opt/tensorrt/lib

  4) 移除 tensorrt.conf 后重建缓存：
     sudo ./ldconfig_inject.sh --remove
EOF
}

# 解析参数
DRY_RUN=false
DO_REMOVE=false
LIBDIRS=()

if [[ $# -eq 0 ]]; then
  err "未提供任何库目录。"
  echo
  print_usage
  exit 1
fi

# 如果出现 help 子命令或 -h/--help，则打印用法与用例
for a in "$@"; do
  if [[ "$a" == "help" || "$a" == "-h" || "$a" == "--help" ]]; then
    print_usage
    exit 0
  fi
done

# 基础平台检查
if ! is_linux; then
  err "仅支持 Linux。"
  exit 1
fi

# 逐一读取参数
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --remove)  DO_REMOVE=true ;;
    *)         LIBDIRS+=("$arg") ;;
  esac
done

# 提权（非 dry-run 且需要更改系统时）
require_root_or_reexec(){
  if "$DRY_RUN"; then
    return 0
  fi
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    info "正在以 sudo 重新执行……"
    exec sudo -E bash "$0" "$@"
  fi
}

# 运行命令（便于统一处理）
run(){
  local -a cmd=("$@")
  "${cmd[@]}"
}

# 读取已存在的非空行
read_existing_lines(){
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 0
  fi
  # shellcheck disable=SC2002
  cat "$path" | awk 'NF{print $0}'
}

# 写入行（文件末尾确保换行）
write_lines(){
  local path="$1"; shift
  local tmp
  tmp="$(mktemp)"
  # 逐行写入
  for l in "$@"; do
    printf "%s\n" "$l" >>"$tmp"
  done
  # 末尾再加一行换行（确保有换行）
  printf "\n" >>"$tmp"
  install -m 0644 "$tmp" "$path"
  rm -f "$tmp"
}

# 合并去重（保持原有顺序，新增项追加在后）
merge_unique(){
  # 参数：existing_lines...  -- 分隔符 --  new_dirs...
  local sep="--SEP--"
  local seen_file
  seen_file="$(mktemp)"
  awk 'BEGIN{}' /dev/null >"$seen_file"

  local -a out=()
  local mode="existing"
  for token in "$@"; do
    if [[ "$token" == "$sep" ]]; then
      mode="new"
      continue
    fi
    if [[ "$mode" == "existing" ]]; then
      if [[ -n "$token" ]]; then
        if ! grep -Fxq "$token" "$seen_file"; then
          echo "$token" >>"$seen_file"
          out+=("$token")
        fi
      fi
    else
      if [[ -n "$token" ]]; then
        if ! grep -Fxq "$token" "$seen_file"; then
          echo "$token" >>"$seen_file"
          out+=("$token")
        fi
      fi
    fi
  done
  rm -f "$seen_file"
  printf '%s\n' "${out[@]}"
}

# 解析真实路径（尽量用 realpath，不在则退回 readlink -f，再不行就原样）
realpath_safe(){
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -f "$p" || echo "$p"
  else
    echo "$p"
  fi
}

# 校验目录并提示是否包含 TRT 库
validate_dirs(){
  local -n _in="$1"
  local -n _out="$2"
  local has_trt="false"
  _out=()
  for d in "${_in[@]}"; do
    local rp
    rp="$(realpath_safe "$d")"
    if [[ ! -d "$rp" ]]; then
      err "目录不存在：$rp"
      exit 2
    fi
    # 检查 libnvinfer*.so*
    shopt -s nullglob
    local matches=("$rp"/libnvinfer*.so*)
    shopt -u nullglob
    if (( ${#matches[@]} > 0 )); then
      has_trt="true"
    fi
    _out+=("$rp")
  done
  if [[ "$has_trt" != "true" ]]; then
    warn "提供的目录中没有检测到 libnvinfer*.so*。\n若你期望注册 TensorRT，请确认传入正确的库目录（如 /opt/tensorrt/lib）。"
  fi
}

backup_conf_if_exists(){
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 0
  fi
  local ts
  ts="$(date +%Y%m%d%H%M%S)"
  local backup="${path}.bak.${ts}"
  cp -a "$path" "$backup"
  info "已创建备份：$backup"
}

print_plan(){
  local conf="$1"; shift
  info "将确保以下目录被注册到 ${conf}:"
  for l in "$@"; do
    printf "  - %s\n" "$l"
  done
}

do_ldconfig(){
  info "正在运行 ldconfig"
  run ldconfig
}

verify(){
  echo
  info "验证："
  echo "  - ldconfig -p | grep nvinfer || true"
  bash -lc 'ldconfig -p | grep -i nvinfer || true'
}

remove_conf(){
  if "$DRY_RUN"; then
    info "[dry-run] 将会删除 ${CONF}（若存在）并执行 ldconfig"
    return 0
  fi
  if [[ -f "$CONF" ]]; then
    info "正在删除 ${CONF}"
    rm -f -- "$CONF"
    do_ldconfig
  else
    info "${CONF} 不存在，无需删除。"
  fi
}

main(){
  if "$DO_REMOVE"; then
    # 仅 --remove 时允许没有库目录
    require_root_or_reexec "$@"
    remove_conf
    exit 0
  fi

  if [[ ${#LIBDIRS[@]} -eq 0 ]]; then
    err "未提供库目录。"
    echo
    print_usage
    exit 1
  fi

  # 校验输入目录并规范化为绝对路径
  local -a libdirs_norm=()
  validate_dirs LIBDIRS libdirs_norm

  # 合并已有配置与新目录（幂等去重）
  mapfile -t existing < <(read_existing_lines "$CONF" || true)
  local sep="--SEP--"
  mapfile -t merged < <(merge_unique "${existing[@]}" "$sep" "${libdirs_norm[@]}")

  print_plan "$CONF" "${merged[@]}"

  if "$DRY_RUN"; then
    echo
    info "[dry-run] 将写入以上条目并执行：ldconfig"
    exit 0
  fi

  # 需要 root
  require_root_or_reexec "$@"

  # 备份并写入
  backup_conf_if_exists "$CONF"
  write_lines "$CONF" "${merged[@]}"
  info "已写入 $((${#merged[@]})) 条到 ${CONF}"

  # 刷新缓存并验证
  do_ldconfig
  verify
}

main "$@"
