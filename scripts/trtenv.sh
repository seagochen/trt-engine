#!/bin/bash
#
# TrtEnv - TensorRT Environment Setup Tool
#
# Comprehensive environment configuration for TensorRT and CUDA:
#   1. Configure shell environment variables (PATH, LD_LIBRARY_PATH)
#   2. Configure system library search paths (ldconfig)
#   3. Install build dependencies (libeigen3-dev, etc.)
#   4. Install performance monitoring tools (jtop/nvitop)
#
# Usage:
#   ./trtenv.sh [OPTIONS]
#
# Options:
#   --setup-all           Configure everything (recommended)
#   --configure-env       Configure shell environment variables only
#   --ldconfig            Configure system ldconfig only
#   --install-deps        Install build dependencies (libeigen3-dev, etc.)
#   --install-monitor     Install performance monitoring tools only
#   --show                Show current configuration
#   --remove              Remove ldconfig configuration
#   --dry-run             Preview changes without modifying files
#   --bashrc PATH         Specify shell rc file (default: ~/.bashrc)
#   --lib-dirs DIR...     Specify library directories for ldconfig
#   --help, -h            Show this help message
#
# Author: TrtEngineToolkits
# Date: 2025-11-28
#

# ----------------------------
# Color codes and utilities
# ----------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${RESET} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${RESET} $1" >&2
}

print_header() {
    echo -e "${BLUE}============================================================${RESET}"
    echo -e "${BLUE}$1${RESET}"
    echo -e "${BLUE}============================================================${RESET}"
}

# ----------------------------
# Platform detection
# ----------------------------
is_jetson() {
    if [[ -f "/etc/nv_tegra_release" ]]; then
        return 0
    fi
    if [[ -f "/proc/device-tree/compatible" ]]; then
        if grep -qi "tegra" /proc/device-tree/compatible 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

is_root() {
    [[ $EUID -eq 0 ]]
}

# ----------------------------
# CUDA Detection
# ----------------------------
detect_cuda() {
    local cuda_root=""
    local cuda_lib=""

    # 1) Standard installation
    if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        cuda_root="/usr/local/cuda"
    fi

    # 2) Version directories (pick highest version)
    if [[ -z "$cuda_root" ]]; then
        for dir in $(ls -d /usr/local/cuda-* 2>/dev/null | sort -rV); do
            if [[ -x "$dir/bin/nvcc" ]]; then
                cuda_root="$dir"
                break
            fi
        done
    fi

    # 3) nvcc in PATH
    if [[ -z "$cuda_root" ]]; then
        local nvcc_path
        nvcc_path=$(command -v nvcc 2>/dev/null)
        if [[ -n "$nvcc_path" ]]; then
            cuda_root=$(dirname "$(dirname "$nvcc_path")")
        fi
    fi

    if [[ -z "$cuda_root" ]]; then
        return 1
    fi

    # Determine lib directory
    if [[ -d "$cuda_root/lib64" ]]; then
        cuda_lib="$cuda_root/lib64"
    elif [[ -d "$cuda_root/lib" ]]; then
        cuda_lib="$cuda_root/lib"
    fi

    echo "$cuda_root:$cuda_lib"
    return 0
}

# ----------------------------
# TensorRT Detection
# ----------------------------
_has_trt_libs() {
    local dir="$1"
    [[ -d "$dir" ]] && ls "$dir"/libnvinfer* &>/dev/null
}

_find_trt_lib_dir() {
    local root="$1"
    for sub in "lib" "lib64" "targets/x86_64-linux-gnu/lib" "targets/aarch64-linux-gnu/lib"; do
        local p="$root/$sub"
        if _has_trt_libs "$p"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

detect_tensorrt() {
    local trt_root=""
    local trt_lib=""

    # Jetson platform
    if is_jetson; then
        if [[ -x "/usr/src/tensorrt/bin/trtexec" ]] || _has_trt_libs "/usr/src/tensorrt/lib"; then
            trt_root="/usr/src/tensorrt"
            trt_lib=$(_find_trt_lib_dir "$trt_root" 2>/dev/null || echo "")
        fi
    fi

    # x86 common directories
    if [[ -z "$trt_root" ]]; then
        for dir in "/opt/tensorrt" $(ls -d /usr/local/TensorRT* 2>/dev/null | sort -rV); do
            if [[ -x "$dir/bin/trtexec" ]] || _find_trt_lib_dir "$dir" &>/dev/null; then
                trt_root="$dir"
                trt_lib=$(_find_trt_lib_dir "$trt_root" 2>/dev/null || echo "")
                break
            fi
        done
    fi

    # System library fallback
    if [[ -z "$trt_root" ]]; then
        for lc in "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu" \
                  "/usr/lib/x86_64-linux-gnu/tensorrt" "/usr/lib/aarch64-linux-gnu/tensorrt"; do
            if _has_trt_libs "$lc"; then
                if [[ "$(basename "$lc")" == "tensorrt" ]]; then
                    trt_root="$(dirname "$lc")"
                else
                    trt_root="$lc"
                fi
                trt_lib="$lc"
                break
            fi
        done
    fi

    if [[ -z "$trt_root" ]]; then
        return 1
    fi

    echo "$trt_root:$trt_lib"
    return 0
}

# ----------------------------
# Module 1: Shell Environment Configuration
# ----------------------------
inject_block() {
    local bashrc_path="$1"
    local marker="$2"
    shift 2
    local lines=("$@")

    local begin="# >>> $marker"
    local end="# <<< $marker"

    # Build block content
    local block="$begin"$'\n'
    for line in "${lines[@]}"; do
        block+="$line"$'\n'
    done
    block+="$end"

    if [[ ! -f "$bashrc_path" ]]; then
        touch "$bashrc_path"
    fi

    local content
    content=$(cat "$bashrc_path")

    if grep -qF "$begin" "$bashrc_path" && grep -qF "$end" "$bashrc_path"; then
        # Replace existing block using perl for multiline replacement
        perl -i -0pe "s/\Q$begin\E.*?\Q$end\E\n?/$block\n/s" "$bashrc_path"
        if [[ "$DRY_RUN" == "true" ]]; then
            print_info "[dry-run] 将在 $bashrc_path 中更新标记块：$marker"
        else
            print_success "$bashrc_path 已更新标记块：$marker"
        fi
    else
        # Append new block
        if [[ "$DRY_RUN" == "true" ]]; then
            print_info "[dry-run] 将在 $bashrc_path 中追加标记块：$marker"
        else
            echo "" >> "$bashrc_path"
            echo "$block" >> "$bashrc_path"
            print_success "$bashrc_path 已追加标记块：$marker"
        fi
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "$block"
    fi
}

configure_shell_env() {
    local bashrc_path="${BASHRC_PATH:-$HOME/.bashrc}"
    bashrc_path="${bashrc_path/#\~/$HOME}"

    print_header "模块 1: 配置 Shell 环境变量"

    local cuda_info trt_info
    local cuda_root cuda_lib trt_root trt_lib
    local configured=false

    # Detect CUDA
    if cuda_info=$(detect_cuda); then
        IFS=':' read -r cuda_root cuda_lib <<< "$cuda_info"
        print_info "检测到 CUDA: $cuda_root"

        local cuda_marker="CUDA Toolkit Environment Variables (trtenv)"
        local cuda_lines=()
        [[ -d "$cuda_root/bin" ]] && cuda_lines+=("export PATH=\"\$PATH:$cuda_root/bin\"")
        [[ -n "$cuda_lib" ]] && cuda_lines+=("export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:+\${LD_LIBRARY_PATH}:}$cuda_lib\"")

        if [[ "$DRY_RUN" != "true" ]]; then
            inject_block "$bashrc_path" "$cuda_marker" "${cuda_lines[@]}"
        else
            inject_block "$bashrc_path" "$cuda_marker" "${cuda_lines[@]}"
        fi
        configured=true
    else
        print_warn "未检测到 CUDA，跳过 CUDA 环境变量配置"
    fi

    # Detect TensorRT
    if trt_info=$(detect_tensorrt); then
        IFS=':' read -r trt_root trt_lib <<< "$trt_info"

        local platform="x86"
        is_jetson && platform="Jetson"

        local trt_marker="TensorRT Environment Variables (trtenv, $platform)"
        local trt_lines=()

        if [[ -d "$trt_root/bin" ]]; then
            trt_lines+=("export PATH=\"\$PATH:$trt_root/bin\"")
            if [[ -x "$trt_root/bin/trtexec" ]]; then
                print_info "检测到 TensorRT: $trt_root (trtexec: $trt_root/bin/trtexec)"
            else
                print_info "检测到 TensorRT: $trt_root"
            fi
        else
            print_info "检测到 TensorRT 库: ${trt_lib:-$trt_root}"
        fi

        [[ -n "$trt_lib" ]] && trt_lines+=("export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH:+\${LD_LIBRARY_PATH}:}$trt_lib\"")

        if [[ ${#trt_lines[@]} -gt 0 ]]; then
            inject_block "$bashrc_path" "$trt_marker" "${trt_lines[@]}"
        fi
        configured=true
    else
        print_warn "未检测到 TensorRT，跳过 TensorRT 环境变量配置"
    fi

    if [[ "$DRY_RUN" != "true" ]] && [[ "$configured" == "true" ]]; then
        echo ""
        print_info "配置已写入: $bashrc_path"
        print_info "执行 'source $bashrc_path' 或重新打开终端以使更改生效"
        return 0
    elif [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        print_info "[dry-run] 未对文件做任何修改"
        return 0
    else
        print_warn "未检测到 CUDA 或 TensorRT，无配置写入"
        return 1
    fi
}

# ----------------------------
# Module 2: Ldconfig Configuration
# ----------------------------
configure_ldconfig() {
    local remove="${1:-false}"
    local CONF_FILE="/etc/ld.so.conf.d/tensorrt.conf"

    print_header "模块 2: 配置系统库搜索路径 (ldconfig)"

    if [[ "$remove" == "true" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            print_info "[dry-run] 将删除 $CONF_FILE 并运行 ldconfig"
            return 0
        fi

        if ! is_root; then
            print_error "需要 root 权限才能修改 ldconfig 配置"
            return 1
        fi

        if [[ -f "$CONF_FILE" ]]; then
            print_info "正在删除 $CONF_FILE"
            rm -f "$CONF_FILE"
            ldconfig
            print_success "已删除 ldconfig 配置"
        else
            print_info "$CONF_FILE 不存在，无需删除"
        fi
        return 0
    fi

    # Auto-detect library directories if not provided
    local lib_dirs=("${LIB_DIRS[@]}")

    if [[ ${#lib_dirs[@]} -eq 0 ]]; then
        local trt_info cuda_info

        if trt_info=$(detect_tensorrt); then
            IFS=':' read -r _ trt_lib <<< "$trt_info"
            [[ -n "$trt_lib" ]] && lib_dirs+=("$trt_lib")
        fi

        if cuda_info=$(detect_cuda); then
            IFS=':' read -r _ cuda_lib <<< "$cuda_info"
            [[ -n "$cuda_lib" ]] && lib_dirs+=("$cuda_lib")
        fi
    fi

    if [[ ${#lib_dirs[@]} -eq 0 ]]; then
        print_warn "未提供库目录且未自动检测到 CUDA/TensorRT 库"
        return 1
    fi

    # Validate directories
    local validated_dirs=()
    local has_trt=false

    for d in "${lib_dirs[@]}"; do
        if [[ ! -d "$d" ]]; then
            print_error "目录不存在: $d"
            return 1
        fi

        if ls "$d"/libnvinfer*.so* &>/dev/null; then
            has_trt=true
        fi

        validated_dirs+=("$(realpath "$d")")
    done

    [[ "$has_trt" == "false" ]] && print_warn "提供的目录中未检测到 libnvinfer*.so* 文件"

    # Read existing configuration and merge
    local existing_lines=()
    if [[ -f "$CONF_FILE" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && existing_lines+=("$line")
        done < "$CONF_FILE"
    fi

    # Merge and deduplicate
    local -A seen
    local unique_lines=()

    for line in "${existing_lines[@]}" "${validated_dirs[@]}"; do
        if [[ -z "${seen[$line]:-}" ]]; then
            unique_lines+=("$line")
            seen[$line]=1
        fi
    done

    print_info "将注册以下库目录到 ldconfig:"
    for line in "${unique_lines[@]}"; do
        echo "  - $line"
    done

    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        print_info "[dry-run] 将写入以上条目并执行 ldconfig"
        return 0
    fi

    if ! is_root; then
        print_error "需要 root 权限才能修改 ldconfig 配置"
        print_info "请使用 sudo 运行此脚本"
        return 1
    fi

    # Backup existing config
    if [[ -f "$CONF_FILE" ]]; then
        local backup="$CONF_FILE.bak.$(date +%s)"
        mv "$CONF_FILE" "$backup"
        print_info "已创建备份: $backup"
    fi

    # Write new configuration
    printf '%s\n' "${unique_lines[@]}" > "$CONF_FILE"
    chmod 644 "$CONF_FILE"
    print_success "已写入 ${#unique_lines[@]} 条记录到 $CONF_FILE"

    # Run ldconfig
    print_info "正在运行 ldconfig..."
    ldconfig
    print_success "ldconfig 配置完成"

    # Verify
    echo ""
    print_info "验证配置:"
    if ldconfig -p | grep -i nvinfer &>/dev/null; then
        ldconfig -p | grep -i nvinfer | while read -r line; do
            echo "  $line"
        done
    else
        print_warn "  未在 ldconfig 缓存中找到 nvinfer 库"
    fi

    return 0
}

# ----------------------------
# Module 3: Install Build Dependencies
# ----------------------------
install_build_dependencies() {
    print_header "模块 3: 安装构建依赖"

    local apt_packages=(
        "libeigen3-dev"    # Eigen3 linear algebra library
    )

    print_info "将安装以下依赖包:"
    for pkg in "${apt_packages[@]}"; do
        echo "  - $pkg"
    done

    if [[ "$DRY_RUN" == "true" ]]; then
        echo ""
        print_info "[dry-run] 将执行: sudo apt install -y ${apt_packages[*]}"
        return 0
    fi

    local sudo_prefix=""
    if ! is_root; then
        print_warn "需要 root 权限安装依赖包，将使用 sudo"
        sudo_prefix="sudo"
    fi

    # Update apt cache
    print_info "正在更新 apt 缓存..."
    $sudo_prefix apt update || true

    # Install packages
    print_info "正在安装依赖包..."
    if $sudo_prefix apt install -y "${apt_packages[@]}"; then
        print_success "构建依赖安装完成"
        return 0
    else
        print_error "安装失败"
        return 1
    fi
}

# ----------------------------
# Module 4: Install Monitoring Tools
# ----------------------------
install_monitoring_tool() {
    print_header "模块 4: 安装性能监控工具"

    local tool cmd
    local need_sudo=false

    if is_jetson; then
        print_info "检测到 Jetson 平台，将安装 jetson-stats (jtop)"
        tool="jetson-stats"
        cmd="pip3 install -U jetson-stats"
        need_sudo=true
    else
        print_info "检测到 x86 平台，将安装 nvitop"
        tool="nvitop"
        cmd="pip3 install -U nvitop"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[dry-run] 将执行: $cmd"
        return 0
    fi

    local sudo_prefix=""
    if [[ "$need_sudo" == "true" ]] && ! is_root; then
        sudo_prefix="sudo"
    fi

    print_info "正在安装 $tool..."
    if $sudo_prefix $cmd; then
        print_success "$tool 安装完成"

        if is_jetson; then
            print_info "运行 'jtop' 命令来启动监控工具"
        else
            print_info "运行 'nvitop' 命令来启动监控工具"
        fi
        return 0
    else
        print_error "安装失败"
        return 1
    fi
}

# ----------------------------
# Module 5: Show Configuration
# ----------------------------
show_configuration() {
    print_header "当前 TensorRT 环境配置"

    # Platform
    echo ""
    if is_jetson; then
        printf "%-20s %s\n" "平台:" "Jetson"
    else
        printf "%-20s %s\n" "平台:" "x86_64"
    fi

    # CUDA
    local cuda_info
    if cuda_info=$(detect_cuda); then
        IFS=':' read -r cuda_root cuda_lib <<< "$cuda_info"
        printf "%-20s %s\n" "CUDA 根目录:" "$cuda_root"
        printf "%-20s %s\n" "CUDA 库目录:" "$cuda_lib"

        if command -v nvcc &>/dev/null; then
            local version
            version=$(nvcc --version 2>/dev/null | grep -i release || true)
            [[ -n "$version" ]] && printf "%-20s %s\n" "CUDA 版本:" "$version"
        fi
    else
        printf "%-20s %s\n" "CUDA:" "未检测到"
    fi

    # TensorRT
    local trt_info
    if trt_info=$(detect_tensorrt); then
        IFS=':' read -r trt_root trt_lib <<< "$trt_info"
        printf "%-20s %s\n" "TensorRT 根目录:" "$trt_root"
        printf "%-20s %s\n" "TensorRT 库目录:" "${trt_lib:-(系统库)}"

        if [[ -x "$trt_root/bin/trtexec" ]]; then
            printf "%-20s %s\n" "trtexec:" "$trt_root/bin/trtexec"
        fi
    else
        printf "%-20s %s\n" "TensorRT:" "未检测到"
    fi

    # Ldconfig
    echo ""
    printf "%-20s\n" "Ldconfig 配置:"
    local conf_file="/etc/ld.so.conf.d/tensorrt.conf"
    if [[ -f "$conf_file" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && echo "  - $line"
        done < "$conf_file"
    else
        echo "  未配置"
    fi

    # Monitoring tools
    echo ""
    printf "%-20s\n" "监控工具:"
    if is_jetson; then
        if command -v jtop &>/dev/null; then
            echo "  jtop: 已安装"
        else
            echo "  jtop: 未安装"
        fi
    else
        if command -v nvitop &>/dev/null; then
            echo "  nvitop: 已安装"
        else
            echo "  nvitop: 未安装"
        fi
    fi

    # Shell environment
    echo ""
    printf "%-20s\n" "Shell 环境变量:"
    local bashrc="$HOME/.bashrc"
    if [[ -f "$bashrc" ]]; then
        if grep -q "trtenv" "$bashrc" 2>/dev/null; then
            echo "  $bashrc: 已配置"
        else
            echo "  $bashrc: 未配置"
        fi
    else
        echo "  $bashrc: 文件不存在"
    fi
}

# ----------------------------
# Help
# ----------------------------
show_usage() {
    echo -e "${GREEN}TrtEnv - TensorRT Environment Setup Tool${RESET}"
    echo ""
    echo -e "${BLUE}Description:${RESET}"
    echo "  Comprehensive environment configuration for TensorRT and CUDA:"
    echo "    1. Configure shell environment variables (PATH, LD_LIBRARY_PATH)"
    echo "    2. Configure system library search paths (ldconfig)"
    echo "    3. Install build dependencies (libeigen3-dev, etc.)"
    echo "    4. Install performance monitoring tools (jtop/nvitop)"
    echo ""
    echo -e "${BLUE}Usage:${RESET}"
    echo "  $0 [OPTIONS]"
    echo ""
    echo -e "${BLUE}Options:${RESET}"
    echo -e "  ${CYAN}Main Actions:${RESET}"
    echo "    --setup-all           Configure everything (recommended)"
    echo "    --configure-env       Configure shell environment variables only"
    echo "    --ldconfig            Configure system ldconfig only"
    echo "    --install-deps        Install build dependencies (libeigen3-dev, etc.)"
    echo "    --install-monitor     Install performance monitoring tools only"
    echo "    --show                Show current configuration"
    echo "    --remove              Remove ldconfig configuration"
    echo ""
    echo -e "  ${CYAN}Modifiers:${RESET}"
    echo "    --dry-run             Preview changes without modifying files"
    echo "    --bashrc PATH         Specify shell rc file (default: ~/.bashrc)"
    echo "    --lib-dirs DIR...     Specify library directories for ldconfig"
    echo ""
    echo -e "  ${CYAN}Help:${RESET}"
    echo "    --help, -h            Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${RESET}"
    echo -e "  ${GREEN}Setup everything (recommended):${RESET}"
    echo "    sudo $0 --setup-all"
    echo ""
    echo -e "  ${GREEN}Preview changes:${RESET}"
    echo "    $0 --setup-all --dry-run"
    echo ""
    echo -e "  ${GREEN}Configure shell environment only:${RESET}"
    echo "    $0 --configure-env"
    echo ""
    echo -e "  ${GREEN}Configure ldconfig only:${RESET}"
    echo "    sudo $0 --ldconfig"
    echo ""
    echo -e "  ${GREEN}Install build dependencies:${RESET}"
    echo "    sudo $0 --install-deps"
    echo ""
    echo -e "  ${GREEN}Install monitoring tools:${RESET}"
    echo "    $0 --install-monitor"
    echo ""
    echo -e "  ${GREEN}Show current configuration:${RESET}"
    echo "    $0 --show"
    echo ""
    echo -e "  ${GREEN}Remove ldconfig configuration:${RESET}"
    echo "    sudo $0 --remove"
    echo ""
    echo -e "${BLUE}Notes:${RESET}"
    echo "  - Some operations require root privileges (sudo)"
    echo "  - --setup-all is the recommended way to configure everything"
    echo "  - Use --dry-run to preview changes before applying them"
    echo "  - After configuration, run 'source ~/.bashrc' or reopen terminal"
}

# ----------------------------
# Main
# ----------------------------
main() {
    # Check Linux
    if [[ "$(uname -s)" != "Linux" ]]; then
        print_error "This script only works on Linux systems"
        exit 1
    fi

    # Parse arguments
    local ACTION=""
    DRY_RUN="false"
    BASHRC_PATH="$HOME/.bashrc"
    LIB_DIRS=()

    local do_configure_env=false
    local do_ldconfig=false
    local do_install_deps=false
    local do_install_monitor=false
    local do_show=false
    local do_remove=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --setup-all)
                do_configure_env=true
                do_ldconfig=true
                do_install_deps=true
                do_install_monitor=true
                shift
                ;;
            --configure-env)
                do_configure_env=true
                shift
                ;;
            --ldconfig)
                do_ldconfig=true
                shift
                ;;
            --install-deps)
                do_install_deps=true
                shift
                ;;
            --install-monitor)
                do_install_monitor=true
                shift
                ;;
            --show)
                do_show=true
                shift
                ;;
            --remove)
                do_remove=true
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --bashrc)
                BASHRC_PATH="$2"
                shift 2
                ;;
            --lib-dirs)
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    LIB_DIRS+=("$1")
                    shift
                done
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # If no action specified, show help
    if ! $do_configure_env && ! $do_ldconfig && ! $do_install_deps && \
       ! $do_install_monitor && ! $do_show && ! $do_remove; then
        show_usage
        exit 0
    fi

    print_info "TrtEnv - TensorRT Environment Setup Tool"
    echo ""

    # Show configuration
    if $do_show; then
        show_configuration
        exit 0
    fi

    # Remove configuration
    if $do_remove; then
        configure_ldconfig "true"
        exit $?
    fi

    # Check if root is needed
    local needs_root=false
    if $do_ldconfig || $do_install_deps; then
        needs_root=true
    fi

    if $needs_root && ! is_root && [[ "$DRY_RUN" != "true" ]]; then
        print_warn "某些操作需要 root 权限"
        print_info "请使用 sudo 运行: sudo $0 $*"
    fi

    # Execute modules
    local success_count=0
    local total_count=0

    if $do_configure_env; then
        ((total_count++))
        if configure_shell_env; then
            ((success_count++))
        fi
        echo ""
    fi

    if $do_ldconfig; then
        ((total_count++))
        if configure_ldconfig; then
            ((success_count++))
        fi
        echo ""
    fi

    if $do_install_deps; then
        ((total_count++))
        if install_build_dependencies; then
            ((success_count++))
        fi
        echo ""
    fi

    if $do_install_monitor; then
        ((total_count++))
        if install_monitoring_tool; then
            ((success_count++))
        fi
        echo ""
    fi

    # Summary
    if [[ "$DRY_RUN" != "true" ]]; then
        print_header "完成: $success_count/$total_count 个模块配置成功"
    fi
}

main "$@"
