#!/bin/bash
#
# Jetson Boost - Performance & Fan Control Tool
#
# Comprehensive performance tuning for NVIDIA Jetson platforms:
#   1. Power mode control (nvpmodel)
#   2. Clock/CUDA core control (jetson_clocks)
#   3. Fan speed control (PWM)
#   4. Auto-start at boot (systemd service)
#
# Usage:
#   sudo ./jetson_boost.sh [OPTION]
#
# Options:
#   --interactive, -i     Interactive menu (default)
#   --maxn                Set to MAXN mode (maximum performance)
#   --10w                 Set to 10W power saving mode
#   --max-clocks          Maximize clocks (CUDA cores)
#   --restore             Restore to saved clock state
#   --fan-on              Force fan to maximum speed
#   --fan-off             Restore fan to auto mode
#   --fan-status          Show current fan status
#   --enable-autostart    Enable fan auto-start at boot
#   --disable-autostart   Disable fan auto-start at boot
#   --show                Show current clock status
#   --help, -h            Show this help message
#
# Author: TrtEngineToolkits
# Date: 2025-11-28
#

# ----------------------------
# Configuration
# ----------------------------
CONFIG_FILE="/root/.jetsonclocks_conf.txt"
BACKUP_FILE="/root/.jetsonclocks_conf.backup.txt"
FAN_SERVICE_FILE="/etc/systemd/system/jetson-boost-fan.service"
FAN_PWM_PATH="/sys/devices/pwm-fan/target_pwm"
FAN_HWMON_PATHS=("/sys/devices/pwm-fan/hwmon" "/sys/class/hwmon")

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
    echo -e "${CYAN}============================================================${RESET}"
    echo -e "${CYAN}$1${RESET}"
    echo -e "${CYAN}============================================================${RESET}"
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

require_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "このスクリプトを実行するには root 権限が必要です。"
        print_info "sudo $0 を使用してください。"
        exit 1
    fi
}

require_jetson() {
    if ! is_jetson; then
        print_error "このスクリプトは NVIDIA Jetson 専用です。終了します。"
        exit 1
    fi
}

cmd_exists() {
    command -v "$1" &>/dev/null
}

# ----------------------------
# nvpmodel Control
# ----------------------------
get_nvpmodel_mode() {
    if ! cmd_exists nvpmodel; then
        echo "不明"
        return
    fi
    local output
    output=$(nvpmodel -q 2>/dev/null)
    echo "$output" | grep -i "Mode" | awk '{print $NF}'
}

set_nvpmodel() {
    local mode="$1"
    print_info "nvpmodel モードを $mode に切り替えます…"
    if nvpmodel -m "$mode"; then
        print_success "電源モードを $mode に設定しました。"
        return 0
    else
        print_error "モード切替に失敗しました。"
        return 1
    fi
}

# ----------------------------
# Fan Control
# ----------------------------
find_fan_pwm_path() {
    local pwm_path=""
    local enable_path=""

    # Method 1: Direct path
    if [[ -f "$FAN_PWM_PATH" ]]; then
        pwm_path="$FAN_PWM_PATH"
        local parent
        parent=$(dirname "$FAN_PWM_PATH")
        for enable_file in "pwm1_enable" "pwm_enable" "temp_control"; do
            if [[ -f "$parent/$enable_file" ]]; then
                enable_path="$parent/$enable_file"
                break
            fi
        done
        echo "$pwm_path:$enable_path"
        return 0
    fi

    # Method 2: Search through hwmon
    for base_path in "${FAN_HWMON_PATHS[@]}"; do
        if [[ ! -d "$base_path" ]]; then
            continue
        fi
        for hwmon_dir in "$base_path"/hwmon*; do
            if [[ -f "$hwmon_dir/pwm1" ]]; then
                pwm_path="$hwmon_dir/pwm1"
                for enable_file in "pwm1_enable" "pwm_enable" "temp_control"; do
                    if [[ -f "$hwmon_dir/$enable_file" ]]; then
                        enable_path="$hwmon_dir/$enable_file"
                        break
                    fi
                done
                echo "$pwm_path:$enable_path"
                return 0
            fi
        done
    done

    return 1
}

set_fan_speed() {
    local speed="$1"
    local manual_mode="${2:-true}"

    local fan_info
    if ! fan_info=$(find_fan_pwm_path); then
        print_error "ファン制御パスが見つかりません。"
        print_info "このデバイスはファン制御をサポートしていない可能性があります。"
        return 1
    fi

    IFS=':' read -r pwm_path enable_path <<< "$fan_info"

    # Set to manual mode if enable path exists
    if [[ "$manual_mode" == "true" ]] && [[ -n "$enable_path" ]]; then
        if echo "1" > "$enable_path" 2>/dev/null; then
            print_info "ファン制御を手動モードに設定しました: $enable_path"
        else
            print_warn "手動モード設定に失敗しました（一部のデバイスでは不要な場合があります）"
        fi
    fi

    # Set PWM speed
    if echo "$speed" > "$pwm_path" 2>/dev/null; then
        print_success "ファン速度を $speed/255 に設定しました: $pwm_path"
        return 0
    else
        print_error "ファン速度の設定に失敗しました"
        return 1
    fi
}

fan_force_on() {
    print_header "ファンを最大速度で強制起動"
    if set_fan_speed 255; then
        print_success "ファンの強制起動に成功しました。"
    else
        print_info "代替方法を試します: jetson_clocks を使用..."
        if cmd_exists jetson_clocks && jetson_clocks --fan 2>/dev/null; then
            print_success "jetson_clocks を使用してファンを起動しました。"
        else
            print_error "ファンの起動に失敗しました。"
            return 1
        fi
    fi
}

fan_force_off() {
    print_header "ファンを自動制御モードに戻す"

    local fan_info
    if ! fan_info=$(find_fan_pwm_path); then
        print_warn "ファン制御パスが見つかりません。"
        return 1
    fi

    IFS=':' read -r pwm_path enable_path <<< "$fan_info"

    # Try to set auto mode via enable file
    if [[ -n "$enable_path" ]]; then
        # Try mode 2 (temperature control) first
        if echo "2" > "$enable_path" 2>/dev/null; then
            print_info "ファン制御を自動モード(温度制御)に設定しました: $enable_path"
        else
            # Try mode 0 (auto)
            echo "0" > "$enable_path" 2>/dev/null && \
                print_info "代替方法で自動モードに設定しました。"
        fi
    fi

    # Set PWM to 0
    echo "0" > "$pwm_path" 2>/dev/null
    print_success "ファンを自動制御モードに戻しました。"
}

fan_show_status() {
    print_header "現在のファン状態"

    local fan_info
    if ! fan_info=$(find_fan_pwm_path); then
        print_warn "ファン制御パスが見つかりません。"
        return 1
    fi

    IFS=':' read -r pwm_path enable_path <<< "$fan_info"

    echo "  PWMパス: $pwm_path"

    if [[ -f "$pwm_path" ]]; then
        local pwm_val
        pwm_val=$(cat "$pwm_path" 2>/dev/null)
        echo "  PWM値: $pwm_val/255"
        local percentage=$((pwm_val * 100 / 255))
        echo "  速度: ${percentage}%"
    fi

    if [[ -n "$enable_path" ]] && [[ -f "$enable_path" ]]; then
        local enable_val
        enable_val=$(cat "$enable_path" 2>/dev/null)
        local mode
        case "$enable_val" in
            0) mode="無効または自動" ;;
            1) mode="手動制御" ;;
            2) mode="自動（温度制御）" ;;
            3) mode="完全速度" ;;
            *) mode="不明 ($enable_val)" ;;
        esac
        echo "  制御モード: $mode"
        echo "  制御パス: $enable_path"
    else
        echo "  制御モード: enable ファイルが見つかりません"
    fi
}

# ----------------------------
# jetson_clocks Control
# ----------------------------
setup_jetson_clocks_config() {
    print_info "クロック設定ファイルの確認とセットアップ中..."

    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_info "クロック設定ファイル ($CONFIG_FILE) が存在しません。新しく作成します…"
        if jetson_clocks --store 2>/dev/null; then
            print_success "設定ファイルを作成しました: $CONFIG_FILE"
        else
            print_warn "設定ファイルの作成に失敗しました。"
        fi
    fi

    # Create initial backup
    if [[ -f "$CONFIG_FILE" ]] && [[ ! -f "$BACKUP_FILE" ]]; then
        print_info "初回バックアップ ($BACKUP_FILE) を作成中…"
        if cp "$CONFIG_FILE" "$BACKUP_FILE"; then
            chmod 444 "$BACKUP_FILE"
            print_success "バックアップを作成しました: $BACKUP_FILE"
        fi
    fi
}

jetson_clocks_max() {
    print_header "全CUDAコアを有効化＆クロック最大化"

    if ! cmd_exists jetson_clocks; then
        print_error "jetson_clocks が見つかりません。"
        print_info "JetPack 付属のユーティリティです。SDK Manager でインストールしてください。"
        return 1
    fi

    setup_jetson_clocks_config

    print_info "全CUDAコアを有効化し、クロックを最大化します…"
    if jetson_clocks; then
        print_success "CUDAコアとクロックの最大化に成功しました。"
        jetson_clocks_show
    else
        print_error "クロックの最大化に失敗しました。"
        return 1
    fi
}

jetson_clocks_restore() {
    print_header "クロック設定を復元"

    if ! cmd_exists jetson_clocks; then
        print_error "jetson_clocks が見つかりません。"
        return 1
    fi

    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "復元するための設定ファイル ($CONFIG_FILE) が見つかりません。"
        print_info "先に「クロック最大化」を実行するか、手動で 'sudo jetson_clocks --store' を実行してください。"
        return 1
    fi

    print_info "クロックを保存された状態に戻します…"
    if jetson_clocks --restore; then
        print_success "クロックの復元に成功しました。"
    else
        print_error "復元に失敗しました。"
        return 1
    fi
}

jetson_clocks_show() {
    echo ""
    print_info "現在のクロック状態:"
    jetson_clocks --show 2>/dev/null || print_error "クロック状態の表示に失敗しました。"
}

# ----------------------------
# Systemd Service for Fan Auto-Start
# ----------------------------
create_fan_service() {
    local script_path
    script_path=$(readlink -f "$0")

    cat > "$FAN_SERVICE_FILE" << EOF
[Unit]
Description=Jetson Boost - Fan Auto-Start at Boot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=$script_path --fan-on
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    if [[ -f "$FAN_SERVICE_FILE" ]]; then
        print_success "systemd サービスファイルを作成しました: $FAN_SERVICE_FILE"
        return 0
    else
        print_error "サービスファイルの作成に失敗しました"
        return 1
    fi
}

enable_fan_autostart() {
    print_header "システム起動時のファン自動起動を有効化"

    if [[ ! -f "$FAN_SERVICE_FILE" ]]; then
        if ! create_fan_service; then
            return 1
        fi
    fi

    systemctl daemon-reload
    if systemctl enable jetson-boost-fan.service 2>/dev/null; then
        print_success "ファン自動起動サービスを有効化しました。"
        print_info "次回の起動時から、ファンが自動的に最大速度で起動します。"
    else
        print_error "サービスの有効化に失敗しました。"
        return 1
    fi
}

disable_fan_autostart() {
    print_header "システム起動時のファン自動起動を無効化"

    if systemctl disable jetson-boost-fan.service 2>/dev/null; then
        print_success "ファン自動起動サービスを無効化しました。"
        systemctl stop jetson-boost-fan.service 2>/dev/null && \
            print_info "サービスを停止しました。"
    else
        print_warn "サービスの無効化に失敗しました（既に無効の可能性があります）。"
    fi
}

fan_autostart_status() {
    print_header "ファン自動起動サービスの状態"
    systemctl status jetson-boost-fan.service --no-pager 2>/dev/null || \
        print_warn "サービスが見つかりません。"
}

# ----------------------------
# Interactive Menu
# ----------------------------
ask_reboot() {
    while true; do
        read -p "変更を有効にするためにシステムを再起動する必要があります。再起動しますか？ (y/n): " ans
        case "$ans" in
            [Yy]*)
                print_info "再起動します…"
                reboot
                ;;
            [Nn]*)
                print_info "再起動をキャンセルしました。"
                return
                ;;
            *)
                print_warn "有効な入力をしてください (y/n)。"
                ;;
        esac
    done
}

show_menu() {
    echo ""
    echo "Jetson Boost - パフォーマンス & ファン制御"
    echo "============================================================"
    echo " 電源/性能モード (nvpmodel)"
    echo "   1) 全性能モード (MAXN)              -> nvpmodel -m 0"
    echo "   2) 省電力モード (10W)               -> nvpmodel -m 1"
    echo ""
    echo " CUDA & クロック制御 (jetson_clocks)"
    echo "   3) 全CUDAコア有効化＆クロック最大化"
    echo "   4) クロック設定を復元（保存済み状態へ）"
    echo "   5) 現在のクロック状態を表示"
    echo ""
    echo " ファン制御（独立）"
    echo "   6) ファンを強制起動（最大速度）"
    echo "   7) ファンを自動制御モードに戻す"
    echo "   8) 現在のファン状態を表示"
    echo ""
    echo " 起動時ファン自動制御"
    echo "   9) システム起動時にファン自動起動を有効化"
    echo "   a) システム起動時にファン自動起動を無効化"
    echo "   b) ファン自動起動サービスの状態を表示"
    echo ""
    echo " 0) 終了"
    echo "============================================================"
}

run_interactive() {
    local nvp_ok=false
    local jc_ok=false

    cmd_exists nvpmodel && nvp_ok=true
    cmd_exists jetson_clocks && jc_ok=true

    if [[ "$nvp_ok" == "false" ]]; then
        print_warn "'nvpmodel' が見つかりません。電源モードの切替は使用できません。"
        print_info "  sudo apt-get update && sudo apt-get install -y nvpmodel"
    fi
    if [[ "$jc_ok" == "false" ]]; then
        print_warn "'jetson_clocks' が見つかりません。クロック制御は使用できません。"
        print_info "  SDK Manager で Jetson の開発者ツールをインストールしてください。"
    fi

    # Pre-setup
    if [[ "$jc_ok" == "true" ]]; then
        setup_jetson_clocks_config
    fi

    # Show current mode
    if [[ "$nvp_ok" == "true" ]]; then
        local cur
        cur=$(get_nvpmodel_mode)
        print_info "現在の電源モード: $cur"
    fi

    # Interactive loop
    while true; do
        show_menu
        read -p "選択 (0-9/a/b): " choice

        case "$choice" in
            0)
                print_info "スクリプトを終了します。"
                break
                ;;
            1)
                if [[ "$nvp_ok" == "false" ]]; then
                    print_warn "nvpmodel が利用できません。"
                    continue
                fi
                set_nvpmodel 0 && ask_reboot
                ;;
            2)
                if [[ "$nvp_ok" == "false" ]]; then
                    print_warn "nvpmodel が利用できません。"
                    continue
                fi
                set_nvpmodel 1 && ask_reboot
                ;;
            3)
                if [[ "$jc_ok" == "false" ]]; then
                    print_warn "jetson_clocks が利用できません。"
                    continue
                fi
                jetson_clocks_max
                ;;
            4)
                if [[ "$jc_ok" == "false" ]]; then
                    print_warn "jetson_clocks が利用できません。"
                    continue
                fi
                jetson_clocks_restore
                ;;
            5)
                if [[ "$jc_ok" == "false" ]]; then
                    print_warn "jetson_clocks が利用できません。"
                    continue
                fi
                jetson_clocks_show
                ;;
            6)
                fan_force_on
                ;;
            7)
                fan_force_off
                ;;
            8)
                fan_show_status
                ;;
            9)
                enable_fan_autostart
                ;;
            a|A)
                disable_fan_autostart
                ;;
            b|B)
                fan_autostart_status
                ;;
            *)
                print_warn "無効な選択です。0-9, a, b のいずれかを入力してください。"
                ;;
        esac
    done
}

# ----------------------------
# Show Status
# ----------------------------
show_status() {
    print_header "現在の状態"

    # nvpmodel status
    if cmd_exists nvpmodel; then
        print_info "電源モード (nvpmodel):"
        nvpmodel -q 2>/dev/null
        echo ""
    fi

    # jetson_clocks status
    if cmd_exists jetson_clocks; then
        print_info "クロック状態 (jetson_clocks):"
        jetson_clocks --show 2>/dev/null
    else
        print_warn "jetson_clocks が利用できません"
    fi

    # Fan status
    echo ""
    fan_show_status
}

# ----------------------------
# Help
# ----------------------------
show_usage() {
    cat << EOF
${GREEN}Jetson Boost - Performance & Fan Control${RESET}

${BLUE}Usage:${RESET}
  sudo $0 [OPTION]

${BLUE}Options:${RESET}
  ${CYAN}Interactive Mode:${RESET}
    --interactive, -i       Launch interactive menu (default)

  ${CYAN}Power Mode Control:${RESET}
    --maxn                  Set power mode to MAXN (maximum performance)
    --10w                   Set power mode to 10W (power saving)

  ${CYAN}Clock Control (CUDA Cores):${RESET}
    --max-clocks            Maximize CPU/GPU clocks and enable all CUDA cores
    --restore               Restore clocks to previously saved state
    --show                  Display current clock and fan status

  ${CYAN}Fan Control:${RESET}
    --fan-on                Force fan to maximum speed (255 PWM)
    --fan-off               Restore fan to automatic control mode
    --fan-status            Show current fan speed and PWM value

  ${CYAN}Auto-Start Control:${RESET}
    --enable-autostart      Enable fan auto-start at system boot
    --disable-autostart     Disable fan auto-start at system boot
    --autostart-status      Show fan auto-start service status

  ${CYAN}Help:${RESET}
    --help, -h              Show this help message

${BLUE}Examples:${RESET}
  ${GREEN}Interactive menu:${RESET}
    sudo $0

  ${GREEN}Maximum performance setup:${RESET}
    sudo $0 --maxn              # Set power mode to MAXN
    sudo $0 --max-clocks        # Maximize clocks/CUDA cores
    sudo $0 --fan-on            # Force fan on

  ${GREEN}Fan control:${RESET}
    sudo $0 --fan-on            # Turn fan on (max speed)
    sudo $0 --fan-off           # Return to auto mode
    sudo $0 --fan-status        # Check fan status

  ${GREEN}Auto-start at boot:${RESET}
    sudo $0 --enable-autostart  # Fan starts automatically at boot
    sudo $0 --disable-autostart # Disable auto-start

  ${GREEN}Restore normal state:${RESET}
    sudo $0 --restore           # Restore clocks
    sudo $0 --fan-off           # Fan to auto mode

${BLUE}Notes:${RESET}
  - This script requires root privileges (sudo)
  - Only works on NVIDIA Jetson platforms
  - Power mode changes (--maxn, --10w) may require a reboot
  - Fan and clock controls are independent
  - Use --restore to return to normal clock state
EOF
}

# ----------------------------
# Main
# ----------------------------
main() {
    # Parse command line arguments
    case "${1:-}" in
        --interactive|-i|"")
            require_root
            require_jetson
            run_interactive
            ;;
        --maxn)
            require_root
            require_jetson
            print_header "Setting MAXN Performance Mode"
            if cmd_exists nvpmodel; then
                set_nvpmodel 0 && ask_reboot
            else
                print_error "nvpmodel が見つかりません。"
                exit 1
            fi
            ;;
        --10w)
            require_root
            require_jetson
            print_header "Setting 10W Power Saving Mode"
            if cmd_exists nvpmodel; then
                set_nvpmodel 1 && ask_reboot
            else
                print_error "nvpmodel が見つかりません。"
                exit 1
            fi
            ;;
        --max-clocks)
            require_root
            require_jetson
            jetson_clocks_max
            ;;
        --restore)
            require_root
            require_jetson
            jetson_clocks_restore
            ;;
        --fan-on)
            require_root
            require_jetson
            fan_force_on
            ;;
        --fan-off)
            require_root
            require_jetson
            fan_force_off
            ;;
        --fan-status)
            require_jetson
            fan_show_status
            ;;
        --enable-autostart)
            require_root
            require_jetson
            enable_fan_autostart
            ;;
        --disable-autostart)
            require_root
            require_jetson
            disable_fan_autostart
            ;;
        --autostart-status)
            require_jetson
            fan_autostart_status
            ;;
        --show)
            require_jetson
            show_status
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
