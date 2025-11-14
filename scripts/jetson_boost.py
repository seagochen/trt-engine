#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

CONFIG_FILE = "/root/.jetsonclocks_conf.txt"
BACKUP_FILE = "/root/.jetsonclocks_conf.backup.txt"
FAN_SERVICE_FILE = "/etc/systemd/system/jetson-boost-fan.service"
FAN_PWM_PATH = "/sys/devices/pwm-fan/target_pwm"
FAN_HWMON_PATHS = [
    "/sys/devices/pwm-fan/hwmon",
    "/sys/class/hwmon"
]

# ---------- utils ----------
def run(cmd, check=True, capture=False, text=True):
    return subprocess.run(
        cmd, check=check, capture_output=capture, text=text, shell=isinstance(cmd, str)
    )

def cmd_exists(name: str) -> bool:
    return run(["bash", "-lc", f"command -v {name}"], check=False).returncode == 0

def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    comp = Path("/proc/device-tree/compatible")
    if comp.exists() and "tegra" in comp.read_text(errors="ignore").lower():
        return True
    return False

def require_root():
    if os.geteuid() != 0:
        print("このスクリプトを実行するには root 権限が必要です。")
        raise SystemExit(1)

# ---------- nvpmodel ----------
def ensure_nvpmodel():
    if not cmd_exists("nvpmodel"):
        print("エラー: nvpmodel がインストールされていません。")
        print("インストールするには以下のコマンドを実行してください:")
        print("  sudo apt-get update && sudo apt-get install -y nvpmodel")
        raise SystemExit(1)

def get_nvpmodel_mode() -> str:
    # 例: "Mode: 0" を抽出
    p = run(["nvpmodel", "-q"], check=False, capture=True)
    out = p.stdout or ""
    for line in out.splitlines():
        if "Mode" in line:
            # "Mode: 0" → 0
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith("Mode"):
                return parts[-1]
    return "不明"

def set_nvpmodel(mode_idx: int):
    print(f"nvpmodel モードを {mode_idx} に切り替えます…")
    run(["nvpmodel", "-m", str(mode_idx)])

# ---------- fan control ----------
def find_fan_pwm_path():
    """查找风扇 PWM 控制路径，返回 (pwm_path, enable_path) 元组"""
    # 方法1: 直接路径
    if Path(FAN_PWM_PATH).exists():
        enable_path = None
        # 查找对应的 enable 文件
        parent = Path(FAN_PWM_PATH).parent
        for enable_file in ["pwm1_enable", "pwm_enable", "temp_control"]:
            candidate = parent / enable_file
            if candidate.exists():
                enable_path = str(candidate)
                break
        return (FAN_PWM_PATH, enable_path)

    # 方法2: 通过 hwmon 查找
    for base_path in FAN_HWMON_PATHS:
        base = Path(base_path)
        if not base.exists():
            continue
        # 遍历 hwmon 设备
        for hwmon_dir in base.glob("hwmon*"):
            pwm_file = hwmon_dir / "pwm1"
            if pwm_file.exists():
                # 查找对应的 enable 文件
                enable_path = None
                for enable_file in ["pwm1_enable", "pwm_enable", "temp_control"]:
                    candidate = hwmon_dir / enable_file
                    if candidate.exists():
                        enable_path = str(candidate)
                        break
                return (str(pwm_file), enable_path)

    return (None, None)

def set_fan_speed(speed: int, manual_mode: bool = True):
    """设置风扇速度 (0-255)

    Args:
        speed: PWM 速度值 (0-255)
        manual_mode: 是否先设置为手动模式 (True=手动, False=自动)
    """
    pwm_path, enable_path = find_fan_pwm_path()
    if not pwm_path:
        print("エラー: ファン制御パスが見つかりません。")
        print("このデバイスはファン制御をサポートしていない可能性があります。")
        return False

    try:
        # 1. 先设置为手动模式（如果有 enable 文件）
        if manual_mode and enable_path:
            try:
                with open(enable_path, 'w') as f:
                    f.write("1")  # 1 = 手动模式
                print(f"ファン制御を手動モードに設定しました: {enable_path}")
            except Exception as e:
                print(f"警告: 手動モード設定に失敗しました: {e}")
                print("  一部のデバイスでは、この手順は不要な場合があります。")

        # 2. PWM 速度を設定
        with open(pwm_path, 'w') as f:
            f.write(str(speed))
        print(f"ファン速度を {speed}/255 に設定しました: {pwm_path}")
        return True
    except Exception as e:
        print(f"エラー: ファン速度の設定に失敗しました: {e}")
        return False

def fan_force_on():
    """强制启动风扇到最大速度"""
    print("ファンを最大速度で強制起動します…")
    if set_fan_speed(255):
        print("ファンの強制起動に成功しました。")
    else:
        # 尝试使用 jetson_clocks
        print("代替方法を試します: jetson_clocks を使用...")
        if run(["jetson_clocks", "--fan"], check=False).returncode == 0:
            print("jetson_clocks を使用してファンを起動しました。")
        else:
            print("エラー: ファンの起動に失敗しました。")

def fan_force_off():
    """关闭强制风扇，恢复自动控制"""
    print("ファンを自動制御モードに戻します…")
    pwm_path, enable_path = find_fan_pwm_path()
    if not pwm_path:
        print("警告: ファン制御パスが見つかりません。")
        return

    try:
        # 方法1: 如果有 enable 文件，设置为自动模式
        if enable_path:
            try:
                with open(enable_path, 'w') as f:
                    # 2 = 自动模式（温度控制）
                    # 某些设备使用 0 = 自动, 1 = 手动
                    # 尝试 2 (通常是自动模式)
                    f.write("2")
                print(f"ファン制御を自動モード(温度制御)に設定しました: {enable_path}")
            except Exception as e:
                print(f"警告: 自動モード設定に失敗しました: {e}")
                # 尝试设置为 0
                try:
                    with open(enable_path, 'w') as f:
                        f.write("0")
                    print("代替方法で自動モードに設定しました。")
                except:
                    pass

        # 方法2: 将 PWM 设置为 0（某些设备这样就能恢复自动模式）
        with open(pwm_path, 'w') as f:
            f.write("0")
        print("ファンを自動制御モードに戻しました。")
    except Exception as e:
        print(f"警告: {e}")
        print("jetson_clocks --restore を試してください。")

def fan_show_status():
    """显示当前风扇状态"""
    print("現在のファン状態:")
    pwm_path, enable_path = find_fan_pwm_path()
    if not pwm_path:
        print("  ファン制御パスが見つかりません。")
        return

    try:
        # 显示 PWM 路径
        print(f"  PWMパス: {pwm_path}")

        # 读取 PWM 值
        with open(pwm_path, 'r') as f:
            pwm_val = f.read().strip()
        print(f"  PWM値: {pwm_val}/255")
        percentage = int(pwm_val) * 100 // 255
        print(f"  速度: {percentage}%")

        # 读取控制模式
        if enable_path:
            try:
                with open(enable_path, 'r') as f:
                    enable_val = f.read().strip()
                mode_map = {
                    "0": "無効または自動",
                    "1": "手動制御",
                    "2": "自動（温度制御）",
                    "3": "完全速度"
                }
                mode = mode_map.get(enable_val, f"不明 ({enable_val})")
                print(f"  制御モード: {mode}")
                print(f"  制御パス: {enable_path}")
            except Exception as e:
                print(f"  制御モード: 読み取り失敗 ({e})")
        else:
            print("  制御モード: enable ファイルが見つかりません")

    except Exception as e:
        print(f"  読み取りエラー: {e}")

# ---------- jetson_clocks ----------
def ensure_jetson_clocks():
    if not cmd_exists("jetson_clocks"):
        print("エラー: jetson_clocks が見つかりません。")
        print("JetPack 付属のユーティリティです。SDK Manager で Jetson の開発者ツールを有効化してください。")
        raise SystemExit(1)

def setup_jetson_clocks_config():
    print("クロック設定ファイルの確認とセットアップ中...")
    if not Path(CONFIG_FILE).exists():
        print(f"クロック設定ファイル ({CONFIG_FILE}) が存在しません。新しく作成します…")
        if run(["jetson_clocks", "--store"], check=False).returncode == 0:
            print(f"設定ファイルを作成しました: {CONFIG_FILE}")
        else:
            print("エラー: 設定ファイルの作成に失敗しました。'jetson_clocks --store' を手動で実行してみてください。")

    # 初回バックアップ
    if Path(CONFIG_FILE).exists() and not Path(BACKUP_FILE).exists():
        print(f"初回バックアップ ({BACKUP_FILE}) を作成中…")
        cp = run(["cp", CONFIG_FILE, BACKUP_FILE], check=False)
        if cp.returncode == 0:
            run(["chmod", "444", BACKUP_FILE], check=False)
            print(f"バックアップを作成しました: {BACKUP_FILE}")
        else:
            print("警告: バックアップの作成に失敗しました。")

def jetson_clocks_max():
    """最大化所有 CUDA 核心和时钟频率"""
    print("全CUDAコアを有効化し、クロックを最大化します…")
    # 使用 jetson_clocks 但不包含风扇控制
    if run(["jetson_clocks"], check=False).returncode == 0:
        print("CUDAコアとクロックの最大化に成功しました。")
        # 显示当前状态
        jetson_clocks_show()
    else:
        print("エラー: クロックの最大化に失敗しました。")

def jetson_clocks_restore():
    """恢复到保存的性能状态"""
    print("クロックを保存された状態に戻します…")
    if not Path(CONFIG_FILE).exists():
        print(f"エラー: 復元するための設定ファイル ({CONFIG_FILE}) が見つかりません。")
        print("先に「クロック最大化」を実行するか、手動で 'sudo jetson_clocks --store' を実行してください。")
        return
    if run(["jetson_clocks", "--restore"], check=False).returncode == 0:
        print("クロックの復元に成功しました。")
    else:
        print("エラー: 復元に失敗しました。")

def jetson_clocks_show():
    """显示当前时钟状态"""
    print("\n現在のクロック状態:")
    if run(["jetson_clocks", "--show"], check=False).returncode != 0:
        print("エラー: クロック状態の表示に失敗しました。")

# ---------- systemd service for auto fan control ----------
def create_fan_service():
    """创建系统启动时自动启动风扇的 systemd 服务"""
    service_content = """[Unit]
Description=Jetson Boost - Fan Auto-Start at Boot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 {script_path} --fan-on
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
""".format(script_path=Path(__file__).resolve())

    try:
        with open(FAN_SERVICE_FILE, 'w') as f:
            f.write(service_content)
        print(f"systemd サービスファイルを作成しました: {FAN_SERVICE_FILE}")
        return True
    except Exception as e:
        print(f"エラー: サービスファイルの作成に失敗しました: {e}")
        return False

def enable_fan_autostart():
    """启用系统启动时自动强制风扇"""
    print("システム起動時のファン自動起動を有効化します…")

    if not Path(FAN_SERVICE_FILE).exists():
        if not create_fan_service():
            return

    # リロードと有効化
    run(["systemctl", "daemon-reload"], check=False)
    if run(["systemctl", "enable", "jetson-boost-fan.service"], check=False).returncode == 0:
        print("ファン自動起動サービスを有効化しました。")
        print("次回の起動時から、ファンが自動的に最大速度で起動します。")
    else:
        print("エラー: サービスの有効化に失敗しました。")

def disable_fan_autostart():
    """禁用系统启动时自动强制风扇"""
    print("システム起動時のファン自動起動を無効化します…")

    if run(["systemctl", "disable", "jetson-boost-fan.service"], check=False).returncode == 0:
        print("ファン自動起動サービスを無効化しました。")
        if run(["systemctl", "stop", "jetson-boost-fan.service"], check=False).returncode == 0:
            print("サービスを停止しました。")
    else:
        print("警告: サービスの無効化に失敗しました（既に無効の可能性があります）。")

def fan_autostart_status():
    """显示自动启动服务状态"""
    print("\nファン自動起動サービスの状態:")
    run(["systemctl", "status", "jetson-boost-fan.service", "--no-pager"], check=False)

# ---------- UI ----------
def ask_reboot():
    while True:
        ans = input("変更を有効にするためにシステムを再起動する必要があります。再起動しますか？ (y/n): ").strip().lower()
        if ans.startswith("y"):
            print("再起動します…")
            run(["reboot"], check=False)
            return
        if ans.startswith("n"):
            print("再起動をキャンセルしました。")
            return
        print("有効な入力をしてください (y/n)。")

def menu():
    print("")
    print("Jetson Boost - パフォーマンス & ファン制御")
    print("=" * 60)
    print(" 電源/性能モード (nvpmodel)")
    print("   1) 全性能モード (MAXN)              -> nvpmodel -m 0")
    print("   2) 省電力モード (10W)               -> nvpmodel -m 1")
    print("")
    print(" CUDA & クロック制御 (jetson_clocks)")
    print("   3) 全CUDAコア有効化＆クロック最大化")
    print("   4) クロック設定を復元（保存済み状態へ）")
    print("   5) 現在のクロック状態を表示")
    print("")
    print(" ファン制御（独立）")
    print("   6) ファンを強制起動（最大速度）")
    print("   7) ファンを自動制御モードに戻す")
    print("   8) 現在のファン状態を表示")
    print("")
    print(" 起動時ファン自動制御")
    print("   9) システム起動時にファン自動起動を有効化")
    print("   a) システム起動時にファン自動起動を無効化")
    print("   b) ファン自動起動サービスの状態を表示")
    print("")
    print(" 0) 終了")
    print("=" * 60)

def main():
    import sys

    # 处理命令行参数（用于 systemd 服务调用）
    if len(sys.argv) > 1:
        require_root()
        if not is_jetson():
            print("このスクリプトは NVIDIA Jetson 専用です。終了します。")
            raise SystemExit(1)

        arg = sys.argv[1]
        if arg == "--fan-on":
            fan_force_on()
            return
        elif arg == "--fan-off":
            fan_force_off()
            return
        elif arg == "--fan-status":
            fan_show_status()
            return
        elif arg == "--max-clocks":
            if cmd_exists("jetson_clocks"):
                jetson_clocks_max()
            else:
                print("エラー: jetson_clocks が見つかりません。")
            return
        elif arg == "--restore":
            if cmd_exists("jetson_clocks"):
                jetson_clocks_restore()
            else:
                print("エラー: jetson_clocks が見つかりません。")
            return
        else:
            print(f"不明なオプション: {arg}")
            print("使用可能なオプション: --fan-on, --fan-off, --fan-status, --max-clocks, --restore")
            return

    # 交互模式
    require_root()

    if not is_jetson():
        print("このスクリプトは NVIDIA Jetson 専用です。終了します。")
        raise SystemExit(1)

    # 工具存在チェック
    nvp_ok = cmd_exists("nvpmodel")
    jc_ok = cmd_exists("jetson_clocks")

    if not nvp_ok:
        print("警告: 'nvpmodel' が見つかりません。電源モードの切替は使用できません。")
        print("  sudo apt-get update && sudo apt-get install -y nvpmodel")
    if not jc_ok:
        print("警告: 'jetson_clocks' が見つかりません。クロック制御は使用できません。")
        print("  SDK Manager で Jetson の開発者ツールをインストールしてください。")

    # 事前セットアップ
    if jc_ok:
        setup_jetson_clocks_config()

    # 初回情報
    if nvp_ok:
        try:
            cur = get_nvpmodel_mode()
            print(f"現在の電源モード: {cur}")
        except Exception:
            print("現在の電源モード取得に失敗しました。")

    # 交互ループ
    while True:
        menu()
        choice = input("選択 (0-9/a/b): ").strip().lower()

        if choice == "0":
            print("スクリプトを終了します。")
            break

        elif choice == "1":
            if not nvp_ok:
                print("nvpmodel が利用できません。")
                continue
            try:
                set_nvpmodel(0)  # MAXN
                ask_reboot()
            except subprocess.CalledProcessError:
                print("エラー: モード切替に失敗しました。")

        elif choice == "2":
            if not nvp_ok:
                print("nvpmodel が利用できません。")
                continue
            try:
                set_nvpmodel(1)  # 10W
                ask_reboot()
            except subprocess.CalledProcessError:
                print("エラー: モード切替に失敗しました。")

        elif choice == "3":
            if not jc_ok:
                print("jetson_clocks が利用できません。")
                continue
            jetson_clocks_max()

        elif choice == "4":
            if not jc_ok:
                print("jetson_clocks が利用できません。")
                continue
            jetson_clocks_restore()

        elif choice == "5":
            if not jc_ok:
                print("jetson_clocks が利用できません。")
                continue
            jetson_clocks_show()

        elif choice == "6":
            fan_force_on()

        elif choice == "7":
            fan_force_off()

        elif choice == "8":
            fan_show_status()

        elif choice == "9":
            enable_fan_autostart()

        elif choice == "a":
            disable_fan_autostart()

        elif choice == "b":
            fan_autostart_status()

        else:
            print("無効な選択です。0-9, a, b のいずれかを入力してください。")

if __name__ == "__main__":
    main()
