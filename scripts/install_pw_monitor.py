#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path


def run_command(cmd, use_sudo=False):
    if use_sudo:
        cmd = ["sudo"] + cmd
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def is_jetson():
    """Detect if the system is NVIDIA Jetson."""
    # Jetson 系统一般有这个文件
    if Path("/etc/nv_tegra_release").exists():
        return True
    # 或者 device-tree 里有 tegra 字样
    dt = Path("/proc/device-tree/compatible")
    if dt.exists() and "tegra" in dt.read_text(errors="ignore").lower():
        return True
    return False


def install_jtop():
    print("[INFO] Installing jetson-stats (jtop) for Jetson...")
    run_command([sys.executable, "-m", "pip", "install", "-U", "jetson-stats"], use_sudo=True)
    print("[INFO] jtop installation complete. Run it with: jtop")


def install_nvitop():
    print("[INFO] Installing nvitop for x86 system...")
    run_command([sys.executable, "-m", "pip", "install", "-U", "nvitop"])
    print("[INFO] nvitop installation complete. Run it with: nvitop")


def main():
    if is_jetson():
        install_jtop()
    else:
        install_nvitop()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
