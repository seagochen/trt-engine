#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TrtEnv - TensorRT Environment Setup Tool

Comprehensive environment configuration for TensorRT and CUDA:
  1. Configure shell environment variables (PATH, LD_LIBRARY_PATH)
  2. Configure system library search paths (ldconfig)
  3. Install performance monitoring tools (jtop/nvitop)

Usage:
  python3 trtenv.py --setup-all           # Configure everything
  python3 trtenv.py --configure-env       # Configure shell environment only
  python3 trtenv.py --ldconfig           # Configure system ldconfig only
  python3 trtenv.py --install-monitor    # Install monitoring tools only
  python3 trtenv.py --show               # Show current configuration
  python3 trtenv.py --remove             # Remove configurations
  python3 trtenv.py --dry-run --setup-all # Show what would be done
"""

import os
import re
import sys
import glob
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Set

# ----------------------------
# Utilities
# ----------------------------
class Color:
    INFO = "\033[1;34m[INFO]\033[0m"
    WARN = "\033[1;33m[WARN]\033[0m"
    ERROR = "\033[1;31m[ERROR]\033[0m"
    SUCCESS = "\033[1;32m[SUCCESS]\033[0m"

def print_info(msg: str) -> None:
    print(f"{Color.INFO} {msg}")

def print_warn(msg: str) -> None:
    print(f"{Color.WARN} {msg}")

def print_error(msg: str) -> None:
    print(f"{Color.ERROR} {msg}", file=sys.stderr)

def print_success(msg: str) -> None:
    print(f"{Color.SUCCESS} {msg}")

def is_jetson() -> bool:
    """Detect if running on NVIDIA Jetson platform."""
    if Path("/etc/nv_tegra_release").exists():
        return True
    dt = Path("/proc/device-tree/compatible")
    if dt.exists() and "tegra" in dt.read_text(errors="ignore").lower():
        return True
    return False

def which(exe: str) -> Optional[Path]:
    """Find executable in PATH."""
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / exe
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    return None

def is_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0

def run_command(cmd: List[str], check: bool = True, capture: bool = False) -> Optional[str]:
    """Run shell command."""
    try:
        if capture:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.stdout
        else:
            subprocess.run(cmd, check=check)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return None

# ----------------------------
# Module 1: CUDA & TensorRT Detection
# ----------------------------
def detect_cuda() -> Optional[Tuple[Path, Path]]:
    """
    Returns (cuda_root, cuda_lib) or None if not found.
    Detection order:
      1) /usr/local/cuda symlink if valid
      2) Highest version under /usr/local/cuda-*
      3) Fallback by locating nvcc in PATH
    """
    candidates: List[Path] = []

    # 1) Standard installation
    cuda_symlink = Path("/usr/local/cuda")
    if (cuda_symlink / "bin" / "nvcc").exists():
        candidates.append(cuda_symlink)

    # 2) Version directories (pick highest version)
    vers = sorted(Path("/usr/local").glob("cuda-*"), key=lambda p: p.name, reverse=True)
    for v in vers:
        if (v / "bin" / "nvcc").exists():
            candidates.append(v)

    # 3) nvcc in PATH
    nvcc = which("nvcc")
    if nvcc:
        candidates.append(nvcc.parent.parent)

    # Deduplicate
    seen: Set[str] = set()
    uniq: List[Path] = []
    for c in candidates:
        if str(c) not in seen:
            uniq.append(c)
            seen.add(str(c))

    for root in uniq:
        lib = root / "lib64"
        if not lib.is_dir():
            lib = root / "lib"
        if (root / "bin" / "nvcc").exists():
            return (root, lib)

    return None

def _trtexec_ok(root: Path) -> bool:
    """Check if bin/trtexec exists under root."""
    trtexec = root / "bin" / "trtexec"
    return trtexec.exists() and os.access(trtexec, os.X_OK)

def _trt_lib_dir(root: Path) -> Optional[Path]:
    """Return TensorRT lib directory under given root."""
    for sub in ["lib", "lib64", "targets/x86_64-linux-gnu/lib", "targets/aarch64-linux-gnu/lib"]:
        p = root / sub
        if p.is_dir() and list(p.glob("libnvinfer*")):
            return p
    return None

def detect_tensorrt() -> Optional[Tuple[Path, Optional[Path]]]:
    """
    Returns (trt_root, trt_lib_dir) or None.
    Detection order:
      Jetson: /usr/src/tensorrt
      x86: /opt/tensorrt, /usr/local/TensorRT*, system libs
    """
    if is_jetson():
        root = Path("/usr/src/tensorrt")
        if _trtexec_ok(root) or _trt_lib_dir(root):
            return (root, _trt_lib_dir(root))

    # x86 common directories
    preferred: List[Path] = []
    opt_trt = Path("/opt/tensorrt")
    if opt_trt.exists():
        preferred.append(opt_trt)

    # Multi-version directories
    tdirs = sorted(Path("/usr/local").glob("TensorRT*"), key=lambda p: p.name, reverse=True)
    preferred.extend(tdirs)

    # Deduplicate
    seen: Set[str] = set()
    uniq: List[Path] = []
    for p in preferred:
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))

    for root in uniq:
        if _trtexec_ok(root) or _trt_lib_dir(root):
            return (root, _trt_lib_dir(root))

    # System library fallback
    lib_candidates = [
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib/x86_64-linux-gnu/tensorrt"),
        Path("/usr/lib/aarch64-linux-gnu/tensorrt"),
    ]
    for lc in lib_candidates:
        if lc.is_dir() and list(lc.glob("libnvinfer*")):
            return (lc.parent if lc.name == "tensorrt" else lc, lc)

    return None

# ----------------------------
# Module 2: Shell Environment Configuration
# ----------------------------
def inject_block(bashrc_path: Path, marker: str, lines: List[str], dry_run: bool = False) -> None:
    """
    Insert or replace a marked block in shell rc file.
    Block format:
      # >>> {marker}
      <lines...>
      # <<< {marker}
    """
    begin = f"# >>> {marker}"
    end = f"# <<< {marker}"

    if not bashrc_path.exists():
        content = ""
    else:
        content = bashrc_path.read_text()

    block = "\n".join([begin, *lines, end]) + "\n"

    if begin in content and end in content:
        # Replace existing block
        new = re.sub(
            rf"{re.escape(begin)}.*?{re.escape(end)}\n?",
            block,
            content,
            flags=re.DOTALL,
        )
        action = "更新"
    else:
        # Append new block
        new = content + ("\n" if content and not content.endswith("\n") else "") + block
        action = "追加"

    if dry_run:
        print_info(f"[dry-run] 将在 {bashrc_path} 中{action}标记块：{marker}")
        print(block)
        return

    bashrc_path.write_text(new)
    print_success(f"{bashrc_path} 已{action}标记块：{marker}")

def export_lines(path_bin: Optional[Path], path_lib: Optional[Path]) -> List[str]:
    """Generate export statements for PATH and LD_LIBRARY_PATH."""
    out: List[str] = []
    if path_bin:
        out.append(f'export PATH="$PATH:{path_bin}"')
    if path_lib:
        out.append('export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}' + f'{path_lib}"')
    return out

def configure_shell_env(bashrc: str = "~/.bashrc", dry_run: bool = False) -> bool:
    """Configure shell environment variables for CUDA and TensorRT."""
    print_info("=" * 60)
    print_info("模块 1: 配置 Shell 环境变量")
    print_info("=" * 60)

    bashrc_path = Path(bashrc).expanduser()

    # Detect CUDA
    cuda_found = detect_cuda()
    if cuda_found:
        cuda_root, cuda_lib = cuda_found
        cuda_marker = "CUDA Toolkit Environment Variables (trtenv)"
        cuda_lines = export_lines(cuda_root / "bin", cuda_lib)
        print_info(f"检测到 CUDA: {cuda_root}")
        inject_block(bashrc_path, cuda_marker, cuda_lines, dry_run=dry_run)
    else:
        print_warn("未检测到 CUDA，跳过 CUDA 环境变量配置")

    # Detect TensorRT
    trt_found = detect_tensorrt()
    if trt_found:
        trt_root, trt_lib = trt_found
        platform = "Jetson" if is_jetson() else "x86"
        trt_marker = f"TensorRT Environment Variables (trtenv, {platform})"
        trt_bin = trt_root / "bin" if (trt_root / "bin").is_dir() else None
        trt_lines = export_lines(trt_bin, trt_lib)

        trtexec = trt_bin / "trtexec" if trt_bin else None
        if trtexec and trtexec.exists():
            print_info(f"检测到 TensorRT: {trt_root} (trtexec: {trtexec})")
        else:
            print_info(f"检测到 TensorRT 库: {trt_lib if trt_lib else trt_root}")
        inject_block(bashrc_path, trt_marker, trt_lines, dry_run=dry_run)
    else:
        print_warn("未检测到 TensorRT，跳过 TensorRT 环境变量配置")

    if not dry_run and (cuda_found or trt_found):
        print_info(f"\n配置已写入: {bashrc_path}")
        print_info(f"执行 'source {bashrc_path}' 或重新打开终端以使更改生效")
        return True
    elif dry_run:
        print_info("\n[dry-run] 未对文件做任何修改")
        return False
    else:
        print_warn("\n未检测到 CUDA 或 TensorRT，无配置写入")
        return False

# ----------------------------
# Module 3: Ldconfig Configuration
# ----------------------------
def configure_ldconfig(lib_dirs: Optional[List[str]] = None, dry_run: bool = False, remove: bool = False) -> bool:
    """Configure system library search paths via ldconfig."""
    print_info("=" * 60)
    print_info("模块 2: 配置系统库搜索路径 (ldconfig)")
    print_info("=" * 60)

    CONF_FILE = "/etc/ld.so.conf.d/tensorrt.conf"

    if remove:
        if dry_run:
            print_info(f"[dry-run] 将删除 {CONF_FILE} 并运行 ldconfig")
            return False

        if not is_root():
            print_error("需要 root 权限才能修改 ldconfig 配置")
            return False

        if Path(CONF_FILE).exists():
            print_info(f"正在删除 {CONF_FILE}")
            Path(CONF_FILE).unlink()
            run_command(["ldconfig"])
            print_success("已删除 ldconfig 配置")
        else:
            print_info(f"{CONF_FILE} 不存在，无需删除")
        return True

    # Auto-detect library directories if not provided
    if not lib_dirs:
        lib_dirs = []

        # Detect TensorRT
        trt_found = detect_tensorrt()
        if trt_found:
            _, trt_lib = trt_found
            if trt_lib:
                lib_dirs.append(str(trt_lib))

        # Detect CUDA
        cuda_found = detect_cuda()
        if cuda_found:
            _, cuda_lib = cuda_found
            if cuda_lib:
                lib_dirs.append(str(cuda_lib))

    if not lib_dirs:
        print_warn("未提供库目录且未自动检测到 CUDA/TensorRT 库")
        return False

    # Validate directories
    validated_dirs: List[str] = []
    has_trt = False
    for d in lib_dirs:
        path = Path(d)
        if not path.exists():
            print_error(f"目录不存在: {d}")
            return False

        # Check for TensorRT libs
        if list(path.glob("libnvinfer*.so*")):
            has_trt = True

        validated_dirs.append(str(path.resolve()))

    if not has_trt:
        print_warn("提供的目录中未检测到 libnvinfer*.so* 文件")

    # Read existing configuration
    existing_lines: List[str] = []
    if Path(CONF_FILE).exists():
        with open(CONF_FILE, 'r') as f:
            existing_lines = [line.strip() for line in f if line.strip()]

    # Merge and deduplicate
    all_lines = existing_lines + validated_dirs
    unique_lines = []
    seen: Set[str] = set()
    for line in all_lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    print_info("将注册以下库目录到 ldconfig:")
    for line in unique_lines:
        print(f"  - {line}")

    if dry_run:
        print_info("\n[dry-run] 将写入以上条目并执行 ldconfig")
        return False

    if not is_root():
        print_error("需要 root 权限才能修改 ldconfig 配置")
        print_info("请使用 sudo 运行此脚本")
        return False

    # Backup existing config
    if Path(CONF_FILE).exists():
        import time
        backup = f"{CONF_FILE}.bak.{int(time.time())}"
        Path(CONF_FILE).rename(backup)
        print_info(f"已创建备份: {backup}")

    # Write new configuration
    with open(CONF_FILE, 'w') as f:
        for line in unique_lines:
            f.write(f"{line}\n")

    os.chmod(CONF_FILE, 0o644)
    print_success(f"已写入 {len(unique_lines)} 条记录到 {CONF_FILE}")

    # Run ldconfig
    print_info("正在运行 ldconfig...")
    run_command(["ldconfig"])
    print_success("ldconfig 配置完成")

    # Verify
    print_info("\n验证配置:")
    output = run_command(["ldconfig", "-p"], capture=True)
    if output and "nvinfer" in output:
        for line in output.split('\n'):
            if 'nvinfer' in line.lower():
                print(f"  {line.strip()}")
    else:
        print_warn("  未在 ldconfig 缓存中找到 nvinfer 库")

    return True

# ----------------------------
# Module 4: Install Monitoring Tools
# ----------------------------
def install_monitoring_tool(dry_run: bool = False) -> bool:
    """Install performance monitoring tools (jtop for Jetson, nvitop for x86)."""
    print_info("=" * 60)
    print_info("模块 3: 安装性能监控工具")
    print_info("=" * 60)

    if is_jetson():
        print_info("检测到 Jetson 平台，将安装 jetson-stats (jtop)")
        tool = "jetson-stats"
        cmd = [sys.executable, "-m", "pip", "install", "-U", "jetson-stats"]
        need_sudo = True
    else:
        print_info("检测到 x86 平台，将安装 nvitop")
        tool = "nvitop"
        cmd = [sys.executable, "-m", "pip", "install", "-U", "nvitop"]
        need_sudo = False

    if dry_run:
        print_info(f"[dry-run] 将执行: {' '.join(cmd)}")
        return False

    try:
        if need_sudo and not is_root():
            cmd = ["sudo"] + cmd

        print_info(f"正在安装 {tool}...")
        run_command(cmd)
        print_success(f"{tool} 安装完成")

        if is_jetson():
            print_info("运行 'jtop' 命令来启动监控工具")
        else:
            print_info("运行 'nvitop' 命令来启动监控工具")

        return True
    except Exception as e:
        print_error(f"安装失败: {e}")
        return False

# ----------------------------
# Module 5: Show Configuration
# ----------------------------
def show_configuration() -> None:
    """Display current environment configuration."""
    print_info("=" * 60)
    print_info("当前 TensorRT 环境配置")
    print_info("=" * 60)

    # Platform
    print(f"\n{'平台:':<20} {'Jetson' if is_jetson() else 'x86_64'}")

    # CUDA
    cuda_found = detect_cuda()
    if cuda_found:
        cuda_root, cuda_lib = cuda_found
        print(f"{'CUDA 根目录:':<20} {cuda_root}")
        print(f"{'CUDA 库目录:':<20} {cuda_lib}")

        nvcc = which("nvcc")
        if nvcc:
            try:
                version = run_command(["nvcc", "--version"], capture=True)
                if version:
                    ver_line = [l for l in version.split('\n') if 'release' in l.lower()]
                    if ver_line:
                        print(f"{'CUDA 版本:':<20} {ver_line[0].strip()}")
            except:
                pass
    else:
        print(f"{'CUDA:':<20} 未检测到")

    # TensorRT
    trt_found = detect_tensorrt()
    if trt_found:
        trt_root, trt_lib = trt_found
        print(f"{'TensorRT 根目录:':<20} {trt_root}")
        print(f"{'TensorRT 库目录:':<20} {trt_lib if trt_lib else '(系统库)'}")

        trtexec = trt_root / "bin" / "trtexec"
        if trtexec.exists():
            print(f"{'trtexec:':<20} {trtexec}")
    else:
        print(f"{'TensorRT:':<20} 未检测到")

    # Ldconfig
    print(f"\n{'Ldconfig 配置:':<20}")
    conf_file = Path("/etc/ld.so.conf.d/tensorrt.conf")
    if conf_file.exists():
        with open(conf_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            print(f"  - {line}")
    else:
        print(f"  未配置")

    # Monitoring tools
    print(f"\n{'监控工具:':<20}")
    if is_jetson():
        if which("jtop"):
            print(f"  jtop: 已安装")
        else:
            print(f"  jtop: 未安装")
    else:
        if which("nvitop"):
            print(f"  nvitop: 已安装")
        else:
            print(f"  nvitop: 未安装")

    # Shell environment
    print(f"\n{'Shell 环境变量:':<20}")
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        content = bashrc.read_text()
        if "trtenv" in content:
            print(f"  {bashrc}: 已配置")
        else:
            print(f"  {bashrc}: 未配置")
    else:
        print(f"  {bashrc}: 文件不存在")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TrtEnv - TensorRT Environment Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 trtenv.py --setup-all                 # Configure everything
  python3 trtenv.py --configure-env             # Configure shell environment only
  python3 trtenv.py --ldconfig                  # Configure ldconfig only
  python3 trtenv.py --install-monitor           # Install monitoring tools
  python3 trtenv.py --show                      # Show current configuration
  python3 trtenv.py --remove                    # Remove ldconfig configuration
  python3 trtenv.py --dry-run --setup-all       # Preview changes
        """
    )

    # Actions
    parser.add_argument("--setup-all", action="store_true", help="Configure everything (env + ldconfig + monitor)")
    parser.add_argument("--configure-env", action="store_true", help="Configure shell environment variables")
    parser.add_argument("--ldconfig", action="store_true", help="Configure system ldconfig")
    parser.add_argument("--install-monitor", action="store_true", help="Install performance monitoring tools")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--remove", action="store_true", help="Remove ldconfig configuration")

    # Options
    parser.add_argument("--bashrc", type=str, default="~/.bashrc", help="Shell rc file (default: ~/.bashrc)")
    parser.add_argument("--lib-dirs", nargs="+", help="Library directories for ldconfig")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")

    args = parser.parse_args()

    # If no action specified, show help
    if not any([args.setup_all, args.configure_env, args.ldconfig, args.install_monitor, args.show, args.remove]):
        parser.print_help()
        return

    print_info("TrtEnv - TensorRT Environment Setup Tool")
    print_info("")

    # Show configuration
    if args.show:
        show_configuration()
        return

    # Remove configuration
    if args.remove:
        configure_ldconfig(dry_run=args.dry_run, remove=True)
        return

    # Setup all
    if args.setup_all:
        args.configure_env = True
        args.ldconfig = True
        args.install_monitor = True

    success_count = 0
    total_count = 0

    # Configure shell environment
    if args.configure_env:
        total_count += 1
        if configure_shell_env(bashrc=args.bashrc, dry_run=args.dry_run):
            success_count += 1
        print()

    # Configure ldconfig
    if args.ldconfig:
        total_count += 1
        if configure_ldconfig(lib_dirs=args.lib_dirs, dry_run=args.dry_run):
            success_count += 1
        print()

    # Install monitoring tools
    if args.install_monitor:
        total_count += 1
        if install_monitoring_tool(dry_run=args.dry_run):
            success_count += 1
        print()

    # Summary
    if not args.dry_run:
        print_info("=" * 60)
        print_info(f"完成: {success_count}/{total_count} 个模块配置成功")
        print_info("=" * 60)

if __name__ == "__main__":
    main()
