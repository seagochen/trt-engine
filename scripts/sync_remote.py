#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import paramiko
import os
import sys
import stat
import json
from getpass import getpass

# ==============================
# 配置部分（从配置文件读取）
# ==============================

CONFIG_FILE = "./scripts/sync_remote_config.json"

def load_config(config_file):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在，请检查路径。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        sys.exit(1)

# 读取配置
config = load_config(CONFIG_FILE)

REMOTE_USER = config.get("REMOTE_USER", "ubuntu")
SSH_KEY_PATH = config.get("SSH_KEY_PATH", None)
LOCAL_PROJECT_DIR = config.get("LOCAL_PROJECT_DIR", ".")
FILES_TO_SYNC = config.get("FILES_TO_SYNC", [])
TARGETS = config.get("TARGETS", [])

# ==============================
# 脚本逻辑部分
# ==============================

def get_password():
    """获取SSH密码"""
    return getpass(f"请输入 {REMOTE_USER} 的SSH密码: ")

def connect_ssh(username, hostname, password=None, key_filename=None):
    """建立SSH连接"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if key_filename:
            client.connect(hostname, username=username, key_filename=key_filename)
        else:
            client.connect(hostname, username=username, password=password)
        return client
    except Exception as e:
        print(f"无法连接到 {hostname}: {e}")
        sys.exit(1)

def sftp_delete_remote_item(sftp, remote_path):
    """删除远程路径"""
    try:
        file_attr = sftp.stat(remote_path)
        if stat.S_ISDIR(file_attr.st_mode):
            # 如果是目录，递归删除其内容
            for entry in sftp.listdir_attr(remote_path):
                entry_path = os.path.join(remote_path, entry.filename).replace("\\", "/")
                sftp_delete_remote_item(sftp, entry_path)
            sftp.rmdir(remote_path)
            print(f"删除远程目录: {remote_path}")
        else:
            sftp.remove(remote_path)
            print(f"删除远程文件: {remote_path}")
    except FileNotFoundError:
        print(f"远程路径 {remote_path} 不存在，跳过删除。")
    except Exception as e:
        print(f"删除远程路径 {remote_path} 失败: {e}")

def sftp_mkdirs(sftp, remote_path):
    """递归创建远程目录"""
    dirs = []
    while len(remote_path) > 1:
        dirs.append(remote_path)
        remote_path, _ = os.path.split(remote_path)
    dirs = dirs[::-1]
    for directory in dirs:
        try:
            sftp.stat(directory)
        except IOError:
            try:
                sftp.mkdir(directory)
                print(f"创建远程目录: {directory}")
            except Exception as e:
                print(f"无法创建目录 {directory}: {e}")
                sys.exit(1)

def sftp_put_dir(sftp, local_dir, remote_dir):
    """上传目录"""
    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        rel_path = "" if rel_path == "." else rel_path
        remote_path = os.path.join(remote_dir, rel_path).replace("\\", "/")
        sftp_mkdirs(sftp, remote_path)
        for file in files:
            local_file = os.path.join(root, file)
            remote_file = os.path.join(remote_path, file).replace("\\", "/")
            try:
                sftp.put(local_file, remote_file)
                print(f"上传文件: {local_file} -> {remote_file}")
            except Exception as e:
                print(f"无法上传文件 {local_file}: {e}")

def sftp_put_file(sftp, local_file, remote_file):
    """上传文件"""
    remote_dir = os.path.dirname(remote_file)
    sftp_mkdirs(sftp, remote_dir)
    try:
        sftp.put(local_file, remote_file)
        print(f"上传文件: {local_file} -> {remote_file}")
    except Exception as e:
        print(f"无法上传文件 {local_file}: {e}")

def sync_target(target):
    """同步到单个目标"""
    remote_host = target.get("REMOTE_HOST")
    remote_dir = target.get("REMOTE_DIR")

    if not remote_host or not remote_dir:
        print("目标配置缺少 REMOTE_HOST 或 REMOTE_DIR，跳过此目标。")
        return

    print(f"开始同步到目标: {remote_host}:{remote_dir}")

    # 获取SSH密码
    password = None
    if not SSH_KEY_PATH:
        password = get_password()

    # 建立SSH连接
    ssh_client = connect_ssh(REMOTE_USER, remote_host, password=password, key_filename=SSH_KEY_PATH)
    try:
        sftp = ssh_client.open_sftp()
    except Exception as e:
        print(f"无法打开SFTP连接: {e}")
        ssh_client.close()
        return

    # 删除并同步文件/文件夹
    for item in FILES_TO_SYNC:
        local_path = os.path.join(LOCAL_PROJECT_DIR, item.strip("/"))
        remote_path = os.path.join(remote_dir, item.strip("/")).replace("\\", "/")

        if item.endswith("/"):
            # 处理文件夹
            if not os.path.isdir(local_path):
                print(f"本地文件夹 {local_path} 不存在，跳过。")
                continue
            sftp_delete_remote_item(sftp, remote_path)
            sftp_put_dir(sftp, local_path, remote_path)
        else:
            # 处理单个文件
            if not os.path.isfile(local_path):
                print(f"本地文件 {local_path} 不存在，跳过。")
                continue
            sftp_delete_remote_item(sftp, remote_path)
            sftp_put_file(sftp, local_path, remote_path)

    # 关闭连接
    sftp.close()
    ssh_client.close()

    print(f"同步到目标 {remote_host}:{remote_dir} 完成！")

def main():
    if not TARGETS:
        print("配置中未定义任何同步目标，退出。")
        sys.exit(1)

    for target in TARGETS:
        sync_target(target)

    print("所有同步任务完成！")

if __name__ == "__main__":
    main()
