import multiprocessing
import os
import signal
import sys

def search_config_files(folder):
    # 获取 configs 目录下的所有配置文件
    config_files = os.listdir(folder)
    return config_files

def run_process(config_name):
    command = f"./adapter configs/{config_name}"
    os.system(command)

def signal_handler(sig, frame):
    print("Caught Ctrl+C! Terminating all processes...")
    pool.terminate()  # 终止所有进程
    pool.join()       # 确保所有进程都终止
    sys.exit(0)       # 退出程序

if __name__ == "__main__":

    # 注册 Ctrl+C 信号处理程序
    signal.signal(signal.SIGINT, signal_handler)

    # 获取 configs 目录下的所有配置文件
    configs = search_config_files('configs')

    try:
        # 创建一个进程池
        pool = multiprocessing.Pool(processes=8)

        # 启动 8 个进程，每个进程执行 run_process 函数
        pool.map(run_process, configs)
    except KeyboardInterrupt:
        # 捕获 Ctrl+C，确保池被正确终止
        print("Caught KeyboardInterrupt, terminating processes...")
        pool.terminate()
        pool.join()
    finally:
        pool.close()
        pool.join()
