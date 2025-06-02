import json

def load_build_dir(default_config_path: str ="/opt/TrtEngineToolkits/pyengine_config.json"):
    try:
        with open(default_config_path, 'r') as f: # 'r' 表示以只读模式打开文件
            configs = json.load(f)
        return configs["build_dir"]
    except FileNotFoundError:
        print(f"错误：配置文件 '{default_config_path}' 未找到。")
        return None # 或者抛出异常，或者返回一个默认值
    except json.JSONDecodeError:
        print(f"错误：配置文件 '{default_config_path}' 不是有效的 JSON 格式。")
        return None # 或者抛出异常
    except KeyError:
        print(f"错误：配置文件 '{default_config_path}' 中缺少 'build_dir' 键。")
        return None # 或者抛出异常
