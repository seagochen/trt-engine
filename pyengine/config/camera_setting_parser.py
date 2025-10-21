# pyengine/config/camera_setting_parser.py
from typing import Tuple, List, Any
import yaml
from pydantic import BaseModel

# ---- 自定义 SafeLoader：识别 !!python/tuple ----
class _SafeLoader(yaml.SafeLoader):
    pass

def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node):
    # 把 !!python/tuple 转成 tuple（也可以换成 list 返回）
    return tuple(loader.construct_sequence(node))

_SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    _construct_python_tuple
)

class CameraParametersConfig(BaseModel):
    camera_height: int
    roll_angle: int
    pitch_angle: int
    yaw_angle: int
    resolution: Tuple[int, int]
    focal_length: Tuple[int, int]
    principal_coord: Tuple[int, int]
    # ground_coords: List[Tuple[int, int]]
    # ground_x_length_calculated: int
    # ground_y_length_calculated: int
    # ground_z_length_calculated: int

def load_camera_settings(path: str) -> CameraParametersConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw_config = yaml.load(f, Loader=_SafeLoader)  # 兼容 !!python/tuple
    return CameraParametersConfig.model_validate(raw_config)

def _tuples_to_lists(obj: Any) -> Any:
    """深度把 tuple 转成 list，避免 dump 出 !!python/tuple"""
    if isinstance(obj, tuple):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, list):
        return [_tuples_to_lists(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    return obj

def save_camera_settings(path: str, cfg: CameraParametersConfig) -> None:
    raw = cfg.model_dump()
    raw = _tuples_to_lists(raw)           # <- 统一转 list
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)
