import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

from pyengine.visualization.color_utils import bgr_to_hex, hex_to_bgr
from pyengine.utils.logger import logger
@dataclass
class KeyPointSchema:
    """数据类，用于存储关键点的名称和 BGR 颜色。"""
    name: str
    color: Tuple[int, int, int]  # Stored as BGR

@dataclass
class SkeletonSchema:
    """数据类，用于存储骨架连接的 ID 和 BGR 颜色。"""
    srt_kpt_id: int
    dst_kpt_id: int
    color: Tuple[int, int, int]  # Stored as BGR

# -----------------------------------------------------------

class SchemeLoader:
    """
    加载并管理关键点、骨骼、Bbox 和高亮颜色的类。
    [MODIFIED] 它会解析指定的 JSON 文件，并将所有十六进制颜色字符串转换为 BGR 格式。
    """

    def __init__(self, schema_file: str):
        self.kpt_color_map: Dict[int, KeyPointSchema] = {}
        self.skeleton_map: List[SkeletonSchema] = []
        self.bbox_colors: List[Tuple[int, int, int]] = []
        self.highlight_colors: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}

        self.load_external_schema(schema_file)
        logger.info("SchemeLoader", f"Successfully loaded and processed schema from '{schema_file}'")

    def load_external_schema(self, schema_file: str):
        if not os.path.isfile(schema_file):
            raise FileNotFoundError(f"Schema file does not exist: {schema_file}")
        try:
            with open(schema_file, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file '{schema_file}': {e}")
        except IOError as e:
            raise FileNotFoundError(f"Error reading file '{schema_file}': {e}")

        # --- [MODIFIED] Parse kpt_color_map ---
        kpt_map_data = json_data.get("kpt_color_map")
        if kpt_map_data and isinstance(kpt_map_data, dict):
            for key_str, item_data in kpt_map_data.items():
                try:
                    key_int = int(key_str)
                    name = item_data.get("name")
                    hex_color = item_data.get("color")
                    if name and isinstance(hex_color, str):
                        bgr_color = hex_to_bgr(hex_color)
                        self.kpt_color_map[key_int] = KeyPointSchema(name=name, color=bgr_color)
                    else:
                        logger.warning("SchemeLoader", f"Skipping invalid kpt_color_map item: key='{key_str}', data={item_data}")
                except (ValueError, TypeError) as e:
                    logger.warning("SchemeLoader", f"Error parsing kpt_color_map item: key='{key_str}', {e}")
        else:
            logger.warning("SchemeLoader", "JSON file does not contain a valid 'kpt_color_map'")

        # --- [MODIFIED] Parse skeleton_map ---
        skeleton_map_data = json_data.get("skeleton_map")
        if skeleton_map_data and isinstance(skeleton_map_data, list):
            for item_data in skeleton_map_data:
                try:
                    srt_id = item_data.get("srt_kpt_id")
                    dst_id = item_data.get("dst_kpt_id")
                    hex_color = item_data.get("color")
                    if isinstance(srt_id, int) and isinstance(dst_id, int) and isinstance(hex_color, str):
                        bgr_color = hex_to_bgr(hex_color)
                        self.skeleton_map.append(SkeletonSchema(
                            srt_kpt_id=srt_id, dst_kpt_id=dst_id, color=bgr_color
                        ))
                    else:
                        logger.warning("SchemeLoader", f"Skipping invalid skeleton_map item: {item_data}")
                except (TypeError, KeyError) as e:
                    logger.warning("SchemeLoader", f"Error parsing skeleton_map item: {item_data}, {e}")
        else:
            logger.warning("SchemeLoader", "JSON file does not contain a valid 'skeleton_map'")

        # --- [MODIFIED] Parse bbox_color ---
        bbox_color_data = json_data.get("bbox_color")
        if bbox_color_data and isinstance(bbox_color_data, list):
            for item_data in bbox_color_data:
                try:
                    hex_color = item_data.get("color")
                    if isinstance(hex_color, str):
                        bgr_color = hex_to_bgr(hex_color)
                        self.bbox_colors.append(bgr_color)
                    else:
                        logger.warning("SchemeLoader", f"Skipping invalid bbox_color item: {item_data}")
                except (TypeError, KeyError) as e:
                    logger.warning("SchemeLoader", f"Error parsing bbox_color item: {item_data}, {e}")
        else:
            logger.warning("SchemeLoader", "JSON file does not contain a valid 'bbox_color'")

        # --- [MODIFIED] Parse highlight_classes ---
        highlight_data = json_data.get("highlight_classes")
        if highlight_data and isinstance(highlight_data, list):
            for item_data in highlight_data:
                try:
                    key_name = item_data.get("name")
                    key_vals = item_data.get("value")
                    if key_name and isinstance(key_vals, list) and len(key_vals) == 2:
                        if isinstance(key_vals[0], str) and isinstance(key_vals[1], str):
                            bgr_color1 = hex_to_bgr(key_vals[0])
                            bgr_color2 = hex_to_bgr(key_vals[1])
                            self.highlight_colors[key_name] = (bgr_color1, bgr_color2)
                        else:
                            logger.warning("SchemeLoader", f"Skipping invalid highlight_classes item, format error: key='{key_name}'")
                    else:
                        logger.warning("SchemeLoader", f"Skipping invalid highlight_classes item, format error: key='{key_name}'")
                except (TypeError, IndexError, ValueError) as e:
                    logger.warning("SchemeLoader", f"Error parsing highlight_classes item: data='{item_data}', {e}")
        else:
            logger.warning("SchemeLoader", "JSON file does not contain a valid 'highlight_classes'")


# --- [MODIFIED] Example Usage ---
if __name__ == '__main__':
    # Create a dummy schema file with the new hex format
    schema_content = """
    {
    "kpt_color_map": {
        "0": { "name": "Nose_JSON", "color": "#FF0080" }, 
        "5": { "name": "R_Shoulder_JSON", "color": "#00FFFF" } 
    },
    "skeleton_map": [
        { "srt_kpt_id": 0, "dst_kpt_id": 5, "color": "#FFFFFF", "description": "Nose to R Shoulder" }
    ],
    "bbox_color" : [
        {"color": "#E60000", "name": "Red"},
        {"color": "#28323C", "name": "Dark Grayish"}
    ],
    "highlight_classes": [
        { "name": "red_white", "value": ["#FF0000", "#FFFFFF"] },
        { "name": "green_white", "value": ["#00FF00", "#FFFFFF"] }
    ]
    }
    """
    dummy_schema_file = "dummy_schema.json"
    with open(dummy_schema_file, "w", encoding='utf-8') as f:
        f.write(schema_content)

    logger.info("SchemeLoader", "--- Loading with dummy schema file ---")
    try:
        loader = SchemeLoader(dummy_schema_file)

        logger.info("SchemeLoader", "\nLoaded Keypoints (BGR format):")
        for idx, kp in loader.kpt_color_map.items():
            hex_color = bgr_to_hex(kp.color)
            logger.info("SchemeLoader", f"  {idx}: Name={kp.name}, Hex={hex_color} -> BGR={kp.color}")

        logger.info("SchemeLoader", "\nLoaded Skeletons (BGR format):")
        for sk in loader.skeleton_map:
            hex_color = bgr_to_hex(sk.color)
            logger.info("SchemeLoader", f"  {sk.srt_kpt_id} -> {sk.dst_kpt_id}, Hex={hex_color} -> BGR={sk.color}")

        logger.info("SchemeLoader", "\nLoaded Bbox Colors (BGR format):")
        for i, color in enumerate(loader.bbox_colors):
            hex_color = bgr_to_hex(color)
            logger.info("SchemeLoader", f"  {i}: Hex={hex_color} -> BGR={color}")

        logger.info("SchemeLoader", "\nLoaded Highlight Colors (BGR format):")
        for key, (color1, color2) in loader.highlight_colors.items():
            hex1 = bgr_to_hex(color1)
            hex2 = bgr_to_hex(color2)
            logger.info("SchemeLoader", f"  '{key}':")
            logger.info("SchemeLoader", f"    Color 1: Hex={hex1} -> BGR={color1}")
            logger.info("SchemeLoader", f"    Color 2: Hex={hex2} -> BGR={color2}")

    except Exception as e:
        logger.error("SchemeLoader", f"Error during loading test: {e}")
    finally:
        if os.path.exists(dummy_schema_file):
            os.remove(dummy_schema_file)
            logger.info("SchemeLoader", f"\n--- Dummy file '{dummy_schema_file}' removed ---")
