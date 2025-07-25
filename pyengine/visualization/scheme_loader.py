import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

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

class SchemaLoader:
    """
    加载并管理关键点、骨骼、Bbox 和高亮颜色的类。
    它会解析指定的 JSON 文件，并将所有颜色从 RGB 格式转换为 BGR 格式。
    """

    def __init__(self, schema_file: str):
        """
        初始化加载器。

        Args:
            schema_file (str): schema JSON 文件的路径。

        Raises:
            FileNotFoundError: 如果 schema 文件不存在。
            ValueError: 如果 JSON 文件格式无效。
        """
        # 为属性添加类型提示
        self.kpt_color_map: Dict[int, KeyPointSchema] = {}
        self.skeleton_map: List[SkeletonSchema] = []
        self.bbox_colors: List[Tuple[int, int, int]] = []
        self.highlight_colors: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {}

        self.load_external_schema(schema_file)
        # print(f"✅ 成功从 '{schema_file}' 加载并处理了 Schema。")
        print(f"✅ Successfully loaded and processed schema from '{schema_file}'.")

    def load_external_schema(self, schema_file: str):
        """
        从指定的 JSON 文件加载所有 schema 数据，并将颜色从 RGB 转换为 BGR。
        """
        if not os.path.isfile(schema_file):
            # raise FileNotFoundError(f"Schema 文件不存在: {schema_file}")
            raise FileNotFoundError(f"Schema file does not exist: {schema_file}")
        try:
            with open(schema_file, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file '{schema_file}': {e}")
            # raise ValueError(f"解析 JSON 文件 '{schema_file}' 出错: {e}")
        except IOError as e:
            raise FileNotFoundError(f"Error reading file '{schema_file}': {e}")
            # raise FileNotFoundError(f"读取文件 '{schema_file}' 出错: {e}")

        # --- 解析 kpt_color_map ---
        kpt_map_data = json_data.get("kpt_color_map")
        if kpt_map_data and isinstance(kpt_map_data, dict):
            for key_str, item_data in kpt_map_data.items():
                try:
                    key_int = int(key_str)
                    name = item_data.get("name")
                    rgb_color = item_data.get("color")
                    if name and isinstance(rgb_color, list) and len(rgb_color) == 3:
                        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                        self.kpt_color_map[key_int] = KeyPointSchema(name=name, color=bgr_color)
                    else:
                        print(f"⚠️ Warning: Skipping invalid kpt_color_map item: key='{key_str}', data={item_data}")
                        # print(f"⚠️ 警告: 跳过无效的 kpt_color_map 项目: key='{key_str}'")
                except (ValueError, TypeError) as e:
                    # print(f"⚠️ 警告: 解析 kpt_color_map 项目时出错: key='{key_str}', {e}")
                    print(f"⚠️ Warning: Error parsing kpt_color_map item: key='{key_str}', {e}")
        else:
            # print("⚠️ 警告: JSON 文件中未找到或无效的 'kpt_color_map'。")
            print("⚠️ Warning: JSON file does not contain a valid 'kpt_color_map'.")

        # --- 解析 skeleton_map ---
        skeleton_map_data = json_data.get("skeleton_map")
        if skeleton_map_data and isinstance(skeleton_map_data, list):
            for item_data in skeleton_map_data:
                try:
                    srt_id = item_data.get("srt_kpt_id")
                    dst_id = item_data.get("dst_kpt_id")
                    rgb_color = item_data.get("color")
                    if (isinstance(srt_id, int) and isinstance(dst_id, int) and
                            isinstance(rgb_color, list) and len(rgb_color) == 3):
                        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                        self.skeleton_map.append(SkeletonSchema(
                            srt_kpt_id=srt_id, dst_kpt_id=dst_id, color=bgr_color
                        ))
                    else:
                        # print(f"⚠️ 警告: 跳过无效的 skeleton_map 项目: {item_data}")
                        print(f"⚠️ Warning: Skipping invalid skeleton_map item: {item_data}")
                except (TypeError, KeyError) as e:
                    # print(f"⚠️ 警告: 解析 skeleton_map 项目时出错: {item_data}, {e}")
                    print(f"⚠️ Warning: Error parsing skeleton_map item: {item_data}, {e}")
        else:
            # print("⚠️ 警告: JSON 文件中未找到或无效的 'skeleton_map'。")
            print("⚠️ Warning: JSON file does not contain a valid 'skeleton_map'.")

        # --- 解析 bbox_color ---
        bbox_color_data = json_data.get("bbox_color")
        if bbox_color_data and isinstance(bbox_color_data, list):
            for item_data in bbox_color_data:
                try:
                    rgb_color = item_data.get("color")
                    if isinstance(rgb_color, list) and len(rgb_color) == 3:
                        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                        self.bbox_colors.append(bgr_color)
                    else:
                        # print(f"⚠️ 警告: 跳过无效的 bbox_color 项目: {item_data}")
                        print(f"⚠️ Warning: Skipping invalid bbox_color item: {item_data}")
                except (TypeError, KeyError) as e:
                    # print(f"⚠️ 警告: 解析 bbox_color 项目时出错: {item_data}, {e}")
                    print(f"⚠️ Warning: Error parsing bbox_color item: {item_data}, {e}")
        else:
            # print("⚠️ 警告: JSON 文件中未找到或无效的 'bbox_color'。")
            print("⚠️ Warning: JSON file does not contain a valid 'bbox_color'.")

        # --- 新增: 解析 highlight_classes ---
        """
          "highlight_classes": [
                { "name": "red_white", "value": [[255, 0, 0], [255, 255, 255]] },
                { "name": "blue_white", "value": [[0, 0, 255], [255, 255, 255]] },
                { "name": "green_white", "value": [[0, 255, 0], [255, 255, 255]] },
                { "name": "yellow_white", "value": [[255, 255, 0], [255, 255, 255]] },
                { "name": "cyan_white", "value": [[0, 255, 255], [255, 255, 255]] },
                { "name": "magenta_white", "value": [[255, 0, 255], [255, 255, 255]] }
            ]
        """

        highlight_data = json_data.get("highlight_classes")
        if highlight_data and isinstance(highlight_data, list):
            for item_data in highlight_data:
                try:
                    key_name = item_data.get("name")
                    key_vals = item_data.get("value")

                    if key_name and isinstance(key_vals, list) and len(key_vals) == 2:
                        if (isinstance(key_vals[0], list) and len(key_vals[0]) == 3 and
                                isinstance(key_vals[1], list) and len(key_vals[1]) == 3):
                            
                            rgb_color1 = key_vals[0]
                            rgb_color2 = key_vals[1]

                            # 将两个颜色都从 RGB 转换为 BGR
                            bgr_color1 = (rgb_color1[2], rgb_color1[1], rgb_color1[0])
                            bgr_color2 = (rgb_color2[2], rgb_color2[1], rgb_color2[0])
                            
                            self.highlight_colors[key_name] = (bgr_color1, bgr_color2)
                        else:
                            # print(f"⚠️ 警告: 跳过无效的 highlight_classes 项目，格式错误: key='{key_name}'")
                            print(f"⚠️ Warning: Skipping invalid highlight_classes item, format error: key='{key_name}'")
                    else:
                        # print(f"⚠️ 警告: 跳过无效的 highlight_classes 项目，格式错误: key='{key_name}'")
                        print(f"⚠️ Warning: Skipping invalid highlight_classes item, format error: key='{key_name}'")


                except (TypeError, IndexError) as e:
                    print(f"⚠️ Warning: Error parsing highlight_classes item: key='{key}', {e}")

            # for key, value in highlight_data.items():
            #     try:
            #         if (isinstance(value, list) and len(value) == 2 and
            #                 isinstance(value[0], list) and len(value[0]) == 3 and
            #                 isinstance(value[1], list) and len(value[1]) == 3):
                        
            #             rgb_color1 = value[0]
            #             rgb_color2 = value[1]

            #             # 将两个颜色都从 RGB 转换为 BGR
            #             bgr_color1 = (rgb_color1[2], rgb_color1[1], rgb_color1[0])
            #             bgr_color2 = (rgb_color2[2], rgb_color2[1], rgb_color2[0])
                        
            #             self.highlight_colors[key] = (bgr_color1, bgr_color2)
            #         else:
            #             # print(f"⚠️ 警告: 跳过无效的 highlight_classes 项目，格式错误: key='{key}'")
            #             print(f"⚠️ Warning: Skipping invalid highlight_classes item, format error: key='{key}'")
            #     except (TypeError, IndexError) as e:
            #         # print(f"⚠️ 警告: 解析 highlight_classes 项目时出错: key='{key}', {e}")
            #         print(f"⚠️ Warning: Error parsing highlight_classes item: key='{key}', {e}")
        else:
            # print("⚠️ 警告: JSON 文件中未找到或无效的 'highlight_classes'。")
            print("⚠️ Warning: JSON file does not contain a valid 'highlight_classes'.")


# --- 示例用法 ---
if __name__ == '__main__':
    # 创建一个包含所有部分的虚拟 scheme.json 文件
    schema_content = """
    {
    "kpt_color_map": {
        "0": { "name": "Nose_JSON", "color": [255, 0, 128] }, 
        "5": { "name": "R_Shoulder_JSON", "color": [0, 255, 255] } 
    },
    "skeleton_map": [
        { "srt_kpt_id": 0, "dst_kpt_id": 5, "color": [255, 255, 255], "description": "Nose to R Shoulder" }
    ],
    "bbox_color" : [
        {"color": [230, 0, 0], "name": "Red_is_Blue_in_BGR"},
        {"color": [40, 50, 60], "name": "Dark Grayish"}
    ],
    "highlight_classes": {
        "red_white": [[255, 0, 0], [255, 255, 255]],
        "green_white": [[0, 255, 0], [255, 255, 255]]
    }
    }
    """
    dummy_schema_file = "dummy_schema.json"
    with open(dummy_schema_file, "w", encoding='utf-8') as f:
        f.write(schema_content)

    print("--- 正在使用虚拟 Schema 文件进行加载 ---")
    try:
        loader = SchemaLoader(dummy_schema_file)
        
        print("\n🎨 已加载的关键点 (BGR 格式):")
        for idx, kp in loader.kpt_color_map.items():
            original_rgb = (kp.color[2], kp.color[1], kp.color[0])
            print(f"  {idx}: Name={kp.name}, RGB={original_rgb} -> BGR={kp.color}")

        print("\n🦴 已加载的骨架 (BGR 格式):")
        for sk in loader.skeleton_map:
            original_rgb = (sk.color[2], sk.color[1], sk.color[0])
            print(f"  {sk.srt_kpt_id} -> {sk.dst_kpt_id}, RGB={original_rgb} -> BGR={sk.color}")

        print("\n🔲 已加载的 Bbox 颜色 (BGR 格式):")
        for i, color in enumerate(loader.bbox_colors):
            original_rgb = (color[2], color[1], color[0])
            print(f"  {i}: RGB={original_rgb} -> BGR={color}")
        
        print("\n✨ 已加载的高亮颜色 (BGR 格式):")
        for key, (color1, color2) in loader.highlight_colors.items():
            original_rgb1 = (color1[2], color1[1], color1[0])
            original_rgb2 = (color2[2], color2[1], color2[0])
            print(f"  '{key}':")
            print(f"    Color 1: RGB={original_rgb1} -> BGR={color1}")
            print(f"    Color 2: RGB={original_rgb2} -> BGR={color2}")

    except Exception as e:
        print(f"❌ 在加载测试中发生错误: {e}")
    finally:
        if os.path.exists(dummy_schema_file):
            os.remove(dummy_schema_file)
            print(f"\n--- 已删除虚拟文件 '{dummy_schema_file}' ---")