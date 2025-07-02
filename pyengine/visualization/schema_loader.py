import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class KeyPointSchema:
    name: str
    color: Tuple[int, int, int]

@dataclass
class SkeletonSchema:
    srt_kpt_id: int
    dst_kpt_id: int
    color: Tuple[int, int, int]
    # Optional description field if your class supports it
    # description: Optional[str] = None

# -----------------------------------------------------------

class SchemaLoader:
    """加载并管理关键点、骨骼映射和 Bbox 颜色的类，可解析指定的 JSON 结构。"""

    def __init__(self, schema_file: Optional[str] = None):
        # Type hints for clarity
        self.kpt_color_map: Dict[int, KeyPointSchema] = {}
        self.skeleton_map: List[SkeletonSchema] = []
        self.bbox_colors: List[Tuple[int, int, int]] = []

        if schema_file and os.path.exists(schema_file):
            try:
                self.load_external_schema(schema_file)
                print(f"Successfully loaded schema from: {schema_file}")
            except (ValueError, FileNotFoundError, KeyError) as e:
                print(f"Warning: Failed to load external schema '{schema_file}': {e}. Falling back to defaults.")
                self._load_defaults()
        else:
            if schema_file:
                 print(f"Warning: Schema file not found at '{schema_file}'. Falling back to defaults.")
            else:
                 print("No schema file provided. Loading defaults.")
            self._load_defaults()

    def _load_defaults(self):
        """Loads all default values."""
        self._load_default_kpt_color_map()
        self._load_default_skeleton_map()
        self._load_default_bbox_colors()

    def _load_default_kpt_color_map(self):
        # Default keypoint definitions
        self.kpt_color_map = {
            0: KeyPointSchema("Nose", (0, 0, 255)), 1: KeyPointSchema("Right Eye", (255, 0, 0)),
            2: KeyPointSchema("Left Eye", (255, 0, 0)), 3: KeyPointSchema("Right Ear", (0, 255, 0)),
            4: KeyPointSchema("Left Ear", (0, 255, 0)), 5: KeyPointSchema("Right Shoulder", (193, 182, 255)),
            6: KeyPointSchema("Left Shoulder", (193, 182, 255)), 7: KeyPointSchema("Right Elbow", (16, 144, 247)),
            8: KeyPointSchema("Left Elbow", (16, 144, 247)), 9: KeyPointSchema("Right Wrist", (1, 240, 255)),
            10: KeyPointSchema("Left Wrist", (1, 240, 255)), 11: KeyPointSchema("Right Hip", (140, 47, 240)),
            12: KeyPointSchema("Left Hip", (140, 47, 240)), 13: KeyPointSchema("Right Knee", (223, 155, 60)),
            14: KeyPointSchema("Left Knee", (223, 155, 60)), 15: KeyPointSchema("Right Ankle", (139, 0, 0)),
            16: KeyPointSchema("Left Ankle", (139, 0, 0))
        }
        print(f"Loaded {len(self.kpt_color_map)} default keypoints.")

    def _load_default_skeleton_map(self):
         # Default skeleton definitions
        self.skeleton_map = [
            SkeletonSchema(0, 1, (0, 0, 255)), SkeletonSchema(0, 2, (0, 0, 255)),
            SkeletonSchema(1, 3, (0, 0, 255)), SkeletonSchema(2, 4, (0, 0, 255)),
            SkeletonSchema(15, 13, (0, 100, 255)), SkeletonSchema(13, 11, (0, 255, 0)),
            SkeletonSchema(16, 14, (255, 0, 0)), SkeletonSchema(14, 12, (0, 0, 255)),
            SkeletonSchema(11, 12, (122, 160, 255)), SkeletonSchema(5, 11, (139, 0, 139)),
            SkeletonSchema(6, 12, (237, 149, 100)), SkeletonSchema(5, 6, (152, 251, 152)),
            SkeletonSchema(5, 7, (148, 0, 69)), SkeletonSchema(6, 8, (0, 75, 255)),
            SkeletonSchema(7, 9, (56, 230, 25)), SkeletonSchema(8, 10, (0, 240, 240))
        ]
        print(f"Loaded {len(self.skeleton_map)} default skeleton links.")

    def _load_default_bbox_colors(self):
        # Default Bbox color list (tuples)
        self.bbox_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (128, 128, 0), (128, 128, 128),
            (0, 128, 255) # BGR: Blue, Green, Red, Cyan, Magenta, Yellow, Purple, Olive, Gray, Orange
        ]
        print(f"Loaded {len(self.bbox_colors)} default bbox colors.")

    # --- MODIFIED METHOD TO PARSE THE SPECIFIC JSON ---
    def load_external_schema(self, schema_file: str):
        """
        Loads keypoint, skeleton, and bbox color data from the specified JSON file format.
        """
        if not os.path.isfile(schema_file):
            raise FileNotFoundError(f"The schema file does not exist: {schema_file}")

        try:
            with open(schema_file, 'r', encoding='utf-8') as file: # Added encoding
                json_data = json.load(file) # Use json.load directly
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file '{schema_file}': {e}")
        except IOError as e:
            raise FileNotFoundError(f"Error reading file '{schema_file}': {e}")

        # --- Parse kpt_color_map ---
        kpt_map_data = json_data.get("kpt_color_map")
        if kpt_map_data and isinstance(kpt_map_data, dict):
            parsed_kpt_map = {}
            for key_str, item_data in kpt_map_data.items():
                try:
                    key_int = int(key_str) # Convert string key "0", "1" to int
                    name = item_data.get("name")
                    color_list = item_data.get("color")
                    if name and isinstance(color_list, list) and len(color_list) == 3:
                        # Create KeyPoint object (assuming this structure)
                        parsed_kpt_map[key_int] = KeyPointSchema(name=name, color=tuple(color_list))
                    else:
                        print(f"Warning: Skipping invalid kpt_color_map item with key '{key_str}': {item_data}")
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error parsing kpt_color_map item with key '{key_str}': {e}")
            if parsed_kpt_map:
                self.kpt_color_map = parsed_kpt_map
            else:
                 print("Warning: No valid keypoints found in kpt_color_map from JSON. Keeping defaults.")
                 self._load_default_kpt_color_map() # Fallback if parsing fails
        else:
             print("Warning: 'kpt_color_map' not found or invalid in JSON. Loading defaults.")
             self._load_default_kpt_color_map() # Fallback if key missing

        # --- Parse skeleton_map ---
        skeleton_map_data = json_data.get("skeleton_map")
        if skeleton_map_data and isinstance(skeleton_map_data, list):
            parsed_skeleton_map = []
            for item_data in skeleton_map_data:
                 try:
                     srt_id = item_data.get("srt_kpt_id")
                     dst_id = item_data.get("dst_kpt_id")
                     color_list = item_data.get("color")
                     # description = item_data.get("description") # Optional

                     if (isinstance(srt_id, int) and isinstance(dst_id, int) and
                         isinstance(color_list, list) and len(color_list) == 3):
                         # Create Skeleton object (assuming this structure)
                         parsed_skeleton_map.append(SkeletonSchema(
                             srt_kpt_id=srt_id,
                             dst_kpt_id=dst_id,
                             color=tuple(color_list)
                             # description=description # Add if your class uses it
                         ))
                     else:
                         print(f"Warning: Skipping invalid skeleton_map item: {item_data}")
                 except (TypeError, KeyError) as e:
                     print(f"Warning: Error parsing skeleton_map item {item_data}: {e}")
            if parsed_skeleton_map:
                self.skeleton_map = parsed_skeleton_map
            else:
                 print("Warning: No valid skeleton links found in skeleton_map from JSON. Keeping defaults.")
                 self._load_default_skeleton_map() # Fallback if parsing fails
        else:
            print("Warning: 'skeleton_map' not found or invalid in JSON. Loading defaults.")
            self._load_default_skeleton_map() # Fallback if key missing


        # --- Parse bbox_color ---
        bbox_color_data = json_data.get("bbox_color")
        if bbox_color_data and isinstance(bbox_color_data, list):
            parsed_bbox_colors = []
            for item_data in bbox_color_data:
                try:
                    color_list = item_data.get("color")
                    # name = item_data.get("name") # Optional, not used in current implementation

                    if isinstance(color_list, list) and len(color_list) == 3:
                        # Append the color tuple directly
                        parsed_bbox_colors.append(tuple(color_list))
                    else:
                        print(f"Warning: Skipping invalid bbox_color item: {item_data}")
                except (TypeError, KeyError) as e:
                     print(f"Warning: Error parsing bbox_color item {item_data}: {e}")
            if parsed_bbox_colors:
                self.bbox_colors = parsed_bbox_colors
            else:
                 print("Warning: No valid bbox colors found in bbox_color from JSON. Keeping defaults.")
                 self._load_default_bbox_colors() # Fallback if parsing fails
        else:
            print("Warning: 'bbox_color' not found or invalid in JSON. Loading defaults.")
            self._load_default_bbox_colors() # Fallback if key missing


# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy schema.json for testing
    schema_content = """
{
  "kpt_color_map": {
    "0": { "name": "Nose_JSON", "color": [255, 0, 255] },
    "5": { "name": "R_Shoulder_JSON", "color": [0, 255, 255] }
  },
  "skeleton_map": [
    { "srt_kpt_id": 0, "dst_kpt_id": 5, "color": [255, 255, 255], "description": "Nose to R Shoulder" }
  ],
  "bbox_color" : [
    {"color": [10, 20, 30], "name": "Dark Blueish"},
    {"color": [40, 50, 60], "name": "Dark Greenish"}
  ]
}
"""
    dummy_schema_file = "dummy_schema.json"
    with open(dummy_schema_file, "w", encoding='utf-8') as f:
        f.write(schema_content)

    print("--- Loading with Dummy Schema ---")
    try:
        loader_external = SchemaLoader(dummy_schema_file)
        print("\nLoaded Keypoints (External):")
        for idx, kp in loader_external.kpt_color_map.items():
            print(f"  {idx}: Name={kp.name}, Color={kp.color}")

        print("\nLoaded Skeletons (External):")
        for sk in loader_external.skeleton_map:
            print(f"  {sk.srt_kpt_id} -> {sk.dst_kpt_id}, Color={sk.color}")

        print("\nLoaded Bbox Colors (External):")
        for i, color in enumerate(loader_external.bbox_colors):
            print(f"  {i}: {color}")

    except Exception as e:
        print(f"Error during external load test: {e}")
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_schema_file):
            os.remove(dummy_schema_file)

    print("\n--- Loading with Defaults (File Not Found) ---")
    try:
        loader_default = SchemaLoader("non_existent_schema.json")
        print("\nLoaded Keypoints (Default):")
        # Print a few defaults
        for i in range(3):
             if i in loader_default.kpt_color_map:
                 kp = loader_default.kpt_color_map[i]
                 print(f"  {i}: Name={kp.name}, Color={kp.color}")
        print("\nLoaded Bbox Colors (Default):")
        # Print a few defaults
        for i in range(3):
             if i < len(loader_default.bbox_colors):
                 print(f"  {i}: {loader_default.bbox_colors[i]}")

    except Exception as e:
         print(f"Error during default load test: {e}")