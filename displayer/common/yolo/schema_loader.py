from common.utils.load_schema import KeyPoint, Skeleton


class SchemaLoader:
    """加载并管理关键点和骨骼映射的类。"""

    def __init__(self, schema_file: str = None):
        self.kpt_color_map = {}
        self.skeleton_map = []
        self.bbox_colors = []
        if schema_file:
            self.load_external_schema(schema_file)
        else:
            self.default_schema()

    def default_schema(self):
        """加载默认的关键点和骨骼映射。"""
        self.kpt_color_map = {
            0: KeyPoint("Nose", (0, 0, 255)),
            1: KeyPoint("Right Eye", (255, 0, 0)),
            2: KeyPoint("Left Eye", (255, 0, 0)),
            3: KeyPoint("Right Ear", (0, 255, 0)),
            4: KeyPoint("Left Ear", (0, 255, 0)),
            5: KeyPoint("Right Shoulder", (193, 182, 255)),
            6: KeyPoint("Left Shoulder", (193, 182, 255)),
            7: KeyPoint("Right Elbow", (16, 144, 247)),
            8: KeyPoint("Left Elbow", (16, 144, 247)),
            9: KeyPoint("Right Wrist", (1, 240, 255)),
            10: KeyPoint("Left Wrist", (1, 240, 255)),
            11: KeyPoint("Right Hip", (140, 47, 240)),
            12: KeyPoint("Left Hip", (140, 47, 240)),
            13: KeyPoint("Right Knee", (223, 155, 60)),
            14: KeyPoint("Left Knee", (223, 155, 60)),
            15: KeyPoint("Right Ankle", (139, 0, 0)),
            16: KeyPoint("Left Ankle", (139, 0, 0))
        }

        self.skeleton_map = [
            Skeleton(0, 1, (0, 0, 255)),  # Nose -> Right Eye
            Skeleton(0, 2, (0, 0, 255)),  # Nose -> Left Eye
            Skeleton(1, 3, (0, 0, 255)),  # Right Eye -> Right Ear
            Skeleton(2, 4, (0, 0, 255)),  # Left Eye -> Left Ear
            Skeleton(15, 13, (0, 100, 255)),  # Right Ankle -> Right Knee
            Skeleton(13, 11, (0, 255, 0)),  # Right Knee -> Right Hip
            Skeleton(16, 14, (255, 0, 0)),  # Left Ankle -> Left Knee
            Skeleton(14, 12, (0, 0, 255)),  # Left Knee -> Left Hip
            Skeleton(11, 12, (122, 160, 255)),  # Right Hip -> Left Hip
            Skeleton(5, 11, (139, 0, 139)),  # Right Shoulder -> Right Hip
            Skeleton(6, 12, (237, 149, 100)),  # Left Shoulder -> Left Hip
            Skeleton(5, 6, (152, 251, 152)),  # Right Shoulder -> Left Shoulder
            Skeleton(5, 7, (148, 0, 69)),  # Right Shoulder -> Right Elbow
            Skeleton(6, 8, (0, 75, 255)),  # Left Shoulder -> Left Elbow
            Skeleton(7, 9, (56, 230, 25)),  # Right Elbow -> Right Wrist
            Skeleton(8, 10, (0, 240, 240))  # Left Elbow -> Left Wrist
        ]

        self.bbox_colors = [
            (255, 0, 0),  # Class 0: Blue
            (0, 255, 0),  # Class 1: Green
            (0, 0, 255),  # Class 2: Red
            (255, 255, 0),  # Class 3: Cyan
            (255, 0, 255),  # Class 4: Magenta
            (0, 255, 255),  # Class 5: Yellow
            (128, 0, 128),  # Class 6: Purple
            (128, 128, 0),  # Class 7: Olive
            (128, 128, 128),  # Class 8: Gray
            (0, 128, 255)  # Class 9: Orange
        ]

    def load_external_schema(self, schema_file: str):
        """加载外部的关键点和骨骼映射。"""
        if not os.path.isfile(schema_file):
            raise FileNotFoundError("The schema file does not exist.")

        kpt_color_map, skeleton_map, bbox_colors = load_schema_from_json(schema_file)
        self.kpt_color_map = kpt_color_map
        self.skeleton_map = skeleton_map
        self.bbox_colors = bbox_colors
