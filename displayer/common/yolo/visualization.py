import cv2
from common.yolo.simple_structs import Yolo, YoloPose, YoloPoint

# 定义关键点结构
class KeyPoint:
    def __init__(self, name, color):
        self.name = name
        self.color = color

# 定义骨骼连接结构
class Skeleton:
    def __init__(self, srt_kpt_id, dst_kpt_id, color):
        self.srt_kpt_id = srt_kpt_id
        self.dst_kpt_id = dst_kpt_id
        self.color = color

# 定义关键点和骨骼映射
kpt_color_map = {
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

skeleton_map = [
    Skeleton(0, 1, (0, 0, 255)),  # Nose -> Right Eye
    Skeleton(0, 2, (0, 0, 255)),  # Nose -> Left Eye
    Skeleton(1, 3, (0, 0, 255)),  # Right Eye -> Right Ear
    Skeleton(2, 4, (0, 0, 255)),  # Left Eye -> Left Ear
    Skeleton(15, 13, (0, 100, 255)),  # Right Ankle -> Right Knee
    Skeleton(13, 11, (0, 255, 0)),    # Right Knee -> Right Hip
    Skeleton(16, 14, (255, 0, 0)),    # Left Ankle -> Left Knee
    Skeleton(14, 12, (0, 0, 255)),    # Left Knee -> Left Hip
    Skeleton(11, 12, (122, 160, 255)),  # Right Hip -> Left Hip
    Skeleton(5, 11, (139, 0, 139)),   # Right Shoulder -> Right Hip
    Skeleton(6, 12, (237, 149, 100)), # Left Shoulder -> Left Hip
    Skeleton(5, 6, (152, 251, 152)),  # Right Shoulder -> Left Shoulder
    Skeleton(5, 7, (148, 0, 69)),     # Right Shoulder -> Right Elbow
    Skeleton(6, 8, (0, 75, 255)),     # Left Shoulder -> Left Elbow
    Skeleton(7, 9, (56, 230, 25)),    # Right Elbow -> Right Wrist
    Skeleton(8, 10, (0, 240, 240))    # Left Elbow -> Left Wrist
]

# Define a list of colors for different classes (for simplicity, we assume 10 classes)
bbox_colors = [
    (255, 0, 0),    # Class 0: Blue
    (0, 255, 0),    # Class 1: Green
    (0, 0, 255),    # Class 2: Red
    (255, 255, 0),  # Class 3: Cyan
    (255, 0, 255),  # Class 4: Magenta
    (0, 255, 255),  # Class 5: Yellow
    (128, 0, 128),  # Class 6: Purple
    (128, 128, 0),  # Class 7: Olive
    (128, 128, 128),# Class 8: Gray
    (0, 128, 255)   # Class 9: Orange
]


def draw_skeletons(image, results: str, different_bbox=False, show_pts=True, show_names=True):
    yolo_pose_results = YoloPose.from_json(results)
    for idx, pose in enumerate(yolo_pose_results):

        # Draw the keypoints
        if show_pts:
            for i, pt in enumerate(pose.pts):
                if pt.conf > 0.2 and i in kpt_color_map:
                    kp = kpt_color_map[i]
                    cv2.circle(image, (pt.x, pt.y), 3, kp.color, -1)

                    if show_names:
                        cv2.putText(image, kp.name, (pt.x, pt.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, kp.color, 1)

        # Draw the skeleton
        for bone in skeleton_map:
            if pose.pts[bone.srt_kpt_id].conf > 0.2 and pose.pts[bone.dst_kpt_id].conf > 0.2:
                cv2.line(image,
                         (pose.pts[bone.srt_kpt_id].x, pose.pts[bone.srt_kpt_id].y),
                         (pose.pts[bone.dst_kpt_id].x, pose.pts[bone.dst_kpt_id].y),
                         bone.color, 2)
                
        # Determine the color of the bounding box
        if different_bbox:
            box_color = bbox_colors[idx % len(bbox_colors)]
        else:
            box_color = (255, 255, 255)

        # Draw the bounding box
        lx, ly, rx, ry = pose.lx, pose.ly, pose.rx, pose.ry
        cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

        # Create the label text with confidence
        label = f"Person: {pose.conf:.2f}"

        # Get text size for background size
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle as background for text
        cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), box_color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def draw_boxes_with_labels(image, results: str, labels: list):
    yolo_results = Yolo.from_json(results)
    for yolo in yolo_results:
        lx, ly, rx, ry, cls, conf = yolo.lx, yolo.ly, yolo.rx, yolo.ry, yolo.cls, yolo.conf

        # Select color based on class
        box_color = bbox_colors[cls % len(bbox_colors)]

        # Draw the bounding box
        cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

        # Create the label text with confidence
        label = "{}: {:.2f}".format(labels[cls], conf)

        # Get text size for background size
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle as background for text
        cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), box_color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image
