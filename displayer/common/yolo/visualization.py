import os
import cv2
import numpy as np
from common.utils.load_schema import KeyPoint, Skeleton


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


def load_external_schema(schema_file: str):

    from common.utils.load_schema import load_schema_from_json

    # Check if the schema file exists
    if not os.path.isfile(schema_file):
        raise FileNotFoundError("The schema file does not exist.")

    # Update the global variables    
    global kpt_color_map, skeleton_map, bbox_colors
    kpt_color_map, skeleton_map, bbox_colors = load_schema_from_json(schema_file)


def draw_skeletons_with_bboxes(image, results: list, different_bbox=False, show_pts=True, show_names=True):
    """
    Draw skeletons on the image.

    :param image: cv2 image
    :param results: list of YoloPose objects
    :param different_bbox: True if different bounding box colors are used
    :param show_pts: True if key points are shown
    :param show_names: True if key point names are shown
    :return:
    """

    for idx, pose in enumerate(results):
        # Draw the key points
        if show_pts:
            for i, pt in enumerate(pose.pts):
                if pt.conf > 0.2 and i in kpt_color_map:
                    kp = kpt_color_map[i]
                    cv2.circle(image, (pt.x, pt.y), 3, kp.color, -1)

                    if show_names:
                        cv2.putText(image, kp.name, (pt.x, pt.y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, kp.color, 1)

        # Draw the skeleton
        for bone in skeleton_map:
            # Easy debug
            srt_kp = pose.pts[bone.srt_kpt_id]
            dst_kp = pose.pts[bone.dst_kpt_id]

            if srt_kp.conf > 0.2 and dst_kp.conf > 0.2 and \
                srt_kp.x > 0 and srt_kp.y > 0 and \
                    dst_kp.x > 0 and dst_kp.y > 0:
                cv2.line(image, (srt_kp.x, srt_kp.y), (dst_kp.x, dst_kp.y), bone.color, 2)
                
        # Determine the color of the bounding box
        if different_bbox:
            box_color = bbox_colors[idx % len(bbox_colors)]
        else:
            box_color = (255, 0, 0)

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


def draw_skeletons_without_bboxes(image, results: list, show_pts=True, show_names=True):
    """
    Draw skeletons on the image without bounding boxes.

    :param image: cv2 image
    :param results: list of YoloPose objects
    :param show_pts: True if key points are shown
    :param show_names: True if key point names are shown
    :return:
    """

    for pose in results:
        # Draw the key points
        if show_pts:
            for i, pt in enumerate(pose.pts):
                if pt.conf > 0.2 and i in kpt_color_map:
                    kp = kpt_color_map[i]
                    cv2.circle(image, (pt.x, pt.y), 3, kp.color, -1)

                    if show_names:
                        cv2.putText(image, kp.name, (pt.x, pt.y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, kp.color, 1)

        # Draw the skeleton
        for bone in skeleton_map:
            # Easy debug
            srt_kp = pose.pts[bone.srt_kpt_id]
            dst_kp = pose.pts[bone.dst_kpt_id]

            if srt_kp.conf > 0.2 and dst_kp.conf > 0.2 and \
                srt_kp.x > 0 and srt_kp.y > 0 and \
                    dst_kp.x > 0 and dst_kp.y > 0:
                cv2.line(image, (srt_kp.x, srt_kp.y), (dst_kp.x, dst_kp.y), bone.color, 2)

    return image


def draw_bboxes_with_labels(image, results: list, labels: list):
    """
    Draw bounding boxes with labels on the image.

    :param image: cv2 image
    :param results: list of Yolo objects
    :param labels: list of strings
    :return:
    """

    for yolo in results:
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


def draw_bboxes_without_labels(image, results: list):
    """
    Draw bounding boxes without labels on the image.

    :param image: cv2 image
    :param results: list of Yolo objects
    :return:
    """

    for yolo in results:
        lx, ly, rx, ry, cls, conf = yolo.lx, yolo.ly, yolo.rx, yolo.ry, yolo.cls, yolo.conf

        # Select color based on class
        box_color = bbox_colors[cls % len(bbox_colors)]

        # Draw the bounding box
        cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

    return image


def draw_facial_vectors_2d(frame, orientation_vectors: list, different_vectors=False, show_legend=False):
    """
    Draw facial orientation vectors on the image.

    :param frame: cv2 image
    :param orientation_vectors: list of FacialOrientation2D objects
    :param different_vectors: True if different vectors are used
    """

    for idx, vector in enumerate(orientation_vectors):
        # Determine the color of the vector
        if different_vectors:
            vector_color = bbox_colors[idx % len(bbox_colors)]
        else:
            vector_color = (255, 0, 0)

        # Draw the orientation vector
        cv2.arrowedLine(frame, (vector.origin_x, vector.origin_y), (vector.dest_x, vector.dest_y),
                        vector_color, 2)
        
        # Draw the legend
        if show_legend:
            face_direction = str(vector)

            # Put the legend text on the image
            cv2.putText(frame, face_direction, (vector.lx + 5, vector.ly + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, vector_color, 1)

    return frame


def draw_sort_bboxes(image, np_data: np.array, labels: list = None):
    """
    Draw bounding boxes with labels on the image.

    :param image: cv2 image
    :param np_data: Numpy array [id, lx, ly, rx, ry, conf, cls, ...], size [N, 58] or [N, 7]
    :param labels: list of strings
    :return:
    """

    # Iterate over each bounding box in np_data
    for i in range(np_data.shape[0]):
        # Get bounding box coordinates and convert them to integers
        lx, ly, rx, ry = map(int, np_data[i, 1:5])

        # Get the color for this bounding box based on its id
        box_color = bbox_colors[int(np_data[i, 0]) % len(bbox_colors)]

        # Determine the label text
        if labels is not None:
            label = f"{labels[int(np_data[i, 6])]}: {np_data[i, 5]:.2f}"
        else:
            label = f"ID: {int(np_data[i, 0])}: {np_data[i, 5]:.2f}"

        # Draw the bounding box
        cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

        # Get text size for background size
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle as background for text
        cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), box_color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Return the image with bounding boxes drawn
    return image