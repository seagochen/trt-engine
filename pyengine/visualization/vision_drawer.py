import time
from typing import Union, Tuple, Dict, List

import cv2
import numpy as np

from pyengine.inference.extend.yolo.data_struct import YoloPose, Yolo, YoloPoseSorted, YoloSorted
from pyengine.inference.extend.yolo import FacialDirection
from pyengine.visualization.schema_loader import SchemaLoader


class VisionDrawer:
    """
    Refactored drawing utility class for visualizing tracking objects, detections,
    skeletons, supplementary info, and pose actions with improved control and clarity.
    """

    def __init__(self,
                 schema_loader: SchemaLoader,
                 object_conf_threshold: float = 0.25,
                 point_conf_threshold: float = 0.25,
                 flashing_frequency_hz: float = 1.0):
        """
        Initializes the VisionDrawer.

        Args:
            schema_loader: Instance of SchemaLoader containing color maps and skeleton definitions.
            object_conf_threshold: Minimum confidence threshold for drawing object bounding boxes.
            point_conf_threshold: Minimum confidence threshold for drawing keypoints.
            flashing_frequency_hz: Frequency in Hz for alternating colors in flashing mode.
        """
        self.kpt_color_map = schema_loader.kpt_color_map
        self.skeleton_map = schema_loader.skeleton_map
        self.bbox_colors = schema_loader.bbox_colors # Color palette for chromatic style

        self.object_conf_threshold = object_conf_threshold
        self.point_conf_threshold = point_conf_threshold

        if flashing_frequency_hz <= 0:
            self.flashing_interval = float('inf') # Effectively disable flashing
        else:
            self.flashing_interval = 1.0 / flashing_frequency_hz
        self._last_flash_time = time.time()
        self._flash_state = False # False = primary color, True = secondary color

        # Mapping for pose action descriptions
        self._action_map = {
            0: "Unknown",
            1: "Bending",
            2: "Sitting",
            3: "Squatting",
            4: "Standing"
        }
        # # Colors for flashing (e.g., red/white, can be made configurable)
        # self._flash_primary_color = (0, 0, 255)  # Red
        # self._flash_secondary_color = (255, 255, 255) # White


    def _update_flash_state(self):
        """Updates the internal flash state based on time and frequency."""
        current_time = time.time()
        if current_time - self._last_flash_time >= self.flashing_interval:
            self._flash_state = not self._flash_state
            self._last_flash_time = current_time

    @staticmethod
    def _draw_bbox_and_label(image: np.ndarray,
                             text: str,
                             bbox_coords: Tuple[int, int, int, int],
                             bbox_color: Tuple[int, int, int],
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             font_scale: float = 0.5,
                             thickness: int = 1):
        """Helper function to draw a bounding box and a label with a background."""
        lx, ly, rx, ry = bbox_coords

        # Draw bounding box
        cv2.rectangle(image, (lx, ly), (rx, ry), bbox_color, thickness)

        if text:
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Draw background rectangle for text
            # Adjust position above bbox
            text_bg_ly = max(0, ly - text_height - baseline - 5)
            text_bg_lx = lx
            text_bg_rx = min(image.shape[1], lx + text_width + 4)
             # Background ends at the top of the bbox
            text_bg_ry = ly

            cv2.rectangle(image,
                          (text_bg_lx, text_bg_ly),
                          (text_bg_rx, text_bg_ry),
                          bbox_color, # Use bbox color for background
                          -1) # Filled rectangle

            # Draw text
             # Position text inside bg
            text_y = max(text_height + 2, ly - baseline - 2)
            cv2.putText(image,
                        text,
                        (lx + 2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        text_color, # White text is usually readable on colored bg
                        thickness,
                        lineType=cv2.LINE_AA)


    def _get_static_bbox_color(self, style: str, oid: int = None) -> Tuple[int, int, int]:
        """Gets a static color based on style or OID for chromatic style."""
        if style == "chromatic" and oid is not None:
             # Ensure oid is non-negative before modulo
            valid_oid = max(0, oid)
            return self.bbox_colors[valid_oid % len(self.bbox_colors)]

        normal_colors = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255), # Corrected Yellow BGR
            "cyan": (255, 255, 0), # Corrected Cyan BGR
            "magenta": (255, 0, 255),
            "gray": (128, 128, 128),
        }
        # Default to white if style not found
        return normal_colors.get(style, (255, 255, 255))

    @staticmethod
    def _get_flashing_bbox_color(style: str) -> List[Tuple[int, int, int]]:
        flashing_colors = {
            "white_red": [(255, 255, 255), (0, 0, 255)],
            "white_blue": [(255, 255, 255), (255, 0, 0)],
            "white_green": [(255, 255, 255), (0, 255, 0)],
            "white_yellow": [(255, 255, 255), (0, 255, 255)],
            "white_cyan": [(255, 255, 255), (255, 255, 0)],
            "white_magenta": [(255, 255, 255), (255, 0, 255)],
            "white_gray": [(255, 255, 255), (128, 128, 128)],
        }

        # Default to white-red if style not found
        return flashing_colors.get(style, [(255, 255, 255), (0, 0, 255)])

    def draw_tracked_object(self,
                            frame: np.ndarray,
                            obj: Union[YoloSorted, YoloPoseSorted],
                            labels: Dict[int, str] = None,
                            bbox_style: str = "chromatic",
                            text_color: Tuple[int, int, int] = (255, 255, 255),
                            thickness: int = 1,
                            font_scale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box and label for a tracked object (with OID).
        Returns the calculated bounding box color.

        Label format: "ID {oid}" or "ID {oid} - {ClassName}" if labels are provided.
        Applies object confidence threshold.

        Args:
            frame: The image frame to draw on (modified in place).
            obj: The tracked object data (YoloSorted or YoloPoseSorted).
            labels: Optional dictionary mapping class index to class name.
            bbox_style: Style for the bounding box color ('chromatic', 'red', 'blue', etc.).
            text_color: Color for the label text (BGR).
            thickness: Thickness for the bounding box lines and text.
            font_scale: Font scale for the label text.

        Returns:
            The modified frame with the drawn bounding box and label.
        """
        if obj.conf < self.object_conf_threshold:
            return None # Skip drawing if below threshold
        
        # 拷贝frame，确保新的frame可写，并且不影响原始frame
        frame_copied = frame.copy()

        oid = obj.oid
        bbox_color = self._get_static_bbox_color(bbox_style, oid)
        bbox_coords = (obj.lx, obj.ly, obj.rx, obj.ry)

        # Determine label text
        label_text = f"ID {oid}"
        if labels and obj.cls in labels:
            label_text += f" - {labels[obj.cls]}"

        self._draw_bbox_and_label(frame_copied, label_text, bbox_coords, bbox_color,
                                  text_color, font_scale, thickness)

        return frame_copied


    def draw_detected_object(self,
                             frame: np.ndarray,
                             obj: Union[Yolo, YoloPose],
                             labels: Union[List[str], str] = None,
                             bbox_style: str = "white",
                             flashing: bool = False,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             thickness: int = 1,
                             font_scale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box and label for a detected object (without OID).
        Returns the calculated bounding box color.

        Label format: "{ClassName} - {Confidence:.2f}". Requires labels dictionary.
        Supports optional time-based flashing bounding box.
        Applies object confidence threshold.

        Args:
            frame: The image frame to draw on (modified in place).
            obj: The detected object data (Yolo or YoloPose).
            labels: List of class names for the detected object, or a single string.
            bbox_style: Style for the bounding box color ('white', 'red', etc.) if not flashing.
            flashing: If True, use time-based flashing colors for the bbox.
            text_color: Color for the label text (BGR).
            thickness: Thickness for the bounding box lines and text.
            font_scale: Font scale for the label text.

        Returns:
            The modified frame with the drawn bounding box and label.
        """
        if obj.conf < self.object_conf_threshold:
            return frame # Skip drawing if below threshold

        # 拷贝frame，确保新的frame可写，并且不影响原始frame
        frame_copied = frame.copy()

        if not labels:
            # Cannot determine label without labels dictionary
            label_text = f"Cls {obj.cls} - {obj.conf:.2f}" # Fallback label
        else:
            if isinstance(labels, str):
                label_text = labels
            elif isinstance(labels, list):
                label_text = f"{labels[obj.cls % len(labels)]} - {obj.conf:.2f}"
            else:
                raise ValueError("labels must be a list of class names or a single string")

        bbox_coords = (obj.lx, obj.ly, obj.rx, obj.ry)

        # Determine bbox color
        if flashing:
            self._update_flash_state() # Ensure flash state is current
            # bbox_color = self._flash_secondary_color if self._flash_state else self._flash_primary_color
            bbox_color = self._get_flashing_bbox_color(bbox_style)[int(self._flash_state)]
        else:
            bbox_color = self._get_static_bbox_color(bbox_style) # Use static color

            if bbox_style == "white":
                text_color = (0, 0, 0)  # Use black text on white bbox

        self._draw_bbox_and_label(frame_copied, label_text, bbox_coords, bbox_color,
                                  text_color, font_scale, thickness)
        
        return frame_copied


    def draw_skeleton(self,
                      frame: np.ndarray,
                      pose: Union[YoloPose, YoloPoseSorted],
                      show_pts: bool = True,
                      show_links: bool = True,
                      show_pts_name: bool = False,
                      kpt_radius: int = 3,
                      link_thickness: int = 1) -> np.ndarray:
        """
        Draws skeleton keypoints and links for a single pose.

        Applies point confidence threshold for drawing keypoints. Links are drawn
        only if both connected keypoints are above the threshold.

        Args:
            frame: The image frame to draw on (modified in place).
            pose: The pose data (YoloPose or YoloPoseSorted).
            show_pts: Whether to draw keypoints.
            show_links: Whether to draw skeleton links (bones).
            show_pts_name: Whether to label keypoints with their names.
            kpt_radius: Radius of the circle used to draw keypoints.
            link_thickness: Thickness of the lines used for skeleton links.

        Returns:
            The modified frame with the drawn skeleton.
        """
        valid_kpts = {} # Store valid keypoints for link drawing: idx -> (x, y, color, name)

        # 拷贝frame，确保新的frame可写，并且不影响原始frame
        frame_copied = frame.copy()

        # Draw keypoints first
        if show_pts:
            for idx, kpt in enumerate(pose.pts):
                # Use point_conf_threshold here
                if kpt.conf >= self.point_conf_threshold and idx in self.kpt_color_map:
                    kp_info = self.kpt_color_map[idx]
                    kpt_x, kpt_y = int(kpt.x), int(kpt.y)
                    # Ensure coordinates are valid before drawing
                    if kpt_x > 0 and kpt_y > 0:
                         cv2.circle(frame_copied, (kpt_x, kpt_y), kpt_radius, kp_info.color, -1)
                         valid_kpts[idx] = (kpt_x, kpt_y, kp_info.color, kp_info.name)
                         if show_pts_name:
                             cv2.putText(frame_copied, kp_info.name, (kpt_x, kpt_y - kpt_radius - 2),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, kp_info.color, 1, lineType=cv2.LINE_AA)

        # Draw skeleton links (bones)
        if show_links:
            for bone in self.skeleton_map:
                # Check if both start and end keypoints are valid (drawn)
                if bone.srt_kpt_id in valid_kpts and bone.dst_kpt_id in valid_kpts:
                    srt_x, srt_y, _, _ = valid_kpts[bone.srt_kpt_id]
                    dst_x, dst_y, _, _ = valid_kpts[bone.dst_kpt_id]
                    cv2.line(frame_copied, (srt_x, srt_y), (dst_x, dst_y),
                             bone.color, link_thickness, lineType=cv2.LINE_AA)
                    
        return frame_copied


    def draw_facial_orientation_vector(self,
                                       frame: np.ndarray,
                                       facial_vector: FacialDirection,
                                       color: Tuple[int, int, int] = (0, 255, 0), # Default Green
                                       thickness: int = 2,
                                       show_dir_name: bool = False,
                                       font_scale: float = 0.5) -> np.ndarray:
        """
        Draws a single facial orientation vector (arrow) on the image.

        Args:
            frame: The image frame to draw on (modified in place).
            facial_vector: The FacialDirection data object.
            color: Color of the arrow and optional text (BGR).
            thickness: Thickness of the arrow line.
            show_dir_name: Whether to display the direction description text.
            font_scale: Font scale for the direction text.

        Returns:
            The modified frame with the drawn vector and optional text.
        """
        op_x, op_y = facial_vector.origin
        module = facial_vector.modulus
        vec_x, vec_y = facial_vector.vector
        orientation_desc = facial_vector.direction_desc

        # 拷贝frame，确保新的frame可写，并且不影响原始frame
        frame_copied = frame.copy()

        # Calculate arrow end point
        # Ensure start and end points are valid integers
        start_point = (int(op_x), int(op_y))
        # Only draw if vector and modulus are meaningful
        if module > 0 and (vec_x != 0 or vec_y != 0):
            end_point = (int(op_x + vec_x * module), int(op_y + vec_y * module))
            cv2.arrowedLine(frame_copied, start_point, end_point, color, thickness, line_type=cv2.LINE_AA)
        else:
            # Optionally draw a small circle at the origin if vector is zero
            cv2.circle(frame_copied, start_point, thickness + 1, color, 1)


        if show_dir_name:
            text_pos = (start_point[0] + 5, start_point[1] - 5) # Position near start point
            cv2.putText(frame_copied, orientation_desc, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            
        return frame_copied

    #
    # def draw_supplementary_info(self,
    #                             frame: np.ndarray,
    #                             obj: Union[Yolo, YoloPose, YoloSorted, YoloPoseSorted],
    #                             supplementary_text: str,
    #                             text_color: Tuple[int, int, int] = (255, 255, 255), # White text
    #                             bg_color: Tuple[int, int, int] = (0, 0, 0),      # Black background
    #                             font_scale: float = 0.4,
    #                             thickness: int = 1,
    #                             padding: int = 3) -> np.ndarray:
    #     """
    #     Draws supplementary multi-line text inside the top-left corner of the object's bounding box.
    #
    #     Args:
    #         frame: The image frame to draw on (modified in place).
    #         obj: The object data (Yolo, YoloPose, YoloSorted, or YoloPoseSorted).
    #         supplementary_text: The text string, potentially containing '\\n' for newlines.
    #         text_color: Color for the text (BGR).
    #         bg_color: Background color for the text label (BGR).
    #         font_scale: Font scale for the text.
    #         thickness: Thickness for the text.
    #         padding: Padding around the text block within the background box.
    #
    #     Returns:
    #         The modified frame with the drawn text.
    #     """
    #     if obj.conf < self.object_conf_threshold or not supplementary_text:
    #         return frame # Skip if confidence low or no text
    #
    #     # 拷贝frame，确保新的frame可写，并且不影响原始frame
    #     frame_copied = frame.copy()
    #
    #     lx, ly = obj.lx, obj.ly
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     lines = supplementary_text.strip().split('\n')
    #
    #     max_text_width = 0
    #     total_text_height = 0
    #     line_heights = []
    #
    #     for line in lines:
    #         (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
    #         max_text_width = max(max_text_width, text_width)
    #         line_actual_height = text_height + baseline
    #         line_heights.append(line_actual_height)
    #         total_text_height += line_actual_height
    #
    #     # Add padding to total height (top/bottom)
    #     total_bg_height = total_text_height + (len(lines) -1) * padding // 2 + 2 * padding # Add inter-line spacing too
    #     total_bg_width = max_text_width + 2 * padding
    #
    #     # Calculate background rectangle position (inside bbox top-left)
    #     rect_tl = (lx, ly)
    #     rect_br = (lx + total_bg_width, ly + total_bg_height)
    #
    #     # Ensure rectangle doesn't exceed image boundaries (optional but good)
    #     rect_br = (min(frame.shape[1], rect_br[0]), min(frame.shape[0], rect_br[1]))
    #     rect_tl = (max(0, rect_tl[0]), max(0, rect_tl[1]))
    #
    #     # Draw background if valid
    #     if rect_br[0] > rect_tl[0] and rect_br[1] > rect_tl[1]:
    #         cv2.rectangle(frame_copied, rect_tl, rect_br, bg_color, -1)
    #
    #         # Draw each line of text
    #         current_y = ly + padding
    #         for i, line in enumerate(lines):
    #              # Get size again to use baseline for positioning
    #              (_, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
    #              # Text origin (bottom-left) for putText
    #              text_origin_y = current_y + text_height
    #              text_origin = (lx + padding, text_origin_y)
    #              cv2.putText(frame_copied, line, text_origin, font, font_scale, text_color, thickness, cv2.LINE_AA)
    #              # Move y down for next line
    #              current_y += line_heights[i] + padding // 2


    def draw_supplementary_info(self,
                                frame: np.ndarray,
                                obj: Union[Yolo, YoloPose, YoloSorted, YoloPoseSorted],
                                supplementary_text: str,
                                text_color: Tuple[int, int, int] = (255, 255, 255),  # White text
                                bg_color: Tuple[int, int, int] = (0, 0, 0),  # Black background
                                font_scale: float = 0.4,
                                thickness: int = 1,
                                padding: int = 3) -> np.ndarray:
        """
        Draws supplementary multi-line text inside the top-left corner of the object's bounding box.
        # ... (Args) ...
        Returns:
            The modified frame with the drawn text.
        """
        # Check 1: Skip if confidence low or no text
        if obj.conf < self.object_conf_threshold or not supplementary_text:
            return frame  # Returns original frame - OK

        # --- Potential Issue Area ---
        # Check 2: Input frame check - Let's add this
        if frame is None:
            print(
                f"ERROR [VisionDrawer::draw_supplementary_info]: Input frame is None for OID {getattr(obj, 'oid', 'N/A')}!")
            return None  # Explicitly return None

        frame_copied = None  # Initialize
        try:
            # Check 3: Copy frame
            frame_copied = frame.copy()
            if frame_copied is None:  # Very unlikely
                print(
                    f"ERROR [VisionDrawer::draw_supplementary_info]: frame.copy() resulted in None for OID {getattr(obj, 'oid', 'N/A')}!")
                return frame  # Return original if copy fails

            lx, ly = obj.lx, obj.ly
            font = cv2.FONT_HERSHEY_SIMPLEX
            lines = supplementary_text.strip().split('\n')

            max_text_width = 0
            total_text_height = 0
            line_heights = []

            # Check 4: Calculate text size (wrap in try-except)
            for line in lines:
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                max_text_width = max(max_text_width, text_width)
                line_actual_height = text_height + baseline
                line_heights.append(line_actual_height)
                total_text_height += line_actual_height

            total_bg_height = total_text_height + (len(lines) - 1) * padding // 2 + 2 * padding
            total_bg_width = max_text_width + 2 * padding

            rect_tl = (lx, ly)
            rect_br = (lx + total_bg_width, ly + total_bg_height)

            rect_br = (min(frame_copied.shape[1], rect_br[0]), min(frame_copied.shape[0], rect_br[1]))
            rect_tl = (max(0, rect_tl[0]), max(0, rect_tl[1]))

            # Check 5: Draw background rectangle (wrap in try-except)
            if rect_br[0] > rect_tl[0] and rect_br[1] > rect_tl[1]:
                cv2.rectangle(frame_copied, rect_tl, rect_br, bg_color, -1)

                # Check 6: Draw text lines (wrap in try-except)
                current_y = ly + padding
                for i, line in enumerate(lines):
                    (_, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                    text_origin_y = current_y + text_height
                    text_origin = (lx + padding, text_origin_y)
                    # Ensure text origin is within frame boundaries before drawing
                    if 0 <= text_origin[0] < frame_copied.shape[1] and 0 <= text_origin[1] < frame_copied.shape[0]:
                        cv2.putText(frame_copied, line, text_origin, font, font_scale, text_color, thickness,
                                    cv2.LINE_AA)
                    else:
                        # Optionally log if text position is out of bounds
                        # print(f"Warning [VisionDrawer::draw_supplementary_info]: Text '{line}' for OID {getattr(obj, 'oid', 'N/A')} at {text_origin} is out of bounds.")
                        pass  # Skip drawing text if out of bounds
                    current_y += line_heights[i] + padding // 2
            else:
                # Optionally log if background rectangle itself is invalid
                # print(f"Warning [VisionDrawer::draw_supplementary_info]: Background rectangle for OID {getattr(obj, 'oid', 'N/A')} is invalid: TL={rect_tl}, BR={rect_br}")
                pass  # Skip drawing if background box has no size

            # Check 7: Final check before returning
            if frame_copied is None:
                print(
                    f"CRITICAL ERROR [VisionDrawer::draw_supplementary_info]: frame_copied is None just before return for OID {getattr(obj, 'oid', 'N/A')}!")
                return frame  # Safest to return original frame

            return frame_copied  # Returns the copy - This is the expected successful path

        except cv2.error as cv_err:
            print(
                f"ERROR [VisionDrawer::draw_supplementary_info]: OpenCV error for OID {getattr(obj, 'oid', 'N/A')}: {cv_err}")
            # Decide: Return original frame or None? Original is safer.
            return frame
        except Exception as e:
            print(
                f"ERROR [VisionDrawer::draw_supplementary_info]: Unexpected error for OID {getattr(obj, 'oid', 'N/A')}: {e}")
            # Decide: Return original frame or None? Original is safer.
            return frame


    def draw_pose_action(self,
                         frame: np.ndarray,
                         pose: Union[YoloPose, YoloPoseSorted],
                         action: int,
                         bbox_color: Tuple[int, int, int], # Expect bbox color now
                         text_color: Tuple[int, int, int] = (255, 255, 255), # White text default
                         font_scale: float = 0.5,
                         thickness: int = 1,
                         padding: int = 3) -> np.ndarray:
        """
        Draws the detected pose action text inside the BOTTOM-RIGHT corner
        of the pose's bounding box, using the provided bbox_color for the background.

        Args:
            frame: The image frame to draw on (modified in place).
            pose: The pose data (YoloPose or YoloPoseSorted).
            action: The action code (int) detected for the pose.
            bbox_color: The color used for the object's bounding box (used as background here).
            text_color: Color for the action text (BGR). Default is white.
            font_scale: Font scale for the action text.
            thickness: Thickness for the text.
            padding: Padding around the text within the background box.

        Returns:
            The modified frame with the drawn action text.
        """
        if pose.conf < self.object_conf_threshold or bbox_color is None:
             # Skip if confidence low or no bbox color provided
             return frame
        
        # 拷贝frame，确保新的frame可写，并且不影响原始frame
        frame_copied = frame.copy()

        action_text = self._action_map.get(action, "Unknown")
        rx, ry = pose.rx, pose.ry # Use bottom-right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(action_text, font, font_scale, thickness)

        # Calculate background rectangle position (aligned to bbox bottom-right)
        total_bg_height = text_height + baseline + 2 * padding
        total_bg_width = text_width + 2 * padding

        rect_tl_x = rx - total_bg_width
        rect_tl_y = ry - total_bg_height
        rect_br = (rx, ry)
        rect_tl = (rect_tl_x, rect_tl_y)


        # Ensure rectangle doesn't exceed image boundaries (optional)
        rect_tl = (max(0, rect_tl[0]), max(0, rect_tl[1]))
        rect_br = (min(frame.shape[1], rect_br[0]), min(frame.shape[0], rect_br[1]))


        # Draw background and text if valid
        if rect_br[0] > rect_tl[0] and rect_br[1] > rect_tl[1]:
             # Use bbox_color for background
            cv2.rectangle(frame_copied, rect_tl, rect_br, bbox_color, -1)
            # Text origin (bottom-left of text), aligned bottom-right inside background
            text_origin_x = rx - text_width - padding
            text_origin_y = ry - baseline - padding
            cv2.putText(frame_copied, action_text, (text_origin_x, text_origin_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return frame_copied
