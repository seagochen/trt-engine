import os

import cv2

from common.yaml.YamlConfig import YamlConfig


class DisplayHandler:
    def __init__(self, config: YamlConfig):

        self.infer_config = config.get_inference_config()
        self.visual_config = config.get_visualization_config()

        self.title = f"JetInfer: {self.infer_config['inference_stakeholder']}"
        self.window_size = (self.visual_config['display_width'], self.visual_config['display_height'])
        self.labels = self.load_labels()

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, *self.window_size)

    def display_frame(self, frame, bounding_boxes):

        # Draw the bounding boxes on the frame
        for _, row in bounding_boxes.iterrows():
            # label
            cls = self.labels.get(row['cls'], "Unknown")
            # confidence
            conf = row['conf']
            # bounding box
            lx, ly, rx, ry = row['lx'], row['ly'], row['rx'], row['ry']

            cv2.rectangle(
                frame,
                (int(lx), int(ly)), # Top-left corner
                (int(rx), int(ry)), # Bottom-right corner
                (0, 255, 0),  # Color: green
                2  # Thickness
            )

            cv2.putText(
                frame,
                f"{cls} {conf:.2f}",
                (int(lx), int(ly - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),  # Color: green
                2  # Thickness
            )

        # Resize the frame to fit the window
        display_frame = self.resize_frame(frame)

        # Display the frame
        cv2.imshow(self.title, display_frame)

    def resize_frame(self, frame):

        # Check if auto-resize is enabled
        if not self.visual_config['auto_resize']:
            return frame

        height, width = frame.shape[:2]
        window_width, window_height = self.window_size

        # Calculate the scaling factor
        scale = min(window_width / width, window_height / height)

        # Resize the frame
        return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def load_labels(self):
        label_file = self.infer_config["inference_labels"]
        label_dict = {}
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                # Use the line number as the label index
                for idx, line in enumerate(f):
                    label_dict[idx] = line.strip()
        return label_dict

    @staticmethod
    def check_for_exit():
        key = cv2.waitKey(1) & 0xFF
        return key == 27  # 27 is the ASCII code for the ESC key

    @staticmethod
    def close():
        cv2.destroyAllWindows()