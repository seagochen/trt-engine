from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

# 从你的文件导入数据结构
from pyengine.inference.c_wrapper.c_pose_data_struct import Rect, PoseDetection
from pyengine.inference.c_wrapper.c_pose_pipeline_wrapper import PosePipelineWrapper
from pyengine.inference.tracker.tracker import UnifiedTrack


class DeepSORTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.5, reid_threshold: float = 0.4):
        self.tracks: List[UnifiedTrack] = []
        self.max_age = max_age  # Max frames for a track to be unmatched before deletion
        self.min_hits = min_hits  # Min hits for a new track to be confirmed
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold  # Cosine similarity threshold for re-ID

    def _iou_cost(self, track_bbox: Rect, det_bbox: Rect) -> float:
        # Calculate IoU distance (1 - IoU)
        # Your IoU calculation logic here, or use a utility function
        # For simplicity, returning 1 if no overlap or negative for distance

        # Convert Rect to (x1, y1, x2, y2)
        bb1 = [track_bbox.x1, track_bbox.y1, track_bbox.x2, track_bbox.y2]
        bb2 = [det_bbox.x1, det_bbox.y1, det_bbox.x2, det_bbox.y2]

        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 1.0  # No overlap, distance is 1 (max)

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return 1.0 - iou  # Return 1 - IoU as distance/cost

    def _cosine_distance(self, track_feature: np.ndarray, det_feature: np.ndarray) -> float:
        # Calculate cosine distance (1 - cosine_similarity)
        # Ensure features are normalized if not already
        track_feature = track_feature / np.linalg.norm(track_feature)
        det_feature = det_feature / np.linalg.norm(det_feature)

        similarity = np.dot(track_feature, det_feature)
        return 1.0 - similarity  # Return 1 - similarity as distance/cost

    def update(self, detections: List[PoseDetection]) -> Dict[int, Rect]:
        # 1. Predict
        for track in self.tracks:
            track.predict()

        # 2. Separate confirmed and unconfirmed tracks (DeepSORT's cascading logic)
        # For simplicity here, we'll treat all tracks equally,
        # A full DeepSORT would prioritize matching to recent tracks first.

        # 3. Create cost matrix
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        cost_matrix = np.full((num_tracks, num_dets), np.inf)

        for i, track in enumerate(self.tracks):
            track_predicted_bbox = track.get_state()
            track_feature = track.get_feature()

            for j, det in enumerate(detections):
                # IoU cost
                iou_cost = self._iou_cost(track_predicted_bbox, det.box)

                # Re-ID cost (only if feature exists)
                reid_cost = np.inf
                if track_feature is not None and det.features:
                    reid_cost = self._cosine_distance(track_feature, np.array(det.features))

                # Combine costs (DeepSORT's typical combination)
                # You'd tune weights or use gating for these
                if iou_cost <= (1.0 - self.iou_threshold) and reid_cost <= (1.0 - self.reid_threshold):
                    # Example weighted sum. DeepSORT uses a more sophisticated approach
                    # with gating on Mahalanobis distance as well.
                    cost_matrix[i, j] = 0.5 * iou_cost + 0.5 * reid_cost
                else:
                    cost_matrix[i, j] = np.inf  # Not compatible

        # 4. Solve Assignment (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks_indices = set(range(num_tracks))
        unmatched_dets_indices = set(range(num_dets))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < np.inf:  # Check if it's a valid match
                matches.append((r, c))
                unmatched_tracks_indices.discard(r)
                unmatched_dets_indices.discard(c)

        # 5. Update Tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Handle unmatched tracks
        for track_idx in unmatched_tracks_indices:
            self.tracks[track_idx].time_since_update += 1

        # Remove old, unmatched tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        # 6. Initialize New Tracks
        for det_idx in unmatched_dets_indices:
            new_track = UnifiedTrack(detections[det_idx])
            self.tracks.append(new_track)

        # Filter out "tentative" new tracks that haven't been seen enough times
        # This is part of DeepSORT's `min_hits` logic
        final_tracks_output = {}
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age < self.min_hits:  # Simplified confirmation
                final_tracks_output[track.track_id] = track.get_state()

        return final_tracks_output


# --- Example Usage (Integration with your PosePipelineWrapper) ---
if __name__ == "__main__":
    # Define the paths of engines and libs (adjust as per your system)
    YOLO_POSE_ENGINE = "/opt/models/yolov8s-pose.engine"
    EFFICIENTNET_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine"
    C_LIB = "/opt/TrtEngineToolkits/lib/libjetson.so"

    # Mock logger if not fully set up in your environment
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DeepSORT_Example")

    pipeline_wrapper = None
    tracker = None
    try:
        # Initialize your C++ pipeline wrapper
        pipeline_wrapper = PosePipelineWrapper(
            pose_model_path=YOLO_POSE_ENGINE,
            feats_model_path=EFFICIENTNET_ENGINE,
            c_lib_path=C_LIB,
            maximum_det_items=100,  # Max detections from the C++ side
            cls_threshold=0.5,  # Detection confidence threshold from C++
            iou_threshold=0.4  # NMS IoU threshold from C++
        )

        # Initialize your DeepSORT tracker
        # max_age: tracks are deleted after this many unmatched frames
        # min_hits: new tracks need this many hits to become confirmed
        tracker = DeepSORTTracker(max_age=30, min_hits=3, iou_threshold=0.3, reid_threshold=0.5)

        # Simulate processing a video sequence
        # You would replace this with actual video frame reading (e.g., using OpenCV)

        # Example image paths (from your wrapper example)
        test_image_paths = [
            "/opt/images/supermarket/customer1.png",
            "/opt/images/supermarket/customer2.png",
            # Add more images to simulate sequence
            "/opt/images/supermarket/customer3.png",
            "/opt/images/supermarket/customer4.png",
            "/opt/images/supermarket/customer5.png",
            "/opt/images/supermarket/customer6.png",
            "/opt/images/supermarket/customer7.png",
            "/opt/images/supermarket/customer8.png",
            "/opt/images/supermarket/staff1.png",
            "/opt/images/supermarket/staff2.png",
            "/opt/images/supermarket/staff3.png",
            "/opt/images/supermarket/staff4.png",
            "/opt/images/supermarket/staff5.png",
            "/opt/images/supermarket/staff6.png",
            "/opt/images/supermarket/staff7.png",
            "/opt/images/supermarket/staff8.png",
        ]

        import cv2

        frame_idx = 0
        for img_path in test_image_paths:
            frame_idx += 1
            print(f"\n--- Processing Frame {frame_idx}: {img_path} ---")

            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Add image to C++ pipeline
            pipeline_wrapper.add_image(img)

            # Run C++ inference to get detections with features
            results_list = pipeline_wrapper.inference()

            if not results_list:
                logger.warning(f"No detections from C++ pipeline for frame {frame_idx}.")
                continue

            # Get detections for the current frame (assuming batch size of 1 for now)
            current_frame_detections = results_list[0].detections

            # Optional: Filter detections based on Python-side confidence threshold if needed
            # For DeepSORT, usually you pass detections that are already high-confidence
            filtered_detections = [
                d for d in current_frame_detections if d.confidence > 0.6  # Example Python-side threshold
            ]

            # Update the DeepSORT tracker
            tracked_objects = tracker.update(filtered_detections)

            print(f"Frame {frame_idx} tracked objects: {len(tracked_objects)}")
            for track_id, bbox in tracked_objects.items():
                print(f"  Track ID: {track_id}, BBox: ({bbox.x1:.2f}, {bbox.y1:.2f}, {bbox.x2:.2f}, {bbox.y2:.2f})")

            # --- Visualization (Optional) ---
            # You can draw bounding boxes and IDs on the original image here
            display_img = img.copy()
            for track_id, bbox in tracked_objects.items():
                x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                            2)

            # cv2.imshow("Tracked Frame", display_img)
            # cv2.waitKey(100) # Wait 100ms

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if pipeline_wrapper:
            pipeline_wrapper.release()
        # cv2.destroyAllWindows() # If you used cv2.imshow
        print("\n--- DeepSORT Example Finished ---")