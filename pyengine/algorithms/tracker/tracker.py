# tracker.py
from collections import deque
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect


class UnifiedTrack:
    """
    A unified Track class, now using a more stable Kalman Filter state representation
    inspired by the original SORT algorithm. It can be used for both DeepSORT and SORT.

    MODIFIED: The state's 'cy' now represents the Y-coordinate of the bottom-center of the bounding box.
    """
    _next_id = 0  # Static variable for unique track IDs

    def __init__(self, detection: ObjectDetection, use_reid: bool = True):
        """
        Initializes a new track.
        Args:
            detection (ObjectDetection): The first detection to initialize the track.
            use_reid (bool): If True, stores and updates Re-ID features (for DeepSORT).
        """
        self.track_id = UnifiedTrack._next_id
        UnifiedTrack._next_id += 1
        self.use_reid = use_reid

        # --- Refactored Kalman Filter based on old_sort.py's principles ---
        # 7D State: [cx, cy, s, r, dcx, dcy, ds]
        # cx: center_x of the bbox
        # cy: y-coordinate of the bbox's bottom edge (bottom_center_y) <-- MODIFIED
        # s: scale (area = w * h)
        # r: aspect ratio (w / h)
        # dcx, dcy, ds: respective velocities
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State Transition Matrix (F)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement Matrix (H) - we only measure position, area, and ratio
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Noise Covariances (tuned for stability)
        self.kf.R[2:, 2:] *= 10.  # Measurement noise
        self.kf.Q[-1, -1] *= 0.01  # Process noise
        self.kf.Q[4:, 4:] *= 0.01

        # Initial State Covariance (P) - high uncertainty
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

        # Initialize state from the first detection
        self._init_state(detection)

        if self.use_reid:
            self.features = deque([np.array(detection.features)], maxlen=100) if detection.features else deque(
                maxlen=100)
        else:
            self.features = None

        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def _rect_to_z(self, rect: Rect) -> np.ndarray:
        """
        Converts a Rect object to a measurement vector [cx, cy, s, r].
        'cy' is now the bottom-center y-coordinate.
        """
        w = rect.x2 - rect.x1
        h = rect.y2 - rect.y1
        cx = rect.x1 + w / 2.0
        # --- CHANGED: Use the bottom edge y-coordinate (y2) instead of the center ---
        cy = rect.y2
        # --- END CHANGE ---
        s = w * h  # Area
        r = w / float(h) if h > 0 else 0  # Aspect Ratio
        return np.array([cx, cy, s, r]).reshape((4, 1))

    def _init_state(self, detection: ObjectDetection):
        """Initializes the Kalman Filter state from a detection."""
        measurement = self._rect_to_z(detection.rect)
        self.kf.x = np.vstack([measurement, np.zeros((3, 1))])  # Initial velocities are zero

    def predict(self):
        """Predicts the track state for the next frame."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: ObjectDetection):
        """Updates the track state with a new matched detection."""
        measurement = self._rect_to_z(detection.rect)
        self.kf.update(measurement)

        if self.use_reid and detection.features:
            self.features.append(np.array(detection.features))

        self.hits += 1
        self.time_since_update = 0

    def get_state(self) -> Rect:
        """
        Gets the predicted bounding box (Rect) from the Kalman Filter state.
        Includes safeguards against invalid values.
        This now interprets 'cy' as the bottom edge y-coordinate.
        """
        cx, cy, s, r = self.kf.x[:4].flatten()

        # --- FIX: Add robust checks to prevent invalid values ---
        # Clamp state variables to be physically plausible.
        s = max(0, s)
        r = max(0.01, r)

        w = np.sqrt(s * r)

        # Avoid division by zero if width is zero.
        if w < 1e-6:
            h = 0
        else:
            h = s / w

        # Final safeguard against NaN from sqrt of a negative number (though s is clamped).
        if np.isnan(w) or np.isnan(h):
            w, h = 0, 0
        # --- END FIX ---

        # --- CHANGED: Convert from bottom-center coordinates back to a Rect ---
        # cx is still the center, so x1 and x2 are calculated as before.
        x1 = cx - w / 2
        x2 = cx + w / 2
        # cy is the bottom edge (y2), so y1 is calculated from it.
        y2 = cy
        y1 = cy - h
        # --- END CHANGE ---

        return Rect(x1=x1, y1=y1, x2=x2, y2=y2)

    def get_mean_feature(self) -> Optional[np.ndarray]:
        """Returns the mean of stored Re-ID features."""
        if self.use_reid and self.features:
            return np.mean(list(self.features), axis=0)
        return None

    def get_last_feature(self) -> Optional[np.ndarray]:
        """Returns the last stored Re-ID feature."""
        if self.use_reid and self.features:
            return self.features[-1]
        return None

    def is_confirmed(self, min_hits: int) -> bool:
        """Checks if the track is confirmed (has enough hits)."""
        return self.hits >= min_hits

    def is_deleted(self, max_age: int) -> bool:
        """Checks if the track should be deleted (lost for too long)."""
        return self.time_since_update > max_age