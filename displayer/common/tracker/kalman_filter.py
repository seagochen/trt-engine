import numpy as np
from scipy.linalg import inv


class KalmanFilter:
    def __init__(self, id, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / h
        self.x = np.array([x, y, s, r, 0, 0, 0], dtype=float)

        self.id = id
        self.time_since_update = 0
        self.hit_streak = 0

        dt = 1
        self.A = np.array([[1, 0, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])

        self.Q = np.eye(7) * 0.01
        self.R = np.eye(4) * 0.1
        self.P = np.eye(7)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.time_since_update += 1
        return self.x

    def update(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / h
        z = np.array([x, y, s, r], dtype=float)
        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y_residual
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self):
        x, y, s, r = self.x[:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / w
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
