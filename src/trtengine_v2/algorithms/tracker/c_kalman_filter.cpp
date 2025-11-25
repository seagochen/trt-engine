/**
 * @file c_kalman_filter.cpp
 * @brief Kalman filter implementation using Eigen library
 *
 * 7D constant-velocity model for bounding box tracking.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/algorithms/tracker/c_kalman_filter.h"

#include <eigen/Eigen/Dense>
#include <cmath>
#include <algorithm>

using Eigen::Matrix;
using Eigen::Vector;

// State dimension (cx, cy, s, r, dcx, dcy, ds)
constexpr int STATE_DIM = 7;
// Measurement dimension (cx, cy, s, r)
constexpr int MEAS_DIM = 4;

using StateVector = Vector<float, STATE_DIM>;
using MeasVector = Vector<float, MEAS_DIM>;
using StateCov = Matrix<float, STATE_DIM, STATE_DIM>;
using MeasCov = Matrix<float, MEAS_DIM, MEAS_DIM>;
using TransitionMatrix = Matrix<float, STATE_DIM, STATE_DIM>;
using MeasurementMatrix = Matrix<float, MEAS_DIM, STATE_DIM>;
using KalmanGain = Matrix<float, STATE_DIM, MEAS_DIM>;

/**
 * @brief Internal Kalman filter structure
 */
struct C_KalmanFilter {
    StateVector x;              // State vector
    StateCov P;                 // State covariance
    TransitionMatrix F;         // State transition matrix
    MeasurementMatrix H;        // Measurement matrix
    StateCov Q;                 // Process noise covariance
    MeasCov R;                  // Measurement noise covariance
};

/**
 * @brief Convert bounding box to measurement vector [cx, cy, s, r]
 */
static MeasVector bbox_to_measurement(const C_BoundingBox* bbox) {
    float width = bbox->x2 - bbox->x1;
    float height = bbox->y2 - bbox->y1;
    float cx = bbox->x1 + width / 2.0f;
    float cy = bbox->y2;  // Bottom-center Y (as in original SORT)
    float s = width * height;  // Scale (area)
    float r = width / std::max(height, 1e-6f);  // Aspect ratio

    MeasVector z;
    z << cx, cy, s, r;
    return z;
}

/**
 * @brief Convert state vector to bounding box
 */
static C_BoundingBox state_to_bbox(const StateVector& x) {
    float cx = x(0);
    float cy = x(1);
    float s = std::max(x(2), 1.0f);  // Ensure positive scale
    float r = std::max(x(3), 0.1f);  // Ensure positive ratio

    // s = w * h, r = w / h
    // => w = sqrt(s * r), h = sqrt(s / r)
    float w = std::sqrt(s * r);
    float h = std::sqrt(s / r);

    C_BoundingBox bbox;
    bbox.x1 = cx - w / 2.0f;
    bbox.y1 = cy - h;  // cy is bottom-center
    bbox.x2 = cx + w / 2.0f;
    bbox.y2 = cy;

    return bbox;
}

extern "C" {

C_KalmanFilter* c_kalman_filter_create(const C_BoundingBox* bbox) {
    if (!bbox) return nullptr;

    C_KalmanFilter* kf = new (std::nothrow) C_KalmanFilter();
    if (!kf) return nullptr;

    // Initialize state from bounding box
    MeasVector z = bbox_to_measurement(bbox);
    kf->x = StateVector::Zero();
    kf->x.head<4>() = z;  // Position states
    // Velocity states initialized to 0

    // State transition matrix (constant velocity model, dt=1)
    // x_new = F * x
    // [cx, cy, s, r, dcx, dcy, ds]^T
    kf->F = TransitionMatrix::Identity();
    kf->F(0, 4) = 1.0f;  // cx += dcx
    kf->F(1, 5) = 1.0f;  // cy += dcy
    kf->F(2, 6) = 1.0f;  // s += ds
    // r (aspect ratio) is assumed constant

    // Measurement matrix
    // z = H * x
    kf->H = MeasurementMatrix::Zero();
    kf->H(0, 0) = 1.0f;  // cx
    kf->H(1, 1) = 1.0f;  // cy
    kf->H(2, 2) = 1.0f;  // s
    kf->H(3, 3) = 1.0f;  // r

    // Measurement noise covariance
    // Higher uncertainty for scale and aspect ratio
    kf->R = MeasCov::Identity();
    kf->R(0, 0) = 1.0f;   // cx variance
    kf->R(1, 1) = 1.0f;   // cy variance
    kf->R(2, 2) = 10.0f;  // s variance (area has more uncertainty)
    kf->R(3, 3) = 10.0f;  // r variance

    // Process noise covariance
    kf->Q = StateCov::Identity();
    kf->Q(0, 0) = 1.0f;    // cx
    kf->Q(1, 1) = 1.0f;    // cy
    kf->Q(2, 2) = 1.0f;    // s
    kf->Q(3, 3) = 1.0f;    // r
    kf->Q(4, 4) = 0.01f;   // dcx (velocity has low process noise)
    kf->Q(5, 5) = 0.01f;   // dcy
    kf->Q(6, 6) = 0.0001f; // ds

    // Initial state covariance
    // High uncertainty for initial state, especially velocities
    kf->P = StateCov::Identity() * 10.0f;
    kf->P(4, 4) = 1000.0f;  // High uncertainty for initial velocity
    kf->P(5, 5) = 1000.0f;
    kf->P(6, 6) = 1000.0f;

    return kf;
}

void c_kalman_filter_destroy(C_KalmanFilter* kf) {
    delete kf;
}

C_BoundingBox c_kalman_filter_predict(C_KalmanFilter* kf) {
    if (!kf) {
        return C_BoundingBox{0, 0, 0, 0};
    }

    // State prediction: x = F * x
    kf->x = kf->F * kf->x;

    // Covariance prediction: P = F * P * F^T + Q
    kf->P = kf->F * kf->P * kf->F.transpose() + kf->Q;

    // Ensure scale remains positive
    if (kf->x(2) < 1.0f) {
        kf->x(2) = 1.0f;
    }
    // Ensure aspect ratio remains positive
    if (kf->x(3) < 0.1f) {
        kf->x(3) = 0.1f;
    }

    return state_to_bbox(kf->x);
}

C_BoundingBox c_kalman_filter_update(C_KalmanFilter* kf, const C_BoundingBox* bbox) {
    if (!kf || !bbox) {
        return kf ? state_to_bbox(kf->x) : C_BoundingBox{0, 0, 0, 0};
    }

    // Convert measurement
    MeasVector z = bbox_to_measurement(bbox);

    // Innovation (measurement residual): y = z - H * x
    MeasVector y = z - kf->H * kf->x;

    // Innovation covariance: S = H * P * H^T + R
    MeasCov S = kf->H * kf->P * kf->H.transpose() + kf->R;

    // Kalman gain: K = P * H^T * S^(-1)
    KalmanGain K = kf->P * kf->H.transpose() * S.inverse();

    // State update: x = x + K * y
    kf->x = kf->x + K * y;

    // Covariance update: P = (I - K * H) * P
    // Using Joseph form for numerical stability:
    // P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
    StateCov I_KH = StateCov::Identity() - K * kf->H;
    kf->P = I_KH * kf->P * I_KH.transpose() + K * kf->R * K.transpose();

    // Ensure scale and ratio remain positive
    if (kf->x(2) < 1.0f) {
        kf->x(2) = 1.0f;
    }
    if (kf->x(3) < 0.1f) {
        kf->x(3) = 0.1f;
    }

    return state_to_bbox(kf->x);
}

C_BoundingBox c_kalman_filter_get_bbox(const C_KalmanFilter* kf) {
    if (!kf) {
        return C_BoundingBox{0, 0, 0, 0};
    }
    return state_to_bbox(kf->x);
}

void c_kalman_filter_reset(C_KalmanFilter* kf, const C_BoundingBox* bbox) {
    if (!kf || !bbox) return;

    // Re-initialize state from bounding box
    MeasVector z = bbox_to_measurement(bbox);
    kf->x = StateVector::Zero();
    kf->x.head<4>() = z;

    // Reset covariance to initial values
    kf->P = StateCov::Identity() * 10.0f;
    kf->P(4, 4) = 1000.0f;
    kf->P(5, 5) = 1000.0f;
    kf->P(6, 6) = 1000.0f;
}

C_KalmanFilter* c_kalman_filter_copy(const C_KalmanFilter* src) {
    if (!src) return nullptr;

    C_KalmanFilter* dst = new (std::nothrow) C_KalmanFilter();
    if (!dst) return nullptr;

    dst->x = src->x;
    dst->P = src->P;
    dst->F = src->F;
    dst->H = src->H;
    dst->Q = src->Q;
    dst->R = src->R;

    return dst;
}

} // extern "C"
