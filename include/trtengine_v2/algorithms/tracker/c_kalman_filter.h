/**
 * @file c_kalman_filter.h
 * @brief Kalman filter for object tracking (7D constant-velocity model)
 *
 * This Kalman filter is specifically designed for bounding box tracking.
 * State vector: [cx, cy, s, r, dcx, dcy, ds] where:
 *   - cx, cy: Center coordinates (cy is bottom-center Y)
 *   - s: Scale (area = width * height)
 *   - r: Aspect ratio (width / height)
 *   - dcx, dcy, ds: Velocities
 *
 * Measurement vector: [cx, cy, s, r]
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_ALGORITHMS_TRACKER_C_KALMAN_FILTER_H
#define TRTENGINE_V2_ALGORITHMS_TRACKER_C_KALMAN_FILTER_H

#include "c_tracker_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque Kalman filter handle
 */
typedef struct C_KalmanFilter C_KalmanFilter;

/**
 * @brief Create a new Kalman filter initialized with a bounding box
 *
 * @param bbox Initial bounding box
 * @return Pointer to new Kalman filter, or NULL on failure
 */
C_KalmanFilter* c_kalman_filter_create(const C_BoundingBox* bbox);

/**
 * @brief Destroy Kalman filter and free memory
 *
 * @param kf Kalman filter to destroy
 */
void c_kalman_filter_destroy(C_KalmanFilter* kf);

/**
 * @brief Predict next state (advance one time step)
 *
 * Propagates the state using the constant-velocity model.
 * After predict, the state represents the predicted position.
 *
 * @param kf Kalman filter
 * @return Predicted bounding box
 */
C_BoundingBox c_kalman_filter_predict(C_KalmanFilter* kf);

/**
 * @brief Update state with a new measurement
 *
 * Corrects the predicted state using the detected bounding box.
 *
 * @param kf Kalman filter
 * @param bbox Detected bounding box (measurement)
 * @return Updated bounding box
 */
C_BoundingBox c_kalman_filter_update(C_KalmanFilter* kf, const C_BoundingBox* bbox);

/**
 * @brief Get current state as bounding box
 *
 * @param kf Kalman filter
 * @return Current state converted to bounding box
 */
C_BoundingBox c_kalman_filter_get_bbox(const C_KalmanFilter* kf);

/**
 * @brief Reset Kalman filter with a new bounding box
 *
 * @param kf Kalman filter
 * @param bbox New initial bounding box
 */
void c_kalman_filter_reset(C_KalmanFilter* kf, const C_BoundingBox* bbox);

/**
 * @brief Copy Kalman filter state
 *
 * @param src Source Kalman filter
 * @return New Kalman filter with copied state, or NULL on failure
 */
C_KalmanFilter* c_kalman_filter_copy(const C_KalmanFilter* src);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_ALGORITHMS_TRACKER_C_KALMAN_FILTER_H
