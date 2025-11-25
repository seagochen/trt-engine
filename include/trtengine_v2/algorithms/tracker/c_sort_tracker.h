/**
 * @file c_sort_tracker.h
 * @brief SORT (Simple Online and Realtime Tracking) algorithm implementation
 *
 * SORT is a pragmatic approach to multiple object tracking with a focus on
 * simple, effective algorithms. It uses a Kalman filter for state estimation
 * and the Hungarian algorithm for data association based on IoU.
 *
 * Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_ALGORITHMS_TRACKER_C_SORT_TRACKER_H
#define TRTENGINE_V2_ALGORITHMS_TRACKER_C_SORT_TRACKER_H

#include "c_tracker_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque SORT tracker handle
 */
typedef struct C_SORTTracker C_SORTTracker;

/**
 * @brief Create a new SORT tracker
 *
 * @param config Tracker configuration (NULL for default)
 * @return Pointer to new tracker, or NULL on failure
 */
C_SORTTracker* c_sort_tracker_create(const C_SORTConfig* config);

/**
 * @brief Destroy SORT tracker and free all resources
 *
 * @param tracker Tracker to destroy
 */
void c_sort_tracker_destroy(C_SORTTracker* tracker);

/**
 * @brief Update tracker with new detections
 *
 * This is the main tracking function. Given a set of detections:
 * 1. Predicts next positions for existing tracks
 * 2. Associates detections with tracks using Hungarian algorithm on IoU
 * 3. Updates matched tracks
 * 4. Creates new tracks for unmatched detections
 * 5. Removes dead tracks
 *
 * @param tracker SORT tracker
 * @param detections Array of detections (bboxes with confidence)
 * @param det_count Number of detections
 * @param result Output track array (caller must free with c_track_array_free)
 * @return true on success, false on failure
 */
bool c_sort_tracker_update(
    C_SORTTracker* tracker,
    const C_Detection* detections,
    size_t det_count,
    C_TrackArray* result
);

/**
 * @brief Get all active tracks (including unconfirmed)
 *
 * @param tracker SORT tracker
 * @param result Output track array (caller must free)
 * @return true on success, false on failure
 */
bool c_sort_tracker_get_tracks(
    const C_SORTTracker* tracker,
    C_TrackArray* result
);

/**
 * @brief Get only confirmed tracks
 *
 * @param tracker SORT tracker
 * @param result Output track array (caller must free)
 * @return true on success, false on failure
 */
bool c_sort_tracker_get_confirmed_tracks(
    const C_SORTTracker* tracker,
    C_TrackArray* result
);

/**
 * @brief Reset tracker (clear all tracks)
 *
 * @param tracker SORT tracker
 */
void c_sort_tracker_reset(C_SORTTracker* tracker);

/**
 * @brief Get current configuration
 *
 * @param tracker SORT tracker
 * @return Current configuration
 */
C_SORTConfig c_sort_tracker_get_config(const C_SORTTracker* tracker);

/**
 * @brief Get next track ID that will be assigned
 *
 * @param tracker SORT tracker
 * @return Next track ID
 */
int c_sort_tracker_get_next_id(const C_SORTTracker* tracker);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_ALGORITHMS_TRACKER_C_SORT_TRACKER_H
