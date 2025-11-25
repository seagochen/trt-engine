/**
 * @file c_deepsort_tracker.h
 * @brief DeepSORT tracking algorithm implementation
 *
 * DeepSORT extends SORT by integrating appearance information (Re-ID features)
 * through a cascaded matching strategy. It provides more robust tracking
 * especially for occluded objects.
 *
 * Key features:
 * - Cascaded matching: First matches recent tracks with IoU, then uses
 *   combined IoU + Re-ID cost for older tracks
 * - Feature gallery: Maintains a gallery of recent appearance features per track
 * - Mahalanobis distance gating for motion-based filtering
 *
 * Reference: Wojke et al., "Simple Online and Realtime Tracking with a
 *            Deep Association Metric", ICIP 2017
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_ALGORITHMS_TRACKER_C_DEEPSORT_TRACKER_H
#define TRTENGINE_V2_ALGORITHMS_TRACKER_C_DEEPSORT_TRACKER_H

#include "c_tracker_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque DeepSORT tracker handle
 */
typedef struct C_DeepSORTTracker C_DeepSORTTracker;

/**
 * @brief Create a new DeepSORT tracker
 *
 * @param config Tracker configuration (NULL for default)
 * @return Pointer to new tracker, or NULL on failure
 */
C_DeepSORTTracker* c_deepsort_tracker_create(const C_DeepSORTConfig* config);

/**
 * @brief Destroy DeepSORT tracker and free all resources
 *
 * @param tracker Tracker to destroy
 */
void c_deepsort_tracker_destroy(C_DeepSORTTracker* tracker);

/**
 * @brief Update tracker with new detections and features
 *
 * Performs cascaded matching:
 * 1. Predicts next positions for existing tracks
 * 2. Stage 1: Match recent tracks (time_since_update == 1) using IoU only
 * 3. Stage 2: Match remaining detections with older tracks using combined
 *    IoU + Re-ID cosine distance
 * 4. Updates matched tracks and their feature galleries
 * 5. Creates new tracks for unmatched high-confidence detections
 * 6. Removes dead tracks
 *
 * @param tracker DeepSORT tracker
 * @param detections Array of detections with features
 * @param det_count Number of detections
 * @param result Output track array (caller must free with c_track_array_free)
 * @return true on success, false on failure
 */
bool c_deepsort_tracker_update(
    C_DeepSORTTracker* tracker,
    const C_Detection* detections,
    size_t det_count,
    C_TrackArray* result
);

/**
 * @brief Update tracker without Re-ID features (falls back to SORT-like behavior)
 *
 * Use this when Re-ID features are not available. The tracker will use
 * only IoU-based matching.
 *
 * @param tracker DeepSORT tracker
 * @param detections Array of detections (features can be NULL)
 * @param det_count Number of detections
 * @param result Output track array (caller must free)
 * @return true on success, false on failure
 */
bool c_deepsort_tracker_update_no_features(
    C_DeepSORTTracker* tracker,
    const C_Detection* detections,
    size_t det_count,
    C_TrackArray* result
);

/**
 * @brief Get all active tracks (including unconfirmed)
 *
 * @param tracker DeepSORT tracker
 * @param result Output track array (caller must free)
 * @return true on success, false on failure
 */
bool c_deepsort_tracker_get_tracks(
    const C_DeepSORTTracker* tracker,
    C_TrackArray* result
);

/**
 * @brief Get only confirmed tracks
 *
 * @param tracker DeepSORT tracker
 * @param result Output track array (caller must free)
 * @return true on success, false on failure
 */
bool c_deepsort_tracker_get_confirmed_tracks(
    const C_DeepSORTTracker* tracker,
    C_TrackArray* result
);

/**
 * @brief Reset tracker (clear all tracks and feature galleries)
 *
 * @param tracker DeepSORT tracker
 */
void c_deepsort_tracker_reset(C_DeepSORTTracker* tracker);

/**
 * @brief Get current configuration
 *
 * @param tracker DeepSORT tracker
 * @return Current configuration
 */
C_DeepSORTConfig c_deepsort_tracker_get_config(const C_DeepSORTTracker* tracker);

/**
 * @brief Get next track ID that will be assigned
 *
 * @param tracker DeepSORT tracker
 * @return Next track ID
 */
int c_deepsort_tracker_get_next_id(const C_DeepSORTTracker* tracker);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_ALGORITHMS_TRACKER_C_DEEPSORT_TRACKER_H
