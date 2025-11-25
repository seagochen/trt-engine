/**
 * @file c_tracker_structures.h
 * @brief Core data structures for SORT/DeepSORT tracking algorithms
 *
 * This file defines the fundamental structures used in object tracking,
 * including bounding boxes, tracks, and tracker configurations.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_ALGORITHMS_TRACKER_C_TRACKER_STRUCTURES_H
#define TRTENGINE_V2_ALGORITHMS_TRACKER_C_TRACKER_STRUCTURES_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of features stored per track (for DeepSORT Re-ID)
 */
#define TRACKER_MAX_FEATURES 100

/**
 * @brief Default feature dimension for Re-ID embeddings
 */
#define TRACKER_DEFAULT_FEATURE_DIM 512

/**
 * @brief Bounding box in float format (x1, y1, x2, y2)
 */
typedef struct {
    float x1;       ///< Left X coordinate
    float y1;       ///< Top Y coordinate
    float x2;       ///< Right X coordinate
    float y2;       ///< Bottom Y coordinate
} C_BoundingBox;

/**
 * @brief Detection input for tracker
 *
 * Contains bounding box, confidence, class ID, and optional Re-ID features
 */
typedef struct {
    C_BoundingBox bbox;             ///< Detection bounding box
    float confidence;               ///< Detection confidence (0.0 - 1.0)
    int class_id;                   ///< Class index
    float* features;                ///< Optional Re-ID feature vector (NULL if not used)
    int feature_dim;                ///< Feature vector dimension (0 if not used)
} C_Detection;

/**
 * @brief Array of detections
 */
typedef struct {
    C_Detection* detections;        ///< Array of detections
    size_t count;                   ///< Number of detections
} C_DetectionArray;

/**
 * @brief Single track state
 *
 * Represents a tracked object with its state, history, and Re-ID features
 */
typedef struct {
    int track_id;                   ///< Unique track identifier
    C_BoundingBox bbox;             ///< Current bounding box (predicted or updated)
    int class_id;                   ///< Class index
    float confidence;               ///< Latest detection confidence

    int hits;                       ///< Number of successful updates (detections matched)
    int age;                        ///< Total age of the track (frames since creation)
    int time_since_update;          ///< Frames since last successful update

    bool is_confirmed;              ///< Whether track is confirmed (hits >= min_hits)
} C_Track;

/**
 * @brief Array of tracks (output from tracker)
 */
typedef struct {
    C_Track* tracks;                ///< Array of tracks
    size_t count;                   ///< Number of tracks
} C_TrackArray;

/**
 * @brief SORT tracker configuration
 */
typedef struct {
    int max_age;                    ///< Maximum frames to keep dead tracks (default: 5)
    int min_hits;                   ///< Minimum hits to confirm track (default: 1)
    float iou_threshold;            ///< IoU threshold for matching (default: 0.3)
    float conf_threshold;           ///< Minimum confidence to create new track (default: 0.5)
} C_SORTConfig;

/**
 * @brief DeepSORT tracker configuration
 */
typedef struct {
    int max_age;                    ///< Maximum frames to keep dead tracks (default: 30)
    int min_hits;                   ///< Minimum hits to confirm track (default: 3)
    float iou_threshold;            ///< IoU threshold for cascade stage 1 (default: 0.3)
    float reid_threshold;           ///< Re-ID cosine distance threshold (default: 0.4)
    float lambda_weight;            ///< Weight for combining IoU and Re-ID (default: 0.5)
    float conf_threshold;           ///< Minimum confidence to create new track (default: 0.5)
    int feature_dim;                ///< Feature vector dimension (default: 512)
} C_DeepSORTConfig;

/**
 * @brief Matching result structure
 */
typedef struct {
    int detection_idx;              ///< Index of matched detection
    int track_idx;                  ///< Index of matched track
} C_Match;

/**
 * @brief Association result from Hungarian algorithm
 */
typedef struct {
    C_Match* matches;               ///< Array of matched pairs
    size_t match_count;             ///< Number of matches
    int* unmatched_detections;      ///< Indices of unmatched detections
    size_t unmatched_det_count;     ///< Number of unmatched detections
    int* unmatched_tracks;          ///< Indices of unmatched tracks
    size_t unmatched_trk_count;     ///< Number of unmatched tracks
} C_AssociationResult;

/**
 * @brief Get default SORT configuration
 * @return Default SORT configuration
 */
C_SORTConfig c_sort_config_default(void);

/**
 * @brief Get default DeepSORT configuration
 * @return Default DeepSORT configuration
 */
C_DeepSORTConfig c_deepsort_config_default(void);

/**
 * @brief Free association result memory
 * @param result Association result to free
 */
void c_association_result_free(C_AssociationResult* result);

/**
 * @brief Free detection array memory
 * @param array Detection array to free
 */
void c_detection_array_free(C_DetectionArray* array);

/**
 * @brief Free track array memory
 * @param array Track array to free
 */
void c_track_array_free(C_TrackArray* array);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_ALGORITHMS_TRACKER_C_TRACKER_STRUCTURES_H
