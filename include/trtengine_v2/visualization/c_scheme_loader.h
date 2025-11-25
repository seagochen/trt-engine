/**
 * @file c_scheme_loader.h
 * @brief Drawing scheme loader from JSON files
 *
 * Loads drawing configurations (keypoint colors, skeleton links, bbox colors)
 * from JSON schema files.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_VISUALIZATION_C_SCHEME_LOADER_H
#define TRTENGINE_V2_VISUALIZATION_C_SCHEME_LOADER_H

#include "c_drawing_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load drawing scheme from JSON file
 *
 * JSON format expected:
 * {
 *   "kpt_color_map": {
 *     "0": { "name": "Nose", "color": [B, G, R] },
 *     ...
 *   },
 *   "skeleton_map": [
 *     { "srt_kpt_id": 0, "dst_kpt_id": 1, "color": [B, G, R], "description": "..." },
 *     ...
 *   ],
 *   "bbox_color": [
 *     { "color": [B, G, R], "name": "..." },
 *     ...
 *   ]
 * }
 *
 * @param json_path Path to JSON schema file
 * @param scheme Output scheme structure
 * @return true on success, false on failure
 */
bool c_scheme_load_from_json(const char* json_path, C_DrawingScheme* scheme);

/**
 * @brief Get default COCO pose drawing scheme
 *
 * Returns a pre-configured scheme for COCO 17-keypoint pose format
 * with standard colors for body parts.
 *
 * @param scheme Output scheme structure
 */
void c_scheme_get_coco_pose(C_DrawingScheme* scheme);

/**
 * @brief Get a simple drawing scheme (no keypoints, basic bbox colors)
 *
 * @param scheme Output scheme structure
 */
void c_scheme_get_simple(C_DrawingScheme* scheme);

/**
 * @brief Print scheme information for debugging
 *
 * @param scheme Scheme to print
 */
void c_scheme_print(const C_DrawingScheme* scheme);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_VISUALIZATION_C_SCHEME_LOADER_H
