/**
 * @file c_inference_drawer.h
 * @brief Inference result drawer - main visualization API
 *
 * This module provides functions to draw inference results (bounding boxes,
 * keypoints, skeleton links, labels) on images using OpenCV.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_VISUALIZATION_C_INFERENCE_DRAWER_H
#define TRTENGINE_V2_VISUALIZATION_C_INFERENCE_DRAWER_H

#include "c_drawing_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque drawer handle
 */
typedef struct C_InferenceDrawer C_InferenceDrawer;

/**
 * @brief Create a new inference drawer
 *
 * @param scheme Drawing scheme (colors, skeleton links, etc.)
 * @param config Drawing configuration (NULL for default)
 * @return Pointer to drawer, or NULL on failure
 */
C_InferenceDrawer* c_inference_drawer_create(
    const C_DrawingScheme* scheme,
    const C_DrawingConfig* config
);

/**
 * @brief Destroy drawer and free resources
 *
 * @param drawer Drawer to destroy
 */
void c_inference_drawer_destroy(C_InferenceDrawer* drawer);

/**
 * @brief Draw skeletons on image
 *
 * This is the main drawing function. It draws all specified elements
 * (bboxes, keypoints, skeleton links, labels) based on the configuration.
 *
 * @param drawer Inference drawer
 * @param image_data BGR image data (will be modified in-place)
 * @param image_width Image width in pixels
 * @param image_height Image height in pixels
 * @param skeletons Array of skeletons to draw
 * @param skeleton_count Number of skeletons
 * @param class_labels Optional class label mapping (NULL to use class_id as label)
 * @param label_count Number of class labels
 * @return true on success, false on failure
 */
bool c_inference_drawer_draw(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    const C_DrawSkeleton* skeletons,
    size_t skeleton_count,
    const C_ClassLabel* class_labels,
    size_t label_count
);

/**
 * @brief Draw a single bounding box
 *
 * @param drawer Inference drawer
 * @param image_data BGR image data
 * @param image_width Image width
 * @param image_height Image height
 * @param x1 Left X coordinate
 * @param y1 Top Y coordinate
 * @param x2 Right X coordinate
 * @param y2 Bottom Y coordinate
 * @param color Box color (NULL for default based on class)
 * @param thickness Line thickness (-1 for config default)
 * @return true on success
 */
bool c_inference_drawer_draw_bbox(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    int x1, int y1, int x2, int y2,
    const C_Color* color,
    int thickness
);

/**
 * @brief Draw keypoints
 *
 * @param drawer Inference drawer
 * @param image_data BGR image data
 * @param image_width Image width
 * @param image_height Image height
 * @param keypoints Array of keypoints
 * @param keypoint_count Number of keypoints
 * @return true on success
 */
bool c_inference_drawer_draw_keypoints(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    const C_DrawKeyPoint* keypoints,
    size_t keypoint_count
);

/**
 * @brief Draw skeleton links
 *
 * @param drawer Inference drawer
 * @param image_data BGR image data
 * @param image_width Image width
 * @param image_height Image height
 * @param keypoints Array of keypoints (skeleton endpoints)
 * @param keypoint_count Number of keypoints
 * @return true on success
 */
bool c_inference_drawer_draw_skeleton(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    const C_DrawKeyPoint* keypoints,
    size_t keypoint_count
);

/**
 * @brief Draw a label at a position
 *
 * @param drawer Inference drawer
 * @param image_data BGR image data
 * @param image_width Image width
 * @param image_height Image height
 * @param x X position (top-left of label box)
 * @param y Y position (top-left of label box)
 * @param label Label text
 * @param bg_color Background color (NULL for no background)
 * @param text_color Text color (NULL for white)
 * @return true on success
 */
bool c_inference_drawer_draw_label(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    int x, int y,
    const char* label,
    const C_Color* bg_color,
    const C_Color* text_color
);

/**
 * @brief Draw direction arrow
 *
 * @param drawer Inference drawer
 * @param image_data BGR image data
 * @param image_width Image width
 * @param image_height Image height
 * @param origin_x Arrow origin X
 * @param origin_y Arrow origin Y
 * @param dir_x Direction unit vector X
 * @param dir_y Direction unit vector Y
 * @param modulus Arrow length
 * @param color Arrow color (NULL for yellow)
 * @return true on success
 */
bool c_inference_drawer_draw_direction(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    float origin_x, float origin_y,
    float dir_x, float dir_y,
    float modulus,
    const C_Color* color
);

/**
 * @brief Update drawing configuration
 *
 * @param drawer Inference drawer
 * @param config New configuration
 */
void c_inference_drawer_set_config(
    C_InferenceDrawer* drawer,
    const C_DrawingConfig* config
);

/**
 * @brief Get current drawing configuration
 *
 * @param drawer Inference drawer
 * @return Current configuration
 */
C_DrawingConfig c_inference_drawer_get_config(const C_InferenceDrawer* drawer);

/**
 * @brief Increment blink counter (for highlight effects)
 *
 * @param drawer Inference drawer
 */
void c_inference_drawer_tick_blink(C_InferenceDrawer* drawer);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_VISUALIZATION_C_INFERENCE_DRAWER_H
