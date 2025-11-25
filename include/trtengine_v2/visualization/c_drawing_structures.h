/**
 * @file c_drawing_structures.h
 * @brief Data structures for visualization/drawing module
 *
 * This file defines structures for drawing inference results including
 * bounding boxes, keypoints, skeleton links, and labels.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#ifndef TRTENGINE_V2_VISUALIZATION_C_DRAWING_STRUCTURES_H
#define TRTENGINE_V2_VISUALIZATION_C_DRAWING_STRUCTURES_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of keypoints supported
 */
#define VIS_MAX_KEYPOINTS 32

/**
 * @brief Maximum number of skeleton links
 */
#define VIS_MAX_SKELETON_LINKS 64

/**
 * @brief Maximum number of bbox colors
 */
#define VIS_MAX_BBOX_COLORS 32

/**
 * @brief Maximum class name length
 */
#define VIS_MAX_CLASS_NAME_LEN 64

/**
 * @brief BGR color (OpenCV format)
 */
typedef struct {
    unsigned char b;    ///< Blue channel (0-255)
    unsigned char g;    ///< Green channel (0-255)
    unsigned char r;    ///< Red channel (0-255)
} C_Color;

/**
 * @brief Keypoint schema (color and name for each joint)
 */
typedef struct {
    int id;                             ///< Keypoint index
    char name[VIS_MAX_CLASS_NAME_LEN];  ///< Keypoint name
    C_Color color;                      ///< Display color
} C_KeyPointSchema;

/**
 * @brief Skeleton link definition
 */
typedef struct {
    int src_kpt_id;                     ///< Source keypoint index
    int dst_kpt_id;                     ///< Destination keypoint index
    C_Color color;                      ///< Link color
    char description[128];              ///< Link description
} C_SkeletonLink;

/**
 * @brief Drawing scheme containing all visual configurations
 */
typedef struct {
    C_KeyPointSchema keypoints[VIS_MAX_KEYPOINTS];  ///< Keypoint schemas
    size_t keypoint_count;                          ///< Number of keypoints

    C_SkeletonLink skeleton_links[VIS_MAX_SKELETON_LINKS];  ///< Skeleton links
    size_t skeleton_link_count;                             ///< Number of links

    C_Color bbox_colors[VIS_MAX_BBOX_COLORS];   ///< Bounding box colors
    size_t bbox_color_count;                    ///< Number of bbox colors
} C_DrawingScheme;

/**
 * @brief Single keypoint with coordinates and confidence
 */
typedef struct {
    float x;        ///< X coordinate
    float y;        ///< Y coordinate
    float conf;     ///< Confidence score (0.0 - 1.0)
} C_DrawKeyPoint;

/**
 * @brief Skeleton to be drawn (combines bbox, keypoints, and metadata)
 */
typedef struct {
    // Bounding box (integer coordinates)
    int bbox_x1;                        ///< Left X
    int bbox_y1;                        ///< Top Y
    int bbox_x2;                        ///< Right X
    int bbox_y2;                        ///< Bottom Y

    // Detection info
    int class_id;                       ///< Class index
    float confidence;                   ///< Detection confidence
    int track_id;                       ///< Track ID (-1 if not tracked)

    // Keypoints
    C_DrawKeyPoint keypoints[VIS_MAX_KEYPOINTS];  ///< Keypoint array
    size_t keypoint_count;                        ///< Number of keypoints

    // Optional: Face/body direction
    bool has_direction;                 ///< Whether direction info is available
    float direction_x;                  ///< Direction unit vector X
    float direction_y;                  ///< Direction unit vector Y
    float direction_modulus;            ///< Direction magnitude (arrow length)
} C_DrawSkeleton;

/**
 * @brief Drawing configuration options
 */
typedef struct {
    float bbox_conf_threshold;      ///< Min confidence to draw bbox (default: 0.5)
    float kpt_conf_threshold;       ///< Min confidence to draw keypoint (default: 0.5)
    int bbox_thickness;             ///< Bounding box line thickness (default: 2)
    int kpt_radius;                 ///< Keypoint circle radius (default: 5)
    int link_thickness;             ///< Skeleton link thickness (default: 2)
    float font_scale;               ///< Label font scale (default: 0.6)
    int font_thickness;             ///< Label font thickness (default: 1)
    bool draw_bbox;                 ///< Draw bounding boxes
    bool draw_keypoints;            ///< Draw keypoints
    bool draw_skeleton;             ///< Draw skeleton links
    bool draw_labels;               ///< Draw labels (class + confidence)
    bool draw_track_id;             ///< Draw track ID in label
    bool draw_direction;            ///< Draw direction arrow
} C_DrawingConfig;

/**
 * @brief Class label mapping entry
 */
typedef struct {
    int class_id;                       ///< Class index
    char name[VIS_MAX_CLASS_NAME_LEN];  ///< Class name
} C_ClassLabel;

/**
 * @brief Get default drawing configuration
 * @return Default configuration
 */
C_DrawingConfig c_drawing_config_default(void);

/**
 * @brief Create a color from RGB values
 * @param r Red (0-255)
 * @param g Green (0-255)
 * @param b Blue (0-255)
 * @return BGR color
 */
C_Color c_color_rgb(unsigned char r, unsigned char g, unsigned char b);

/**
 * @brief Create a color from BGR values
 * @param b Blue (0-255)
 * @param g Green (0-255)
 * @param r Red (0-255)
 * @return BGR color
 */
C_Color c_color_bgr(unsigned char b, unsigned char g, unsigned char r);

/**
 * @brief Predefined colors
 */
extern const C_Color C_COLOR_RED;
extern const C_Color C_COLOR_GREEN;
extern const C_Color C_COLOR_BLUE;
extern const C_Color C_COLOR_YELLOW;
extern const C_Color C_COLOR_CYAN;
extern const C_Color C_COLOR_MAGENTA;
extern const C_Color C_COLOR_WHITE;
extern const C_Color C_COLOR_BLACK;

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_VISUALIZATION_C_DRAWING_STRUCTURES_H
