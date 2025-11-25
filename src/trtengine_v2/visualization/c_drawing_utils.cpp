/**
 * @file c_drawing_utils.cpp
 * @brief Drawing utility functions and constants
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/visualization/c_drawing_structures.h"

extern "C" {

// Predefined colors (BGR format)
const C_Color C_COLOR_RED     = {0, 0, 255};
const C_Color C_COLOR_GREEN   = {0, 255, 0};
const C_Color C_COLOR_BLUE    = {255, 0, 0};
const C_Color C_COLOR_YELLOW  = {0, 255, 255};
const C_Color C_COLOR_CYAN    = {255, 255, 0};
const C_Color C_COLOR_MAGENTA = {255, 0, 255};
const C_Color C_COLOR_WHITE   = {255, 255, 255};
const C_Color C_COLOR_BLACK   = {0, 0, 0};

C_DrawingConfig c_drawing_config_default(void) {
    C_DrawingConfig config;
    config.bbox_conf_threshold = 0.5f;
    config.kpt_conf_threshold = 0.5f;
    config.bbox_thickness = 2;
    config.kpt_radius = 5;
    config.link_thickness = 2;
    config.font_scale = 0.6f;
    config.font_thickness = 1;
    config.draw_bbox = true;
    config.draw_keypoints = true;
    config.draw_skeleton = true;
    config.draw_labels = true;
    config.draw_track_id = true;
    config.draw_direction = false;
    return config;
}

C_Color c_color_rgb(unsigned char r, unsigned char g, unsigned char b) {
    C_Color c;
    c.r = r;
    c.g = g;
    c.b = b;
    return c;
}

C_Color c_color_bgr(unsigned char b, unsigned char g, unsigned char r) {
    C_Color c;
    c.b = b;
    c.g = g;
    c.r = r;
    return c;
}

} // extern "C"
