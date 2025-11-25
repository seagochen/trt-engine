/**
 * @file c_inference_drawer.cpp
 * @brief Inference result drawer implementation using OpenCV
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/visualization/c_inference_drawer.h"

#include <opencv2/opencv.hpp>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cmath>

/**
 * @brief Internal drawer structure
 */
struct C_InferenceDrawer {
    C_DrawingScheme scheme;
    C_DrawingConfig config;
    int blink_counter;

    C_InferenceDrawer(const C_DrawingScheme* s, const C_DrawingConfig* c)
        : blink_counter(0)
    {
        if (s) {
            scheme = *s;
        } else {
            std::memset(&scheme, 0, sizeof(scheme));
        }

        if (c) {
            config = *c;
        } else {
            config = c_drawing_config_default();
        }
    }
};

// Helper: Convert C_Color to OpenCV Scalar
static cv::Scalar color_to_scalar(const C_Color& c) {
    return cv::Scalar(c.b, c.g, c.r);
}

// Helper: Get bbox color by class index
static C_Color get_bbox_color(const C_InferenceDrawer* drawer, int class_id) {
    if (drawer->scheme.bbox_color_count == 0) {
        return C_COLOR_GREEN;
    }
    size_t idx = static_cast<size_t>(class_id) % drawer->scheme.bbox_color_count;
    return drawer->scheme.bbox_colors[idx];
}

// Helper: Get keypoint color by index
static C_Color get_kpt_color(const C_InferenceDrawer* drawer, int kpt_id) {
    for (size_t i = 0; i < drawer->scheme.keypoint_count; ++i) {
        if (drawer->scheme.keypoints[i].id == kpt_id) {
            return drawer->scheme.keypoints[i].color;
        }
    }
    return C_COLOR_WHITE;
}

// Helper: Clamp coordinate to image bounds
static int clamp_coord(int val, int min_val, int max_val) {
    return std::max(min_val, std::min(val, max_val));
}

extern "C" {

C_InferenceDrawer* c_inference_drawer_create(
    const C_DrawingScheme* scheme,
    const C_DrawingConfig* config
) {
    return new (std::nothrow) C_InferenceDrawer(scheme, config);
}

void c_inference_drawer_destroy(C_InferenceDrawer* drawer) {
    delete drawer;
}

bool c_inference_drawer_draw(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    const C_DrawSkeleton* skeletons,
    size_t skeleton_count,
    const C_ClassLabel* class_labels,
    size_t label_count
) {
    if (!drawer || !image_data || image_width <= 0 || image_height <= 0) {
        return false;
    }

    // Wrap raw data in cv::Mat (no copy)
    cv::Mat image(image_height, image_width, CV_8UC3, image_data);

    for (size_t i = 0; i < skeleton_count; ++i) {
        const C_DrawSkeleton& skel = skeletons[i];

        // Check confidence threshold
        if (skel.confidence < drawer->config.bbox_conf_threshold) {
            continue;
        }

        C_Color bbox_color = get_bbox_color(drawer, skel.class_id);

        // Draw bounding box
        if (drawer->config.draw_bbox) {
            int x1 = clamp_coord(skel.bbox_x1, 0, image_width - 1);
            int y1 = clamp_coord(skel.bbox_y1, 0, image_height - 1);
            int x2 = clamp_coord(skel.bbox_x2, 0, image_width - 1);
            int y2 = clamp_coord(skel.bbox_y2, 0, image_height - 1);

            cv::rectangle(
                image,
                cv::Point(x1, y1),
                cv::Point(x2, y2),
                color_to_scalar(bbox_color),
                drawer->config.bbox_thickness
            );
        }

        // Draw skeleton links first (so keypoints are on top)
        if (drawer->config.draw_skeleton && skel.keypoint_count > 0) {
            for (size_t l = 0; l < drawer->scheme.skeleton_link_count; ++l) {
                const C_SkeletonLink& link = drawer->scheme.skeleton_links[l];

                if (link.src_kpt_id < 0 || link.src_kpt_id >= static_cast<int>(skel.keypoint_count) ||
                    link.dst_kpt_id < 0 || link.dst_kpt_id >= static_cast<int>(skel.keypoint_count)) {
                    continue;
                }

                const C_DrawKeyPoint& src = skel.keypoints[link.src_kpt_id];
                const C_DrawKeyPoint& dst = skel.keypoints[link.dst_kpt_id];

                // Only draw if both points have sufficient confidence
                if (src.conf < drawer->config.kpt_conf_threshold ||
                    dst.conf < drawer->config.kpt_conf_threshold) {
                    continue;
                }

                int x1 = clamp_coord(static_cast<int>(std::round(src.x)), 0, image_width - 1);
                int y1 = clamp_coord(static_cast<int>(std::round(src.y)), 0, image_height - 1);
                int x2 = clamp_coord(static_cast<int>(std::round(dst.x)), 0, image_width - 1);
                int y2 = clamp_coord(static_cast<int>(std::round(dst.y)), 0, image_height - 1);

                cv::line(
                    image,
                    cv::Point(x1, y1),
                    cv::Point(x2, y2),
                    color_to_scalar(link.color),
                    drawer->config.link_thickness
                );
            }
        }

        // Draw keypoints
        if (drawer->config.draw_keypoints && skel.keypoint_count > 0) {
            for (size_t k = 0; k < skel.keypoint_count; ++k) {
                const C_DrawKeyPoint& kpt = skel.keypoints[k];

                if (kpt.conf < drawer->config.kpt_conf_threshold) {
                    continue;
                }

                int x = clamp_coord(static_cast<int>(std::round(kpt.x)), 0, image_width - 1);
                int y = clamp_coord(static_cast<int>(std::round(kpt.y)), 0, image_height - 1);

                C_Color kpt_color = get_kpt_color(drawer, static_cast<int>(k));

                cv::circle(
                    image,
                    cv::Point(x, y),
                    drawer->config.kpt_radius,
                    color_to_scalar(kpt_color),
                    -1  // Filled circle
                );
            }
        }

        // Draw direction arrow
        if (drawer->config.draw_direction && skel.has_direction) {
            // Calculate arrow origin (center of skeleton or face area)
            float origin_x, origin_y;
            if (skel.keypoint_count > 0 && skel.keypoints[0].conf > drawer->config.kpt_conf_threshold) {
                // Use nose as origin if available
                origin_x = skel.keypoints[0].x;
                origin_y = skel.keypoints[0].y;
            } else {
                // Use bbox center
                origin_x = (skel.bbox_x1 + skel.bbox_x2) / 2.0f;
                origin_y = (skel.bbox_y1 + skel.bbox_y2) / 2.0f;
            }

            float end_x = origin_x + skel.direction_x * skel.direction_modulus;
            float end_y = origin_y + skel.direction_y * skel.direction_modulus;

            int ox = clamp_coord(static_cast<int>(std::round(origin_x)), 0, image_width - 1);
            int oy = clamp_coord(static_cast<int>(std::round(origin_y)), 0, image_height - 1);
            int ex = clamp_coord(static_cast<int>(std::round(end_x)), 0, image_width - 1);
            int ey = clamp_coord(static_cast<int>(std::round(end_y)), 0, image_height - 1);

            cv::arrowedLine(
                image,
                cv::Point(ox, oy),
                cv::Point(ex, ey),
                cv::Scalar(0, 255, 255),  // Yellow
                2,
                cv::LINE_AA,
                0,
                0.3  // Tip length ratio
            );
        }

        // Draw label
        if (drawer->config.draw_labels) {
            char label_text[256];

            // Find class name
            const char* class_name = nullptr;
            for (size_t l = 0; l < label_count; ++l) {
                if (class_labels[l].class_id == skel.class_id) {
                    class_name = class_labels[l].name;
                    break;
                }
            }

            if (drawer->config.draw_track_id && skel.track_id >= 0) {
                if (class_name) {
                    snprintf(label_text, sizeof(label_text), "#%d %s %.2f",
                             skel.track_id, class_name, skel.confidence);
                } else {
                    snprintf(label_text, sizeof(label_text), "#%d cls%d %.2f",
                             skel.track_id, skel.class_id, skel.confidence);
                }
            } else {
                if (class_name) {
                    snprintf(label_text, sizeof(label_text), "%s %.2f",
                             class_name, skel.confidence);
                } else {
                    snprintf(label_text, sizeof(label_text), "cls%d %.2f",
                             skel.class_id, skel.confidence);
                }
            }

            // Calculate text size
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(
                label_text,
                cv::FONT_HERSHEY_SIMPLEX,
                drawer->config.font_scale,
                drawer->config.font_thickness,
                &baseline
            );

            // Position above bbox
            int label_x = clamp_coord(skel.bbox_x1, 0, image_width - 1);
            int label_y = clamp_coord(skel.bbox_y1 - 5, 0, image_height - 1);

            // Adjust if label would go above image
            if (label_y - text_size.height < 0) {
                label_y = text_size.height + 5;
            }

            // Draw background rectangle
            cv::rectangle(
                image,
                cv::Point(label_x, label_y - text_size.height - baseline),
                cv::Point(label_x + text_size.width, label_y + baseline),
                color_to_scalar(bbox_color),
                -1  // Filled
            );

            // Draw text
            cv::putText(
                image,
                label_text,
                cv::Point(label_x, label_y),
                cv::FONT_HERSHEY_SIMPLEX,
                drawer->config.font_scale,
                cv::Scalar(255, 255, 255),  // White text
                drawer->config.font_thickness
            );
        }
    }

    return true;
}

bool c_inference_drawer_draw_bbox(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    int x1, int y1, int x2, int y2,
    const C_Color* color,
    int thickness
) {
    if (!drawer || !image_data || image_width <= 0 || image_height <= 0) {
        return false;
    }

    cv::Mat image(image_height, image_width, CV_8UC3, image_data);

    C_Color c = color ? *color : C_COLOR_GREEN;
    int t = (thickness > 0) ? thickness : drawer->config.bbox_thickness;

    x1 = clamp_coord(x1, 0, image_width - 1);
    y1 = clamp_coord(y1, 0, image_height - 1);
    x2 = clamp_coord(x2, 0, image_width - 1);
    y2 = clamp_coord(y2, 0, image_height - 1);

    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color_to_scalar(c), t);

    return true;
}

bool c_inference_drawer_draw_keypoints(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    const C_DrawKeyPoint* keypoints,
    size_t keypoint_count
) {
    if (!drawer || !image_data || !keypoints || image_width <= 0 || image_height <= 0) {
        return false;
    }

    cv::Mat image(image_height, image_width, CV_8UC3, image_data);

    for (size_t k = 0; k < keypoint_count; ++k) {
        const C_DrawKeyPoint& kpt = keypoints[k];

        if (kpt.conf < drawer->config.kpt_conf_threshold) {
            continue;
        }

        int x = clamp_coord(static_cast<int>(std::round(kpt.x)), 0, image_width - 1);
        int y = clamp_coord(static_cast<int>(std::round(kpt.y)), 0, image_height - 1);

        C_Color kpt_color = get_kpt_color(drawer, static_cast<int>(k));

        cv::circle(image, cv::Point(x, y), drawer->config.kpt_radius, color_to_scalar(kpt_color), -1);
    }

    return true;
}

bool c_inference_drawer_draw_skeleton(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    const C_DrawKeyPoint* keypoints,
    size_t keypoint_count
) {
    if (!drawer || !image_data || !keypoints || image_width <= 0 || image_height <= 0) {
        return false;
    }

    cv::Mat image(image_height, image_width, CV_8UC3, image_data);

    for (size_t l = 0; l < drawer->scheme.skeleton_link_count; ++l) {
        const C_SkeletonLink& link = drawer->scheme.skeleton_links[l];

        if (link.src_kpt_id < 0 || link.src_kpt_id >= static_cast<int>(keypoint_count) ||
            link.dst_kpt_id < 0 || link.dst_kpt_id >= static_cast<int>(keypoint_count)) {
            continue;
        }

        const C_DrawKeyPoint& src = keypoints[link.src_kpt_id];
        const C_DrawKeyPoint& dst = keypoints[link.dst_kpt_id];

        if (src.conf < drawer->config.kpt_conf_threshold ||
            dst.conf < drawer->config.kpt_conf_threshold) {
            continue;
        }

        int x1 = clamp_coord(static_cast<int>(std::round(src.x)), 0, image_width - 1);
        int y1 = clamp_coord(static_cast<int>(std::round(src.y)), 0, image_height - 1);
        int x2 = clamp_coord(static_cast<int>(std::round(dst.x)), 0, image_width - 1);
        int y2 = clamp_coord(static_cast<int>(std::round(dst.y)), 0, image_height - 1);

        cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2),
                 color_to_scalar(link.color), drawer->config.link_thickness);
    }

    return true;
}

bool c_inference_drawer_draw_label(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    int x, int y,
    const char* label,
    const C_Color* bg_color,
    const C_Color* text_color
) {
    if (!drawer || !image_data || !label || image_width <= 0 || image_height <= 0) {
        return false;
    }

    cv::Mat image(image_height, image_width, CV_8UC3, image_data);

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(
        label,
        cv::FONT_HERSHEY_SIMPLEX,
        drawer->config.font_scale,
        drawer->config.font_thickness,
        &baseline
    );

    x = clamp_coord(x, 0, image_width - text_size.width);
    y = clamp_coord(y, text_size.height, image_height - 1);

    if (bg_color) {
        cv::rectangle(
            image,
            cv::Point(x, y - text_size.height - baseline),
            cv::Point(x + text_size.width, y + baseline),
            color_to_scalar(*bg_color),
            -1
        );
    }

    C_Color tc = text_color ? *text_color : C_COLOR_WHITE;
    cv::putText(
        image,
        label,
        cv::Point(x, y),
        cv::FONT_HERSHEY_SIMPLEX,
        drawer->config.font_scale,
        color_to_scalar(tc),
        drawer->config.font_thickness
    );

    return true;
}

bool c_inference_drawer_draw_direction(
    C_InferenceDrawer* drawer,
    unsigned char* image_data,
    int image_width,
    int image_height,
    float origin_x, float origin_y,
    float dir_x, float dir_y,
    float modulus,
    const C_Color* color
) {
    if (!drawer || !image_data || image_width <= 0 || image_height <= 0) {
        return false;
    }

    cv::Mat image(image_height, image_width, CV_8UC3, image_data);

    float end_x = origin_x + dir_x * modulus;
    float end_y = origin_y + dir_y * modulus;

    int ox = clamp_coord(static_cast<int>(std::round(origin_x)), 0, image_width - 1);
    int oy = clamp_coord(static_cast<int>(std::round(origin_y)), 0, image_height - 1);
    int ex = clamp_coord(static_cast<int>(std::round(end_x)), 0, image_width - 1);
    int ey = clamp_coord(static_cast<int>(std::round(end_y)), 0, image_height - 1);

    C_Color c = color ? *color : C_COLOR_YELLOW;

    cv::arrowedLine(
        image,
        cv::Point(ox, oy),
        cv::Point(ex, ey),
        color_to_scalar(c),
        2,
        cv::LINE_AA,
        0,
        0.3
    );

    return true;
}

void c_inference_drawer_set_config(
    C_InferenceDrawer* drawer,
    const C_DrawingConfig* config
) {
    if (!drawer || !config) return;
    drawer->config = *config;
}

C_DrawingConfig c_inference_drawer_get_config(const C_InferenceDrawer* drawer) {
    if (!drawer) return c_drawing_config_default();
    return drawer->config;
}

void c_inference_drawer_tick_blink(C_InferenceDrawer* drawer) {
    if (!drawer) return;
    drawer->blink_counter++;
}

} // extern "C"
