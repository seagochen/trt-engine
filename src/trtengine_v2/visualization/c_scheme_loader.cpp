/**
 * @file c_scheme_loader.cpp
 * @brief Drawing scheme loader implementation
 *
 * Loads and parses JSON schema files for visualization configuration.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/visualization/c_scheme_loader.h"
#include "trtengine_v2/utils/logger.h"

#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <string>

#define SCHEME_LOADER_MODULE "SchemeLoader"

// Simple JSON parsing (using a lightweight approach without external dependencies)
// For production use, consider using nlohmann/json or rapidjson

namespace {

// Skip whitespace
const char* skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        ++p;
    }
    return p;
}

// Parse a JSON integer
bool parse_int(const char*& p, int& value) {
    p = skip_ws(p);
    if (!*p) return false;

    bool negative = false;
    if (*p == '-') {
        negative = true;
        ++p;
    }

    if (!(*p >= '0' && *p <= '9')) return false;

    value = 0;
    while (*p >= '0' && *p <= '9') {
        value = value * 10 + (*p - '0');
        ++p;
    }

    if (negative) value = -value;
    return true;
}

// Parse a JSON string (simple version)
bool parse_string(const char*& p, char* out, size_t max_len) {
    p = skip_ws(p);
    if (*p != '"') return false;
    ++p;

    size_t i = 0;
    while (*p && *p != '"' && i < max_len - 1) {
        if (*p == '\\' && *(p + 1)) {
            ++p;
            switch (*p) {
                case 'n': out[i++] = '\n'; break;
                case 't': out[i++] = '\t'; break;
                case '"': out[i++] = '"'; break;
                case '\\': out[i++] = '\\'; break;
                default: out[i++] = *p; break;
            }
        } else {
            out[i++] = *p;
        }
        ++p;
    }
    out[i] = '\0';

    if (*p == '"') ++p;
    return true;
}

// Parse a color array [B, G, R]
bool parse_color_array(const char*& p, C_Color& color) {
    p = skip_ws(p);
    if (*p != '[') return false;
    ++p;

    int b, g, r;
    if (!parse_int(p, b)) return false;
    p = skip_ws(p);
    if (*p == ',') ++p;

    if (!parse_int(p, g)) return false;
    p = skip_ws(p);
    if (*p == ',') ++p;

    if (!parse_int(p, r)) return false;
    p = skip_ws(p);

    if (*p == ']') ++p;

    color.b = static_cast<unsigned char>(b);
    color.g = static_cast<unsigned char>(g);
    color.r = static_cast<unsigned char>(r);
    return true;
}

// Find a key in JSON object
const char* find_key(const char* json, const char* key) {
    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\"", key);

    const char* pos = strstr(json, search_key);
    if (!pos) return nullptr;

    pos += strlen(search_key);
    pos = skip_ws(pos);
    if (*pos == ':') ++pos;
    return skip_ws(pos);
}

} // anonymous namespace

extern "C" {

bool c_scheme_load_from_json(const char* json_path, C_DrawingScheme* scheme) {
    if (!json_path || !scheme) return false;

    // Initialize scheme
    std::memset(scheme, 0, sizeof(C_DrawingScheme));

    // Read file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        LOG_ERROR(SCHEME_LOADER_MODULE, std::string("Failed to open schema file: ") + json_path);
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    const char* json = json_str.c_str();

    // Parse kpt_color_map
    const char* kpt_map = find_key(json, "kpt_color_map");
    if (kpt_map && *kpt_map == '{') {
        ++kpt_map;
        while (*kpt_map && *kpt_map != '}' && scheme->keypoint_count < VIS_MAX_KEYPOINTS) {
            kpt_map = skip_ws(kpt_map);
            if (*kpt_map == '"') {
                // Parse keypoint ID
                char id_str[16];
                if (parse_string(kpt_map, id_str, sizeof(id_str))) {
                    int kpt_id = atoi(id_str);
                    kpt_map = skip_ws(kpt_map);
                    if (*kpt_map == ':') ++kpt_map;
                    kpt_map = skip_ws(kpt_map);

                    if (*kpt_map == '{') {
                        C_KeyPointSchema& kpt = scheme->keypoints[scheme->keypoint_count];
                        kpt.id = kpt_id;

                        // Find name
                        const char* name_pos = find_key(kpt_map, "name");
                        if (name_pos) {
                            parse_string(name_pos, kpt.name, sizeof(kpt.name));
                        }

                        // Find color
                        const char* color_pos = find_key(kpt_map, "color");
                        if (color_pos) {
                            parse_color_array(color_pos, kpt.color);
                        }

                        scheme->keypoint_count++;

                        // Skip to end of object
                        int depth = 1;
                        ++kpt_map;
                        while (*kpt_map && depth > 0) {
                            if (*kpt_map == '{') depth++;
                            else if (*kpt_map == '}') depth--;
                            ++kpt_map;
                        }
                    }
                }
            }
            kpt_map = skip_ws(kpt_map);
            if (*kpt_map == ',') ++kpt_map;
        }
    }

    // Parse skeleton_map
    const char* skel_map = find_key(json, "skeleton_map");
    if (skel_map && *skel_map == '[') {
        ++skel_map;
        while (*skel_map && *skel_map != ']' && scheme->skeleton_link_count < VIS_MAX_SKELETON_LINKS) {
            skel_map = skip_ws(skel_map);
            if (*skel_map == '{') {
                C_SkeletonLink& link = scheme->skeleton_links[scheme->skeleton_link_count];

                // Parse src keypoint
                const char* src_pos = find_key(skel_map, "srt_kpt_id");
                if (src_pos) parse_int(src_pos, link.src_kpt_id);

                // Parse dst keypoint
                const char* dst_pos = find_key(skel_map, "dst_kpt_id");
                if (dst_pos) parse_int(dst_pos, link.dst_kpt_id);

                // Parse color
                const char* color_pos = find_key(skel_map, "color");
                if (color_pos) parse_color_array(color_pos, link.color);

                // Parse description
                const char* desc_pos = find_key(skel_map, "description");
                if (desc_pos) parse_string(desc_pos, link.description, sizeof(link.description));

                scheme->skeleton_link_count++;

                // Skip to end of object
                int depth = 1;
                ++skel_map;
                while (*skel_map && depth > 0) {
                    if (*skel_map == '{') depth++;
                    else if (*skel_map == '}') depth--;
                    ++skel_map;
                }
            }
            skel_map = skip_ws(skel_map);
            if (*skel_map == ',') ++skel_map;
        }
    }

    // Parse bbox_color
    const char* bbox_colors = find_key(json, "bbox_color");
    if (bbox_colors && *bbox_colors == '[') {
        ++bbox_colors;
        while (*bbox_colors && *bbox_colors != ']' && scheme->bbox_color_count < VIS_MAX_BBOX_COLORS) {
            bbox_colors = skip_ws(bbox_colors);
            if (*bbox_colors == '{') {
                const char* color_pos = find_key(bbox_colors, "color");
                if (color_pos) {
                    parse_color_array(color_pos, scheme->bbox_colors[scheme->bbox_color_count]);
                    scheme->bbox_color_count++;
                }

                // Skip to end of object
                int depth = 1;
                ++bbox_colors;
                while (*bbox_colors && depth > 0) {
                    if (*bbox_colors == '{') depth++;
                    else if (*bbox_colors == '}') depth--;
                    ++bbox_colors;
                }
            }
            bbox_colors = skip_ws(bbox_colors);
            if (*bbox_colors == ',') ++bbox_colors;
        }
    }

    char log_buf[256];
    snprintf(log_buf, sizeof(log_buf), "Loaded schema: %zu keypoints, %zu skeleton links, %zu bbox colors",
             scheme->keypoint_count, scheme->skeleton_link_count, scheme->bbox_color_count);
    LOG_INFO(SCHEME_LOADER_MODULE, log_buf);

    return true;
}

void c_scheme_get_coco_pose(C_DrawingScheme* scheme) {
    if (!scheme) return;
    std::memset(scheme, 0, sizeof(C_DrawingScheme));

    // COCO 17 keypoints
    const struct { const char* name; unsigned char b, g, r; } kpts[] = {
        {"Nose",           0, 0, 255},
        {"Right Eye",      255, 0, 0},
        {"Left Eye",       255, 0, 0},
        {"Right Ear",      0, 255, 0},
        {"Left Ear",       0, 255, 0},
        {"Right Shoulder", 193, 182, 255},
        {"Left Shoulder",  193, 182, 255},
        {"Right Elbow",    16, 144, 247},
        {"Left Elbow",     16, 144, 247},
        {"Right Wrist",    1, 240, 255},
        {"Left Wrist",     1, 240, 255},
        {"Right Hip",      140, 47, 240},
        {"Left Hip",       140, 47, 240},
        {"Right Knee",     223, 155, 60},
        {"Left Knee",      223, 155, 60},
        {"Right Ankle",    139, 0, 0},
        {"Left Ankle",     139, 0, 0},
    };

    scheme->keypoint_count = 17;
    for (int i = 0; i < 17; ++i) {
        scheme->keypoints[i].id = i;
        strncpy(scheme->keypoints[i].name, kpts[i].name, VIS_MAX_CLASS_NAME_LEN - 1);
        scheme->keypoints[i].color = {kpts[i].b, kpts[i].g, kpts[i].r};
    }

    // COCO skeleton links
    const struct { int src, dst; unsigned char b, g, r; } links[] = {
        {0, 1, 0, 0, 255},      // Nose -> Right Eye
        {0, 2, 0, 0, 255},      // Nose -> Left Eye
        {1, 3, 0, 0, 255},      // Right Eye -> Right Ear
        {2, 4, 0, 0, 255},      // Left Eye -> Left Ear
        {15, 13, 0, 100, 255},  // Right Ankle -> Right Knee
        {13, 11, 0, 255, 0},    // Right Knee -> Right Hip
        {16, 14, 255, 0, 0},    // Left Ankle -> Left Knee
        {14, 12, 0, 0, 255},    // Left Knee -> Left Hip
        {11, 12, 122, 160, 255},// Right Hip -> Left Hip
        {5, 11, 139, 0, 139},   // Right Shoulder -> Right Hip
        {6, 12, 237, 149, 100}, // Left Shoulder -> Left Hip
        {5, 6, 152, 251, 152},  // Right Shoulder -> Left Shoulder
        {5, 7, 148, 0, 69},     // Right Shoulder -> Right Elbow
        {6, 8, 0, 75, 255},     // Left Shoulder -> Left Elbow
        {7, 9, 56, 230, 25},    // Right Elbow -> Right Wrist
        {8, 10, 0, 240, 240},   // Left Elbow -> Left Wrist
    };

    scheme->skeleton_link_count = 16;
    for (int i = 0; i < 16; ++i) {
        scheme->skeleton_links[i].src_kpt_id = links[i].src;
        scheme->skeleton_links[i].dst_kpt_id = links[i].dst;
        scheme->skeleton_links[i].color = {links[i].b, links[i].g, links[i].r};
    }

    // Default bbox colors
    const unsigned char bbox_cols[][3] = {
        {0, 0, 230},    // Red
        {230, 0, 0},    // Blue
        {0, 230, 0},    // Green
        {230, 230, 0},  // Cyan
        {230, 0, 230},  // Magenta
        {0, 230, 230},  // Yellow
        {128, 0, 128},  // Purple
        {128, 128, 0},  // Olive
    };

    scheme->bbox_color_count = 8;
    for (int i = 0; i < 8; ++i) {
        scheme->bbox_colors[i] = {bbox_cols[i][0], bbox_cols[i][1], bbox_cols[i][2]};
    }
}

void c_scheme_get_simple(C_DrawingScheme* scheme) {
    if (!scheme) return;
    std::memset(scheme, 0, sizeof(C_DrawingScheme));

    // Simple bbox colors only
    const unsigned char bbox_cols[][3] = {
        {0, 0, 255},    // Red
        {0, 255, 0},    // Green
        {255, 0, 0},    // Blue
        {0, 255, 255},  // Yellow
        {255, 255, 0},  // Cyan
    };

    scheme->bbox_color_count = 5;
    for (int i = 0; i < 5; ++i) {
        scheme->bbox_colors[i] = {bbox_cols[i][0], bbox_cols[i][1], bbox_cols[i][2]};
    }
}

void c_scheme_print(const C_DrawingScheme* scheme) {
    if (!scheme) return;

    char buf[256];

    LOG_INFO(SCHEME_LOADER_MODULE, "Drawing Scheme:");
    snprintf(buf, sizeof(buf), "  Keypoints: %zu", scheme->keypoint_count);
    LOG_INFO(SCHEME_LOADER_MODULE, buf);
    for (size_t i = 0; i < scheme->keypoint_count; ++i) {
        snprintf(buf, sizeof(buf), "    [%d] %s: BGR(%d, %d, %d)",
                 scheme->keypoints[i].id,
                 scheme->keypoints[i].name,
                 scheme->keypoints[i].color.b,
                 scheme->keypoints[i].color.g,
                 scheme->keypoints[i].color.r);
        LOG_INFO(SCHEME_LOADER_MODULE, buf);
    }

    snprintf(buf, sizeof(buf), "  Skeleton Links: %zu", scheme->skeleton_link_count);
    LOG_INFO(SCHEME_LOADER_MODULE, buf);
    for (size_t i = 0; i < scheme->skeleton_link_count; ++i) {
        snprintf(buf, sizeof(buf), "    %d -> %d: BGR(%d, %d, %d)",
                 scheme->skeleton_links[i].src_kpt_id,
                 scheme->skeleton_links[i].dst_kpt_id,
                 scheme->skeleton_links[i].color.b,
                 scheme->skeleton_links[i].color.g,
                 scheme->skeleton_links[i].color.r);
        LOG_INFO(SCHEME_LOADER_MODULE, buf);
    }

    snprintf(buf, sizeof(buf), "  BBox Colors: %zu", scheme->bbox_color_count);
    LOG_INFO(SCHEME_LOADER_MODULE, buf);
}

} // extern "C"
