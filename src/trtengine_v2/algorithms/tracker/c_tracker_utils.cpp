/**
 * @file c_tracker_utils.cpp
 * @brief Utility functions for tracking: IoU, cosine distance, Hungarian algorithm
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/algorithms/tracker/c_tracker_structures.h"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

using Eigen::MatrixXf;
using Eigen::VectorXf;

extern "C" {

C_SORTConfig c_sort_config_default(void) {
    C_SORTConfig config;
    config.max_age = 5;
    config.min_hits = 1;
    config.iou_threshold = 0.3f;
    config.conf_threshold = 0.5f;
    return config;
}

C_DeepSORTConfig c_deepsort_config_default(void) {
    C_DeepSORTConfig config;
    config.max_age = 30;
    config.min_hits = 3;
    config.iou_threshold = 0.3f;
    config.reid_threshold = 0.4f;
    config.lambda_weight = 0.5f;
    config.conf_threshold = 0.5f;
    config.feature_dim = TRACKER_DEFAULT_FEATURE_DIM;
    return config;
}

void c_association_result_free(C_AssociationResult* result) {
    if (!result) return;
    delete[] result->matches;
    delete[] result->unmatched_detections;
    delete[] result->unmatched_tracks;
    result->matches = nullptr;
    result->unmatched_detections = nullptr;
    result->unmatched_tracks = nullptr;
    result->match_count = 0;
    result->unmatched_det_count = 0;
    result->unmatched_trk_count = 0;
}

void c_detection_array_free(C_DetectionArray* array) {
    if (!array) return;
    if (array->detections) {
        for (size_t i = 0; i < array->count; ++i) {
            delete[] array->detections[i].features;
        }
        delete[] array->detections;
    }
    array->detections = nullptr;
    array->count = 0;
}

void c_track_array_free(C_TrackArray* array) {
    if (!array) return;
    delete[] array->tracks;
    array->tracks = nullptr;
    array->count = 0;
}

} // extern "C"

// ============================================================================
// Internal utility functions (C++ only)
// ============================================================================

namespace tracker_utils {

/**
 * @brief Calculate IoU between two bounding boxes
 */
float iou(const C_BoundingBox& a, const C_BoundingBox& b) {
    float x_left = std::max(a.x1, b.x1);
    float y_top = std::max(a.y1, b.y1);
    float x_right = std::min(a.x2, b.x2);
    float y_bottom = std::min(a.y2, b.y2);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection = (x_right - x_left) * (y_bottom - y_top);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return intersection / union_area;
}

/**
 * @brief Calculate cosine distance between two feature vectors
 * @return Distance in [0, 2], where 0 = identical, 2 = opposite
 */
float cosine_distance(const float* feat1, const float* feat2, int dim) {
    if (!feat1 || !feat2 || dim <= 0) {
        return 1.0f;  // Maximum distance if features not available
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (int i = 0; i < dim; ++i) {
        dot_product += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 < 1e-6f || norm2 < 1e-6f) {
        return 1.0f;
    }

    float similarity = dot_product / (norm1 * norm2);
    return 1.0f - similarity;  // Convert similarity to distance
}

/**
 * @brief Build IoU cost matrix (1 - IoU) for Hungarian algorithm
 *
 * @param det_boxes Detection bounding boxes
 * @param det_count Number of detections
 * @param trk_boxes Track predicted bounding boxes
 * @param trk_count Number of tracks
 * @return Cost matrix (det_count x trk_count)
 */
MatrixXf build_iou_cost_matrix(
    const C_BoundingBox* det_boxes,
    size_t det_count,
    const C_BoundingBox* trk_boxes,
    size_t trk_count
) {
    MatrixXf cost(det_count, trk_count);

    for (size_t d = 0; d < det_count; ++d) {
        for (size_t t = 0; t < trk_count; ++t) {
            float iou_val = iou(det_boxes[d], trk_boxes[t]);
            cost(d, t) = 1.0f - iou_val;  // Convert to cost (lower is better)
        }
    }

    return cost;
}

/**
 * @brief Hungarian algorithm (Kuhn-Munkres) for optimal assignment
 *
 * This is a simplified implementation suitable for tracking.
 * Uses the auction algorithm approach for efficiency.
 *
 * @param cost Cost matrix (rows = detections, cols = tracks)
 * @param threshold Maximum cost for valid assignment
 * @param result Output association result
 */
void hungarian_solve(
    const MatrixXf& cost,
    float threshold,
    C_AssociationResult* result
) {
    const int n_rows = cost.rows();
    const int n_cols = cost.cols();

    if (n_rows == 0 || n_cols == 0) {
        result->matches = nullptr;
        result->match_count = 0;
        result->unmatched_detections = new int[n_rows];
        result->unmatched_det_count = n_rows;
        result->unmatched_tracks = new int[n_cols];
        result->unmatched_trk_count = n_cols;
        for (int i = 0; i < n_rows; ++i) result->unmatched_detections[i] = i;
        for (int i = 0; i < n_cols; ++i) result->unmatched_tracks[i] = i;
        return;
    }

    // Make the matrix square by padding with high cost
    const int n = std::max(n_rows, n_cols);
    MatrixXf padded = MatrixXf::Constant(n, n, threshold + 1.0f);
    padded.topLeftCorner(n_rows, n_cols) = cost;

    // Hungarian algorithm (Kuhn-Munkres)
    // Step 1: Subtract row minimum from each row
    // Step 2: Subtract column minimum from each column
    // Step 3: Cover zeros with minimum lines
    // Step 4: Create additional zeros
    // Repeat until optimal assignment found

    // Row reduction
    for (int i = 0; i < n; ++i) {
        float row_min = padded.row(i).minCoeff();
        padded.row(i).array() -= row_min;
    }

    // Column reduction
    for (int j = 0; j < n; ++j) {
        float col_min = padded.col(j).minCoeff();
        padded.col(j).array() -= col_min;
    }

    // Assignment using greedy approach with refinement
    std::vector<int> row_assignment(n, -1);
    std::vector<int> col_assignment(n, -1);
    std::vector<bool> row_covered(n, false);
    std::vector<bool> col_covered(n, false);

    // Greedy initial assignment
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (std::abs(padded(i, j)) < 1e-6f && row_assignment[i] < 0 && col_assignment[j] < 0) {
                row_assignment[i] = j;
                col_assignment[j] = i;
            }
        }
    }

    // Augment path algorithm for remaining unassigned
    auto augment = [&]() -> bool {
        std::fill(row_covered.begin(), row_covered.end(), false);
        std::fill(col_covered.begin(), col_covered.end(), false);

        // Cover columns with assignments
        for (int i = 0; i < n; ++i) {
            if (row_assignment[i] >= 0) {
                col_covered[row_assignment[i]] = true;
            }
        }

        // Count covered columns
        int covered_count = 0;
        for (int j = 0; j < n; ++j) {
            if (col_covered[j]) covered_count++;
        }

        if (covered_count == n) return true;  // Done

        // Find uncovered zero and try to augment
        float min_uncovered = std::numeric_limits<float>::max();
        for (int i = 0; i < n; ++i) {
            if (row_assignment[i] >= 0) continue;  // Already assigned
            for (int j = 0; j < n; ++j) {
                if (col_assignment[j] >= 0) continue;  // Already assigned
                if (padded(i, j) < min_uncovered) {
                    min_uncovered = padded(i, j);
                }
            }
        }

        if (min_uncovered > 1e-6f && min_uncovered < std::numeric_limits<float>::max()) {
            // Subtract minimum from uncovered rows, add to covered columns
            for (int i = 0; i < n; ++i) {
                if (row_assignment[i] < 0) {
                    padded.row(i).array() -= min_uncovered;
                }
            }
            for (int j = 0; j < n; ++j) {
                if (col_assignment[j] >= 0) {
                    padded.col(j).array() += min_uncovered;
                }
            }
        }

        // Try to find new assignments
        for (int i = 0; i < n; ++i) {
            if (row_assignment[i] >= 0) continue;
            for (int j = 0; j < n; ++j) {
                if (col_assignment[j] >= 0) continue;
                if (std::abs(padded(i, j)) < 1e-6f) {
                    row_assignment[i] = j;
                    col_assignment[j] = i;
                    break;
                }
            }
        }

        return false;
    };

    // Run augmentation until complete
    for (int iter = 0; iter < n * 2; ++iter) {
        if (augment()) break;
    }

    // Extract results
    std::vector<C_Match> matches;
    std::vector<int> unmatched_det;
    std::vector<int> unmatched_trk;

    for (int i = 0; i < n_rows; ++i) {
        int j = row_assignment[i];
        if (j >= 0 && j < n_cols && cost(i, j) <= threshold) {
            matches.push_back({i, j});
        } else {
            unmatched_det.push_back(i);
        }
    }

    for (int j = 0; j < n_cols; ++j) {
        int i = col_assignment[j];
        if (i < 0 || i >= n_rows || cost(i, j) > threshold) {
            unmatched_trk.push_back(j);
        }
    }

    // Copy results
    result->match_count = matches.size();
    result->matches = matches.empty() ? nullptr : new C_Match[matches.size()];
    for (size_t i = 0; i < matches.size(); ++i) {
        result->matches[i] = matches[i];
    }

    result->unmatched_det_count = unmatched_det.size();
    result->unmatched_detections = unmatched_det.empty() ? nullptr : new int[unmatched_det.size()];
    for (size_t i = 0; i < unmatched_det.size(); ++i) {
        result->unmatched_detections[i] = unmatched_det[i];
    }

    result->unmatched_trk_count = unmatched_trk.size();
    result->unmatched_tracks = unmatched_trk.empty() ? nullptr : new int[unmatched_trk.size()];
    for (size_t i = 0; i < unmatched_trk.size(); ++i) {
        result->unmatched_tracks[i] = unmatched_trk[i];
    }
}

/**
 * @brief Linear assignment using IoU cost
 */
void linear_assignment_iou(
    const C_BoundingBox* det_boxes,
    size_t det_count,
    const C_BoundingBox* trk_boxes,
    size_t trk_count,
    float iou_threshold,
    C_AssociationResult* result
) {
    MatrixXf cost = build_iou_cost_matrix(det_boxes, det_count, trk_boxes, trk_count);
    hungarian_solve(cost, 1.0f - iou_threshold, result);
}

} // namespace tracker_utils
