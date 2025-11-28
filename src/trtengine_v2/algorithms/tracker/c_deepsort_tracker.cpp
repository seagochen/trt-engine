/**
 * @file c_deepsort_tracker.cpp
 * @brief DeepSORT tracking algorithm implementation
 *
 * DeepSORT extends SORT with Re-ID features and cascaded matching.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/algorithms/tracker/c_deepsort_tracker.h"
#include "trtengine_v2/algorithms/tracker/c_kalman_filter.h"

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstring>
#include <cmath>

using Eigen::MatrixXf;
using Eigen::VectorXf;

// Forward declarations
namespace tracker_utils {
    float iou(const C_BoundingBox& a, const C_BoundingBox& b);
    float cosine_distance(const float* feat1, const float* feat2, int dim);
    void hungarian_solve(const MatrixXf& cost, float threshold, C_AssociationResult* result);
}

/**
 * @brief Feature gallery for a single track
 */
struct FeatureGallery {
    std::deque<std::vector<float>> features;
    static constexpr size_t MAX_GALLERY_SIZE = 100;

    void add_feature(const float* feat, int dim) {
        if (!feat || dim <= 0) return;

        std::vector<float> f(feat, feat + dim);
        features.push_back(std::move(f));

        while (features.size() > MAX_GALLERY_SIZE) {
            features.pop_front();
        }
    }

    /**
     * @brief Compute minimum cosine distance to query feature
     */
    float min_distance(const float* query, int dim) const {
        if (!query || dim <= 0 || features.empty()) {
            return 1.0f;
        }

        float min_dist = std::numeric_limits<float>::max();
        for (const auto& feat : features) {
            if (static_cast<int>(feat.size()) == dim) {
                float dist = tracker_utils::cosine_distance(feat.data(), query, dim);
                min_dist = std::min(min_dist, dist);
            }
        }
        return min_dist;
    }

    /**
     * @brief Compute mean cosine distance to query feature
     */
    float mean_distance(const float* query, int dim) const {
        if (!query || dim <= 0 || features.empty()) {
            return 1.0f;
        }

        float sum_dist = 0.0f;
        int count = 0;
        for (const auto& feat : features) {
            if (static_cast<int>(feat.size()) == dim) {
                sum_dist += tracker_utils::cosine_distance(feat.data(), query, dim);
                count++;
            }
        }
        return count > 0 ? sum_dist / count : 1.0f;
    }
};

/**
 * @brief Internal DeepSORT track representation
 */
struct DeepSORTTrack {
    int track_id;
    C_KalmanFilter* kf;
    int class_id;
    float confidence;
    int hits;
    int age;
    int time_since_update;
    FeatureGallery gallery;
    int feature_dim;

    DeepSORTTrack(int id, const C_BoundingBox* bbox, int cls, float conf,
                  const float* feat, int feat_dim)
        : track_id(id)
        , kf(c_kalman_filter_create(bbox))
        , class_id(cls)
        , confidence(conf)
        , hits(1)
        , age(1)
        , time_since_update(0)
        , feature_dim(feat_dim)
    {
        if (feat && feat_dim > 0) {
            gallery.add_feature(feat, feat_dim);
        }
    }

    ~DeepSORTTrack() {
        c_kalman_filter_destroy(kf);
    }

    DeepSORTTrack(const DeepSORTTrack&) = delete;
    DeepSORTTrack& operator=(const DeepSORTTrack&) = delete;
};

/**
 * @brief DeepSORT tracker internal state
 */
struct C_DeepSORTTracker {
    C_DeepSORTConfig config;
    std::vector<DeepSORTTrack*> tracks;
    int next_id;

    C_DeepSORTTracker(const C_DeepSORTConfig& cfg)
        : config(cfg)
        , next_id(1)
    {}

    ~C_DeepSORTTracker() {
        for (auto* track : tracks) {
            delete track;
        }
    }
};

/**
 * @brief Build combined IoU + Re-ID cost matrix
 */
static MatrixXf build_combined_cost_matrix(
    const C_Detection* detections,
    const std::vector<int>& det_indices,
    const std::vector<DeepSORTTrack*>& tracks,
    const std::vector<int>& trk_indices,
    const std::vector<C_BoundingBox>& predicted_boxes,
    const C_DeepSORTConfig& config
) {
    size_t n_det = det_indices.size();
    size_t n_trk = trk_indices.size();

    MatrixXf cost = MatrixXf::Constant(n_det, n_trk, 1e6f);  // High cost = gated

    for (size_t d = 0; d < n_det; ++d) {
        int det_idx = det_indices[d];
        const C_Detection& det = detections[det_idx];

        for (size_t t = 0; t < n_trk; ++t) {
            int trk_idx = trk_indices[t];
            const DeepSORTTrack* track = tracks[trk_idx];
            const C_BoundingBox& pred_box = predicted_boxes[trk_idx];

            // IoU cost
            float iou_val = tracker_utils::iou(det.bbox, pred_box);
            float iou_cost = 1.0f - iou_val;

            // Gate by IoU
            if (iou_cost > 1.0f - config.iou_threshold) {
                continue;  // Keep as gated (high cost)
            }

            // Re-ID cost (if features available)
            float reid_cost = 1.0f;
            if (det.features && det.feature_dim > 0 && !track->gallery.features.empty()) {
                reid_cost = track->gallery.min_distance(det.features, det.feature_dim);

                // Gate by Re-ID threshold
                if (reid_cost > config.reid_threshold) {
                    continue;  // Keep as gated
                }
            }

            // Combined cost
            cost(d, t) = config.lambda_weight * iou_cost +
                        (1.0f - config.lambda_weight) * reid_cost;
        }
    }

    return cost;
}

extern "C" {

C_DeepSORTTracker* c_deepsort_tracker_create(const C_DeepSORTConfig* config) {
    C_DeepSORTConfig cfg = config ? *config : c_deepsort_config_default();
    return new (std::nothrow) C_DeepSORTTracker(cfg);
}

void c_deepsort_tracker_destroy(C_DeepSORTTracker* tracker) {
    delete tracker;
}

bool c_deepsort_tracker_update(
    C_DeepSORTTracker* tracker,
    const C_Detection* detections,
    size_t det_count,
    C_TrackArray* result
) {
    if (!tracker || !result) return false;

    result->tracks = nullptr;
    result->count = 0;

    // Step 1: Predict new locations for existing tracks
    std::vector<C_BoundingBox> predicted_boxes;
    predicted_boxes.reserve(tracker->tracks.size());

    for (auto* track : tracker->tracks) {
        C_BoundingBox pred = c_kalman_filter_predict(track->kf);
        predicted_boxes.push_back(pred);
        track->age++;
        track->time_since_update++;
    }

    // Track indices for matching
    std::vector<bool> det_matched(det_count, false);
    std::vector<bool> trk_matched(tracker->tracks.size(), false);

    // Step 2: Cascade matching - prioritize recently seen tracks
    // Stage 1: Match confirmed tracks by time_since_update (most recent first)
    for (int cascade_level = 1; cascade_level <= tracker->config.max_age; ++cascade_level) {
        // Collect tracks at this cascade level
        std::vector<int> cascade_trk_indices;
        for (size_t t = 0; t < tracker->tracks.size(); ++t) {
            if (!trk_matched[t] && tracker->tracks[t]->time_since_update == cascade_level) {
                cascade_trk_indices.push_back(static_cast<int>(t));
            }
        }

        if (cascade_trk_indices.empty()) continue;

        // Collect unmatched detections
        std::vector<int> unmatched_det_indices;
        for (size_t d = 0; d < det_count; ++d) {
            if (!det_matched[d]) {
                unmatched_det_indices.push_back(static_cast<int>(d));
            }
        }

        if (unmatched_det_indices.empty()) break;

        // Build cost matrix
        MatrixXf cost = build_combined_cost_matrix(
            detections, unmatched_det_indices,
            tracker->tracks, cascade_trk_indices,
            predicted_boxes, tracker->config
        );

        // Solve assignment
        C_AssociationResult assoc = {nullptr, 0, nullptr, 0, nullptr, 0};
        float cost_threshold = tracker->config.lambda_weight * (1.0f - tracker->config.iou_threshold) +
                              (1.0f - tracker->config.lambda_weight) * tracker->config.reid_threshold;
        tracker_utils::hungarian_solve(cost, cost_threshold, &assoc);

        // Process matches
        for (size_t i = 0; i < assoc.match_count; ++i) {
            int local_det = assoc.matches[i].detection_idx;
            int local_trk = assoc.matches[i].track_idx;
            int det_idx = unmatched_det_indices[local_det];
            int trk_idx = cascade_trk_indices[local_trk];

            DeepSORTTrack* track = tracker->tracks[trk_idx];
            const C_Detection& det = detections[det_idx];

            c_kalman_filter_update(track->kf, &det.bbox);
            track->confidence = det.confidence;
            track->class_id = det.class_id;
            track->hits++;
            track->time_since_update = 0;

            // Update feature gallery
            if (det.features && det.feature_dim > 0) {
                track->gallery.add_feature(det.features, det.feature_dim);
            }

            det_matched[det_idx] = true;
            trk_matched[trk_idx] = true;
        }

        c_association_result_free(&assoc);
    }

    // Step 3: IoU-only matching for remaining unconfirmed tracks
    {
        std::vector<int> unconfirmed_trk_indices;
        for (size_t t = 0; t < tracker->tracks.size(); ++t) {
            if (!trk_matched[t] && tracker->tracks[t]->hits < tracker->config.min_hits) {
                unconfirmed_trk_indices.push_back(static_cast<int>(t));
            }
        }

        std::vector<int> remaining_det_indices;
        for (size_t d = 0; d < det_count; ++d) {
            if (!det_matched[d]) {
                remaining_det_indices.push_back(static_cast<int>(d));
            }
        }

        if (!unconfirmed_trk_indices.empty() && !remaining_det_indices.empty()) {
            // Build IoU-only cost matrix
            size_t n_det = remaining_det_indices.size();
            size_t n_trk = unconfirmed_trk_indices.size();
            MatrixXf iou_cost(n_det, n_trk);

            for (size_t d = 0; d < n_det; ++d) {
                int det_idx = remaining_det_indices[d];
                for (size_t t = 0; t < n_trk; ++t) {
                    int trk_idx = unconfirmed_trk_indices[t];
                    float iou_val = tracker_utils::iou(
                        detections[det_idx].bbox,
                        predicted_boxes[trk_idx]
                    );
                    iou_cost(d, t) = 1.0f - iou_val;
                }
            }

            C_AssociationResult assoc = {nullptr, 0, nullptr, 0, nullptr, 0};
            tracker_utils::hungarian_solve(iou_cost, 1.0f - tracker->config.iou_threshold, &assoc);

            for (size_t i = 0; i < assoc.match_count; ++i) {
                int local_det = assoc.matches[i].detection_idx;
                int local_trk = assoc.matches[i].track_idx;
                int det_idx = remaining_det_indices[local_det];
                int trk_idx = unconfirmed_trk_indices[local_trk];

                DeepSORTTrack* track = tracker->tracks[trk_idx];
                const C_Detection& det = detections[det_idx];

                c_kalman_filter_update(track->kf, &det.bbox);
                track->confidence = det.confidence;
                track->class_id = det.class_id;
                track->hits++;
                track->time_since_update = 0;

                if (det.features && det.feature_dim > 0) {
                    track->gallery.add_feature(det.features, det.feature_dim);
                }

                det_matched[det_idx] = true;
                trk_matched[trk_idx] = true;
            }

            c_association_result_free(&assoc);
        }
    }

    // Step 4: Create new tracks for unmatched high-confidence detections
    for (size_t d = 0; d < det_count; ++d) {
        if (!det_matched[d] && detections[d].confidence >= tracker->config.conf_threshold) {
            const C_Detection& det = detections[d];
            DeepSORTTrack* new_track = new DeepSORTTrack(
                tracker->next_id++,
                &det.bbox,
                det.class_id,
                det.confidence,
                det.features,
                det.feature_dim
            );
            tracker->tracks.push_back(new_track);
        }
    }

    // Step 5: Remove dead tracks
    auto remove_it = std::remove_if(
        tracker->tracks.begin(),
        tracker->tracks.end(),
        [&tracker](DeepSORTTrack* track) {
            if (track->time_since_update > tracker->config.max_age) {
                delete track;
                return true;
            }
            return false;
        }
    );
    tracker->tracks.erase(remove_it, tracker->tracks.end());

    // Step 6: Return confirmed tracks
    std::vector<C_Track> confirmed;
    for (const auto* track : tracker->tracks) {
        bool is_confirmed = (track->hits >= tracker->config.min_hits);
        bool is_active = (track->time_since_update == 0);

        if (is_confirmed && is_active) {
            C_Track out;
            out.track_id = track->track_id;
            out.bbox = c_kalman_filter_get_bbox(track->kf);
            out.class_id = track->class_id;
            out.confidence = track->confidence;
            out.hits = track->hits;
            out.age = track->age;
            out.time_since_update = track->time_since_update;
            out.is_confirmed = true;
            confirmed.push_back(out);
        }
    }

    result->count = confirmed.size();
    if (result->count > 0) {
        result->tracks = new C_Track[result->count];
        std::memcpy(result->tracks, confirmed.data(), result->count * sizeof(C_Track));
    }

    return true;
}

bool c_deepsort_tracker_update_no_features(
    C_DeepSORTTracker* tracker,
    const C_Detection* detections,
    size_t det_count,
    C_TrackArray* result
) {
    // Simply call the main update - it handles missing features gracefully
    return c_deepsort_tracker_update(tracker, detections, det_count, result);
}

bool c_deepsort_tracker_get_tracks(
    const C_DeepSORTTracker* tracker,
    C_TrackArray* result
) {
    if (!tracker || !result) return false;

    result->count = tracker->tracks.size();
    if (result->count == 0) {
        result->tracks = nullptr;
        return true;
    }

    result->tracks = new C_Track[result->count];
    for (size_t i = 0; i < result->count; ++i) {
        const DeepSORTTrack* track = tracker->tracks[i];
        result->tracks[i].track_id = track->track_id;
        result->tracks[i].bbox = c_kalman_filter_get_bbox(track->kf);
        result->tracks[i].class_id = track->class_id;
        result->tracks[i].confidence = track->confidence;
        result->tracks[i].hits = track->hits;
        result->tracks[i].age = track->age;
        result->tracks[i].time_since_update = track->time_since_update;
        result->tracks[i].is_confirmed = (track->hits >= tracker->config.min_hits);
    }

    return true;
}

bool c_deepsort_tracker_get_confirmed_tracks(
    const C_DeepSORTTracker* tracker,
    C_TrackArray* result
) {
    if (!tracker || !result) return false;

    std::vector<C_Track> confirmed;
    for (const auto* track : tracker->tracks) {
        if (track->hits >= tracker->config.min_hits) {
            C_Track out;
            out.track_id = track->track_id;
            out.bbox = c_kalman_filter_get_bbox(track->kf);
            out.class_id = track->class_id;
            out.confidence = track->confidence;
            out.hits = track->hits;
            out.age = track->age;
            out.time_since_update = track->time_since_update;
            out.is_confirmed = true;
            confirmed.push_back(out);
        }
    }

    result->count = confirmed.size();
    if (result->count > 0) {
        result->tracks = new C_Track[result->count];
        std::memcpy(result->tracks, confirmed.data(), result->count * sizeof(C_Track));
    } else {
        result->tracks = nullptr;
    }

    return true;
}

void c_deepsort_tracker_reset(C_DeepSORTTracker* tracker) {
    if (!tracker) return;

    for (auto* track : tracker->tracks) {
        delete track;
    }
    tracker->tracks.clear();
    tracker->next_id = 1;
}

C_DeepSORTConfig c_deepsort_tracker_get_config(const C_DeepSORTTracker* tracker) {
    if (!tracker) return c_deepsort_config_default();
    return tracker->config;
}

int c_deepsort_tracker_get_next_id(const C_DeepSORTTracker* tracker) {
    if (!tracker) return 0;
    return tracker->next_id;
}

} // extern "C"
