/**
 * @file c_sort_tracker.cpp
 * @brief SORT (Simple Online and Realtime Tracking) implementation
 *
 * @author TrtEngineToolkits
 * @date 2025-11-25
 */

#include "trtengine_v2/algorithms/tracker/c_sort_tracker.h"
#include "trtengine_v2/algorithms/tracker/c_kalman_filter.h"

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cstring>

// Forward declaration of internal utility functions
namespace tracker_utils {
    float iou(const C_BoundingBox& a, const C_BoundingBox& b);
    void linear_assignment_iou(
        const C_BoundingBox* det_boxes,
        size_t det_count,
        const C_BoundingBox* trk_boxes,
        size_t trk_count,
        float iou_threshold,
        C_AssociationResult* result
    );
}

/**
 * @brief Internal track representation
 */
struct SORTTrack {
    int track_id;
    C_KalmanFilter* kf;
    int class_id;
    float confidence;
    int hits;
    int age;
    int time_since_update;

    SORTTrack(int id, const C_BoundingBox* bbox, int cls, float conf)
        : track_id(id)
        , kf(c_kalman_filter_create(bbox))
        , class_id(cls)
        , confidence(conf)
        , hits(1)
        , age(1)
        , time_since_update(0)
    {}

    ~SORTTrack() {
        c_kalman_filter_destroy(kf);
    }

    // Disable copy
    SORTTrack(const SORTTrack&) = delete;
    SORTTrack& operator=(const SORTTrack&) = delete;

    // Enable move
    SORTTrack(SORTTrack&& other) noexcept
        : track_id(other.track_id)
        , kf(other.kf)
        , class_id(other.class_id)
        , confidence(other.confidence)
        , hits(other.hits)
        , age(other.age)
        , time_since_update(other.time_since_update)
    {
        other.kf = nullptr;
    }
};

/**
 * @brief SORT tracker internal state
 */
struct C_SORTTracker {
    C_SORTConfig config;
    std::vector<SORTTrack*> tracks;
    int next_id;

    C_SORTTracker(const C_SORTConfig& cfg)
        : config(cfg)
        , next_id(1)
    {}

    ~C_SORTTracker() {
        for (auto* track : tracks) {
            delete track;
        }
    }
};

extern "C" {

C_SORTTracker* c_sort_tracker_create(const C_SORTConfig* config) {
    C_SORTConfig cfg = config ? *config : c_sort_config_default();
    return new (std::nothrow) C_SORTTracker(cfg);
}

void c_sort_tracker_destroy(C_SORTTracker* tracker) {
    delete tracker;
}

bool c_sort_tracker_update(
    C_SORTTracker* tracker,
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

    // Step 2: Associate detections with tracks using IoU
    std::vector<C_BoundingBox> det_boxes;
    det_boxes.reserve(det_count);
    for (size_t i = 0; i < det_count; ++i) {
        det_boxes.push_back(detections[i].bbox);
    }

    C_AssociationResult assoc = {nullptr, 0, nullptr, 0, nullptr, 0};

    if (!tracker->tracks.empty() && det_count > 0) {
        tracker_utils::linear_assignment_iou(
            det_boxes.data(), det_count,
            predicted_boxes.data(), predicted_boxes.size(),
            tracker->config.iou_threshold,
            &assoc
        );
    } else {
        // No tracks or no detections - everything is unmatched
        assoc.unmatched_det_count = det_count;
        assoc.unmatched_detections = det_count > 0 ? new int[det_count] : nullptr;
        for (size_t i = 0; i < det_count; ++i) {
            assoc.unmatched_detections[i] = static_cast<int>(i);
        }
        assoc.unmatched_trk_count = tracker->tracks.size();
        assoc.unmatched_tracks = tracker->tracks.empty() ? nullptr : new int[tracker->tracks.size()];
        for (size_t i = 0; i < tracker->tracks.size(); ++i) {
            assoc.unmatched_tracks[i] = static_cast<int>(i);
        }
    }

    // Step 3: Update matched tracks
    for (size_t i = 0; i < assoc.match_count; ++i) {
        int det_idx = assoc.matches[i].detection_idx;
        int trk_idx = assoc.matches[i].track_idx;

        SORTTrack* track = tracker->tracks[trk_idx];
        c_kalman_filter_update(track->kf, &detections[det_idx].bbox);
        track->confidence = detections[det_idx].confidence;
        track->class_id = detections[det_idx].class_id;
        track->hits++;
        track->time_since_update = 0;
    }

    // Step 4: Create new tracks for unmatched detections with high confidence
    for (size_t i = 0; i < assoc.unmatched_det_count; ++i) {
        int det_idx = assoc.unmatched_detections[i];
        const C_Detection& det = detections[det_idx];

        if (det.confidence >= tracker->config.conf_threshold) {
            SORTTrack* new_track = new SORTTrack(
                tracker->next_id++,
                &det.bbox,
                det.class_id,
                det.confidence
            );
            tracker->tracks.push_back(new_track);
        }
    }

    // Step 5: Remove dead tracks
    auto remove_it = std::remove_if(
        tracker->tracks.begin(),
        tracker->tracks.end(),
        [&tracker](SORTTrack* track) {
            if (track->time_since_update > tracker->config.max_age) {
                delete track;
                return true;
            }
            return false;
        }
    );
    tracker->tracks.erase(remove_it, tracker->tracks.end());

    // Free association result
    c_association_result_free(&assoc);

    // Step 6: Return confirmed tracks
    std::vector<C_Track> confirmed;
    for (const auto* track : tracker->tracks) {
        // Track is confirmed if hits >= min_hits AND recently updated
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

    // Copy to output
    result->count = confirmed.size();
    if (result->count > 0) {
        result->tracks = new C_Track[result->count];
        std::memcpy(result->tracks, confirmed.data(), result->count * sizeof(C_Track));
    }

    return true;
}

bool c_sort_tracker_get_tracks(
    const C_SORTTracker* tracker,
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
        const SORTTrack* track = tracker->tracks[i];
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

bool c_sort_tracker_get_confirmed_tracks(
    const C_SORTTracker* tracker,
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

void c_sort_tracker_reset(C_SORTTracker* tracker) {
    if (!tracker) return;

    for (auto* track : tracker->tracks) {
        delete track;
    }
    tracker->tracks.clear();
    tracker->next_id = 1;
}

C_SORTConfig c_sort_tracker_get_config(const C_SORTTracker* tracker) {
    if (!tracker) return c_sort_config_default();
    return tracker->config;
}

int c_sort_tracker_get_next_id(const C_SORTTracker* tracker) {
    if (!tracker) return 0;
    return tracker->next_id;
}

} // extern "C"
