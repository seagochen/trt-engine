//
// Created by xtcjj on 2025/08/14.
//

#ifndef COMBINEDPROJECT_C_STRUCTS_H
#define COMBINEDPROJECT_C_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif

    // C-compatible structure for a single keypoint in a pose.
    typedef struct {
        float x;
        float y;
        float conf; // Confidence score for the keypoint
    } C_KeyPoint;

    // C-compatible structure for a single YoloPose detection.
    typedef struct {
        int lx, ly, rx, ry; // Bounding box coordinates
        int cls;            // Class index (e.g., person)
        int num_pts;        // Number of keypoints in 'pts' array
        float conf;         // Overall confidence score for the pose detection
        C_KeyPoint* pts;    // Pointer to an array of C_KeyPoint structures.
        float* feats;       // Pointer to an array of additional features (e.g., pose embeddings), size is 256
    } C_YoloPose;

    // C-compatible structure for the results of pose detections for one image.
    typedef struct {
        int image_idx;          // Original index of the image within the input batch
        int num_detections;     // Number of C_YoloPose detections for this image
        C_YoloPose* detections; // Pointer to an array of C_YoloPose structures.
        // This memory must be freed by the API consumer.
    } C_ImagePoseResults;

    // C-compatible structure for the aggregated results of a batch of images.
    typedef struct {
        int num_images;             // Number of images in this result batch
        C_ImagePoseResults* results; // Pointer to an array of C_ImagePoseResults structures.
        // This memory must be freed by the API consumer.
    } C_BatchedPoseResults;


#ifdef __cplusplus
};
#endif

#endif //COMBINEDPROJECT_C_STRUCTS_H