#include <vector>
#include <omp.h> // For OpenMP parallelization
#include "serverlet/models/common/yolo_coords.h"
#include "serverlet/models/common/yolo_dstruct.h"

void cvtXYWHCoordsToYolo(const std::vector<float>& input, std::vector<Yolo>& output,
                            int features, int samples, float target_width, float target_height)
{
    output.clear();
    output.resize(samples); // 预分配内存，确保线程安全

    #pragma omp parallel for
    for (int i = 0; i < samples; ++i) {
        // 计算边界框坐标
        const float cx = input[i * features + 0] * target_width;
        const float cy = input[i * features + 1] * target_height;
        const float w = input[i * features + 2] * target_width;
        const float h = input[i * features + 3] * target_height;

        Yolo& yolo = output[i]; // 直接引用 vector 中的元素
        yolo.conf = input[i * features + 4];
        yolo.cls = static_cast<int>(input[i * features + 5]);
        yolo.lx = static_cast<int>(cx - w / 2.0f);
        yolo.ly = static_cast<int>(cy - h / 2.0f);
        yolo.rx = static_cast<int>(cx + w / 2.0f);
        yolo.ry = static_cast<int>(cy + h / 2.0f);
    }
}


void cvtXYWHCoordsToYoloPose(const std::vector<float>& input, std::vector<YoloPose>& output,
                            int features, int samples, float target_width, float target_height)
{
    output.clear();
    output.resize(samples); // 预分配内存，确保线程安全

    #pragma omp parallel for
    for (int i = 0; i < samples; ++i) {
        // 计算边界框坐标
        const float cx = input[i * features + 0] * target_width;
        const float cy = input[i * features + 1] * target_height;
        const float w = input[i * features + 2] * target_width;
        const float h = input[i * features + 3] * target_height;

        YoloPose& yolo_pose = output[i]; // 直接引用 vector 中的元素
        yolo_pose.conf = input[i * features + 4];
        yolo_pose.cls = 0; // Pose任务默认类别为0
        yolo_pose.lx = static_cast<int>(cx - w / 2.0f);
        yolo_pose.ly = static_cast<int>(cy - h / 2.0f);
        yolo_pose.rx = static_cast<int>(cx + w / 2.0f);
        yolo_pose.ry = static_cast<int>(cy + h / 2.0f);

        // 处理17个关键点
        yolo_pose.pts.resize(17);
        for (int j = 0; j < 17; ++j) {
            yolo_pose.pts[j].x = static_cast<int>(input[i * features + 5 + j * 3] * target_width);
            yolo_pose.pts[j].y = static_cast<int>(input[i * features + 5 + j * 3 + 1] * target_height);
            yolo_pose.pts[j].conf = input[i * features + 5 + j * 3 + 2];
        }
    }
}
