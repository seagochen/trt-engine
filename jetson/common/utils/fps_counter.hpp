//
// Created by ubuntu on 9/9/24.
//

#ifndef INFER_FPSCOUNTER_HPP
#define INFER_FPSCOUNTER_HPP

#include <chrono>

// FPS Calculator class
class FPSCounter {
public:
    FPSCounter() : frame_count(0), fps(0.0), start_time(std::chrono::high_resolution_clock::now()) {}

    void countFrames() {
        frame_count++;
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        if (elapsed_time.count() >= 1.0) {
            fps = frame_count / elapsed_time.count();
            frame_count = 0;
            start_time = end_time;
        }
    }

    [[nodiscard]] double getFPS() const { return fps; }

private:
    int frame_count;
    double fps;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

#endif //INFER_FPSCOUNTER_HPP
