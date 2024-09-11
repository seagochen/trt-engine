//
// Created by ubuntu on 9/12/24.
//

#ifndef JETSON_INFER_TIMER_H
#define JETSON_INFER_TIMER_H

#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <tuple>

class Timer {
public:
    template<typename Function, typename... Args>
    Timer(int interval, Function&& f, Args&&... args)
            : interval_(interval), active_(true) {
        auto task = std::bind(std::forward<Function>(f), std::forward<Args>(args)...);
        thread_ = std::thread([this, task]() {
            while (active_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
                if (active_) {
                    task();
                }
            }
        });
    }

    ~Timer() {
        stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void stop() {
        active_ = false;
    }

private:
    int interval_;
    std::atomic<bool> active_;
    std::thread thread_;
};

#endif //JETSON_INFER_TIMER_H
