//
// Created by ubuntu on 9/26/24.
//

#ifndef LOCK_FREE_QUEUE_HPP
#define LOCK_FREE_QUEUE_HPP

#include <atomic>
#include <iostream>
#include <thread>

template<typename T>
class LockFreeQueue {
public:
    LockFreeQueue(size_t capacity)
        : capacity(capacity + 1), head(0), tail(0), buffer(new T[capacity + 1]) {}

    ~LockFreeQueue() {
        delete[] buffer;
    }

    bool enqueue(const T& item) {
        size_t tailPos = tail.load(std::memory_order_relaxed);
        size_t nextTailPos = increment(tailPos);

        if (nextTailPos == head.load(std::memory_order_acquire)) {
            // Queue is full
            return false;
        }

        buffer[tailPos] = item;
        tail.store(nextTailPos, std::memory_order_release);
        return true;
    }

    bool dequeue(T& item) {
        size_t headPos = head.load(std::memory_order_relaxed);

        if (headPos == tail.load(std::memory_order_acquire)) {
            // Queue is empty
            return false;
        }

        item = buffer[headPos];
        head.store(increment(headPos), std::memory_order_release);
        return true;
    }

    // Return the number of elements in the queue
    size_t size() const {
        size_t headPos = head.load(std::memory_order_relaxed);
        size_t tailPos = tail.load(std::memory_order_relaxed);

        if (tailPos >= headPos) {
            return tailPos - headPos;
        } else {
            return capacity - headPos + tailPos;
        }
    }

private:
    size_t increment(size_t pos) const {
        return (pos + 1) % capacity;
    }

    const size_t capacity;
    std::atomic<size_t> head, tail;
    T* buffer;
};

#endif // LOCK_FREE_QUEUE_HPP