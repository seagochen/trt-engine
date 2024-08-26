#ifndef JETSON_INFER_TENSOR_HPP
#define JETSON_INFER_TENSOR_HPP

#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <numeric>
#include <stdexcept>
#include <algorithm>

////////////////////////////////////// Tensor Data Type //////////////////////////////////////

enum tensor_type {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8,
    // Add other data types as needed
};

////////////////////////////////////// Tensor Dim //////////////////////////////////////

struct TensorDimensions {
    std::vector<int> dims;
    size_t mem_size;  // 数据所占内存大小
    tensor_type type;

    /**
     * @brief 默认构造函数
     */
    TensorDimensions() : mem_size(0), type(FLOAT32) {}

    /**
     * @brief 构造函数
     * @param dims
     * @param type
     */
    TensorDimensions(std::vector<int> dims, tensor_type type)
            : dims(std::move(dims)), type(type) {
        mem_size = calculateMemSize();
    }

    /**
     * @brief 计算数据所占内存大小
     * @return
     */
    [[nodiscard]] size_t calculateMemSize() const {
        static const std::unordered_map<tensor_type, size_t> type_size_map = {
                {FLOAT32, sizeof(float)},
                {FLOAT64, sizeof(double)},
                {INT32, sizeof(int32_t)},
                {INT64, sizeof(int64_t)},
                {UINT8, sizeof(uint8_t)}
        };

        auto it = type_size_map.find(type);
        if (it == type_size_map.end()) {
            throw std::runtime_error("Unsupported data type.");
        }

        size_t elementSize = it->second;
        size_t totalElements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
        return totalElements * elementSize;
    }
};

////////////////////////////////////// Tensor Base //////////////////////////////////////

template <typename T>
class TensorBase {
protected:
    TensorDimensions dims;

public:
    /**
     * @brief 默认构造函数
     */
    TensorBase() = default;

    /**
     * @brief 通过TensorDimensions构造
     * @param dims
     */
    explicit TensorBase(TensorDimensions dims) : dims(std::move(dims)) {}

    /**
     * @brief 虚析构函数
     */
    virtual ~TensorBase() = default;

    /**
     * @brief 获取数据维度信息
     * @return
     */
    [[nodiscard]] const TensorDimensions& getDims() const { return dims; }

    /**
     * @brief 获取数据类型
     * @return
     */
    [[nodiscard]] size_t getMemSize() const { return dims.mem_size; }

    /**
     * @brief 获取数据元素个数
     * @return
     */
    [[nodiscard]] size_t getElementCount() const { return dims.mem_size / sizeof(T); }

    /**
     * 纯虚函数，用于暴露数据指针，子类需要实现
     * @return
     */
    virtual void* ptr() = 0;
};

////////////////////////////////////// Forward Declaration //////////////////////////////////////

template <typename T>
class CpuTensor;

////////////////////////////////////// Tensor CUDA //////////////////////////////////////

template <typename T>
class CudaTensor : public TensorBase<T> {
private:
    std::unique_ptr<T, decltype(&cudaFree)> data;

public:

    /**
     * @brief 默认构造函数
     */
    CudaTensor() : TensorBase<T>(), data(nullptr, cudaFree) {}

    /**
     * @brief 通过TensorDimensions构造
     * @param dims
     */
    explicit CudaTensor(const TensorDimensions& dims) : TensorBase<T>(dims), data(nullptr, cudaFree) {
        T* buffer;
        cudaError_t err = cudaMalloc(&buffer, dims.mem_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        data.reset(buffer);
    }

    /**
     * @brief 实现copyFrom方法，从std::vector中拷贝数据
     * @param inputData
     */
    void copyFrom(const std::vector<T>& inputData) {
        if (inputData.size() * sizeof(T) != this->dims.mem_size) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(this->data.get(), inputData.data(), this->dims.mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 从另一个CudaTensor中拷贝数据
     * @param other
     */
    void copyFrom(const CudaTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(this->data.get(), other.ptr(), this->dims.mem_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 从CpuTensor中拷贝数据
     * @param cpuTensor
     */
    void copyFrom(const CpuTensor<T>& cpuTensor) {
        if (this->dims.mem_size != cpuTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(this->data.get(), cpuTensor.ptr(), this->dims.mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 将数据拷贝到std::vector中
     * @param outputData
     */
    void copyTo(std::vector<T>& outputData) {
        if (outputData.size() * sizeof(T) != this->dims.mem_size) {
            outputData.resize(this->dims.mem_size / sizeof(T));
        }
        cudaError_t err = cudaMemcpy(outputData.data(), data.get(), this->dims.mem_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 将数据拷贝到另一个CudaTensor中
     * @param other
     */
    void copyTo(CudaTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(other.ptr(), data.get(), this->dims.mem_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 将数据拷贝到CpuTensor中
     * @param cpuTensor
     */
    void copyTo(CpuTensor<T>& cpuTensor) {
        if (this->dims.mem_size != cpuTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(cpuTensor.ptr(), data.get(), this->dims.mem_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 暴露数据指针，方便其他模块直接调用
     * @return
     */
    void* ptr() override {
        return data.get();
    }

    /**
     * @brief 移动赋值运算符
     * @param other
     * @return
     */
    CudaTensor& operator=(CudaTensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            this->dims = std::move(other.dims);
        }
        return *this;
    }

    /**
     * @brief 移动构造函数
     * @param other
     */
    CudaTensor(CudaTensor&& other) noexcept = default;
};

////////////////////////////////////// Tensor CPU //////////////////////////////////////

template <typename T>
class CpuTensor : public TensorBase<T> {
private:
    std::vector<T> data;

public:

    /**
     * @brief 默认构造函数
     */
    CpuTensor() = default;

    /**
     * @brief 通过TensorDimensions构造
     * @param dims
     */
    explicit CpuTensor(const TensorDimensions& dims)
            : TensorBase<T>(dims), data(dims.mem_size / sizeof(T)) {}

    /**
     * @brief 从std::vector中拷贝数据
     * @param inputData
     */
    void copyFrom(const std::vector<T>& inputData) {
        if (inputData.size() * sizeof(T) != this->dims.mem_size) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        data = inputData;
    }

    /**
     * @brief 从另一个CudaTensor中拷贝数据
     * @param cudaTensor
     */
    void copyFrom(const CudaTensor<T>& cudaTensor) {
        if (this->dims.mem_size != cudaTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(data.data(), cudaTensor.ptr(), this->dims.mem_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 从另一个CpuTensor中拷贝数据
     * @param other
     */
    void copyFrom(const CpuTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        data = other.data;
    }

    /**
     * @brief 将数据拷贝到std::vector中
     * @param outputData
     */
    void copyTo(std::vector<T>& outputData) {
        outputData = data;
    }

    /**
     * @brief 将数据拷贝到另一个CudaTensor中
     * @param cudaTensor
     */
    void copyTo(CudaTensor<T>& cudaTensor) {
        if (this->dims.mem_size != cudaTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(cudaTensor.ptr(), data.data(), this->dims.mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    /**
     * @brief 将数据拷贝到CpuTensor中
     * @param other
     */
    void copyTo(CpuTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        other.copyFrom(data);
    }

    /**
     * @brief 暴露数据指针，方便其他模块直接调用
     * @return
     */
    void* ptr() override {
        return data.data();
    }

    /**
     * @brief 移动赋值运算符
     * @param other
     * @return
     */
    CpuTensor& operator=(CpuTensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            this->dims = std::move(other.dims);
        }
        return *this;
    }

    /**
     * @brief 移动构造函数
     * @param other
     */
    CpuTensor(CpuTensor&& other) noexcept = default;
};

#endif //JETSON_INFER_TENSOR_HPP
