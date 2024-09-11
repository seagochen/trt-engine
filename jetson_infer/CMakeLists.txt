cmake_minimum_required(VERSION 3.16)

project(jetson_infer LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 统一设置 CUDA 路径
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64/libcudart.so)

# 设置SimpleCudaToolkits的路径
set(SIMPLE_CUDA_TOOLKITS_PATH /opt/SimpleCudaToolkits)
set(SIMPLE_CUDA_TOOLKITS_INC ${SIMPLE_CUDA_TOOLKITS_PATH}/include)
set(SIMPLE_CUDA_TOOLKITS_LIB ${SIMPLE_CUDA_TOOLKITS_PATH}/lib)

# 检查系统架构以区分 TensorRT 的设置
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")  # Jetson 平台（ARM64 架构）
    message(STATUS "Configuring for Jetson platform (ARM64)")

    # 使用系统路径查找 TensorRT
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS /usr/include /usr/local/include
    )

    find_library(NVINFER_LIB nvinfer
            HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib
    )

    find_library(NVPARSERS_LIB nvparsers
            HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib
    )

    find_library(NVONNXPARSER_LIB nvonnxparser
            HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib
    )

    set(TENSORRT_LIBS
            ${NVINFER_LIB}
            ${NVPARSERS_LIB}
            ${NVONNXPARSER_LIB}
    )

else()  # x86 平台
    message(STATUS "Configuring for x86 platform")

    # 设置TensorRT的路径, 使用相对路径
    set(TENSORRT_INCLUDE_DIRS /opt/tensorrt/include)
    set(TENSORRT_LIBRARIES /opt/tensorrt/lib)
    set(TENSORRT_LIBS
            ${TENSORRT_LIBRARIES}/libnvinfer.so
            ${TENSORRT_LIBRARIES}/libnvonnxparser.so
    )
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 MQTT (Mosquitto)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)

# 查找 YAML
find_package(yaml-cpp REQUIRED)

# 查找 Protobuf
find_package(Protobuf REQUIRED)

# 查找所有 common 目录下的子目录源文件
file(GLOB_RECURSE SOURCE_FILES "common/*.cpp" "yolo/*.cpp" "common/protobufs/*.pb.cc")

# 添加可执行文件
add_executable(jetson_infer local.cpp ${SOURCE_FILES})

# 包含目录设置
target_include_directories(jetson_infer PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${MOSQUITTO_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        ${Protobuf_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(jetson_infer PRIVATE
        ${OpenCV_LIBRARIES}
        ${TENSORRT_LIBS}
        ${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a
        ${CUDA_LIBRARIES}
        ${MOSQUITTO_LIBRARIES}
        yaml-cpp
        ${Protobuf_LIBRARIES}
)
