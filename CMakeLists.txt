cmake_minimum_required(VERSION 3.16)

project(CombinedProject LANGUAGES CXX)

# ------------------ 基础设置 ------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------ CUDA 与 SimpleCudaToolkits 设置 ------------------
# 如果有可能，建议使用 find_package(CUDA) 来查找 CUDA 环境
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARIES /usr/local/cuda/lib64/libcudart.so)

set(SIMPLE_CUDA_TOOLKITS_PATH /opt/SimpleCudaToolkits)
set(SIMPLE_CUDA_TOOLKITS_INC ${SIMPLE_CUDA_TOOLKITS_PATH}/include)
set(SIMPLE_CUDA_TOOLKITS_LIB ${SIMPLE_CUDA_TOOLKITS_PATH}/lib)

# ------------------ TensorRT 检测 ------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")  # Jetson 平台（ARM64 架构）
    message(STATUS "Configuring for Jetson platform (ARM64)")

    # 使用系统路径查找 TensorRT，统一使用变量名 TENSORRT_INCLUDE_DIRS
    find_path(TENSORRT_INCLUDE_DIRS NvInfer.h
              HINTS /usr/include /usr/local/include
    )

    # 如果找到了 TensorRT 的头文件路径，则输出
    if(TENSORRT_INCLUDE_DIRS)
        message(STATUS "Found TensorRT include directory: ${TENSORRT_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "TensorRT include directory NOT found. Check HINTS in find_path.")
    endif()

    # 查找 TensorRT 的库文件
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

    # 设置 TensorRT 的路径（示例：/opt/tensorrt）
    set(TENSORRT_INCLUDE_DIRS /opt/tensorrt/include)
    set(TENSORRT_LIBRARIES /opt/tensorrt/lib)
    set(TENSORRT_LIBS
        ${TENSORRT_LIBRARIES}/libnvinfer.so
        ${TENSORRT_LIBRARIES}/libnvonnxparser.so
    )
endif()

# ------------------ 查找 OpenCV ------------------
find_package(OpenCV REQUIRED)

# ------------------ 编译选项(可选) ------------------
# 如果所有目标都需要此选项，则全局设置也可接受
add_compile_options(-fvisibility=default)

# ------------------ 源文件收集 ------------------
# 使用 CONFIGURE_DEPENDS 使 CMake 在源文件改变时能自动重新生成构建系统
file(GLOB_RECURSE
        COMMON_SOURCE_FILES
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# ------------------ 生成目标：c_api_test (可执行文件) ------------------
# 1) 测试专用的两个源
set(TEST_SPECIAL_SOURCES
        "${PROJECT_SOURCE_DIR}/c_api_test.cpp"
)

# 2) 合并公共源和测试专用源
list(APPEND TEST_SPECIAL_SOURCES ${COMMON_SOURCE_FILES})

# 3) 生成可执行文件
add_executable(c_api_test ${TEST_SPECIAL_SOURCES})

# 设置目标属性
set_target_properties(c_api_test PROPERTIES
        OUTPUT_NAME "c_api_test"
)

# 添加包含目录
target_include_directories(c_api_test PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)

# 链接所需的库
target_link_libraries(c_api_test PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
)

# ------------------ 生成目标：jetson_shared (共享库) ------------------
add_library(jetson_shared SHARED ${COMMON_SOURCE_FILES})
set_target_properties(jetson_shared PROPERTIES
        OUTPUT_NAME "jetson"
)
target_include_directories(jetson_shared PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(jetson_shared PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
)
# ------------------ 生成目标：jetson_static (静态库) ------------------
add_library(jetson_static STATIC ${COMMON_SOURCE_FILES})
set_target_properties(jetson_static PROPERTIES
        OUTPUT_NAME "jetson"
)
target_include_directories(jetson_static PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(jetson_static PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
)
