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

# ------------------ 查找多线程库 ------------------
find_package(Threads REQUIRED)

# ------------------ OpenMP 支持 ------------------
find_package(OpenMP QUIET COMPONENTS CXX) # 尝试查找 C++ 的 OpenMP 支持

if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP_CXX")
    # 添加 OpenMP 编译选项和链接库
    # OpenMP_CXX_FLAGS 会包含 -fopenmp 或 /openmp
    # OpenMP_CXX_LIBRARIES 会包含 libgomp 或 libomp 等运行时库
    set(OpenMP_FLAGS ${OpenMP_CXX_FLAGS})
    set(OpenMP_LIBS ${OpenMP_CXX_LIBRARIES})
else()
    message(WARNING "OpenMP CXX support not found. Continuing without OpenMP.")
    set(OpenMP_FLAGS "")
    set(OpenMP_LIBS "")
endif()

# ------------------ 编译选项(可选) ------------------
# 如果所有目标都需要此选项，则全局设置也可接受
add_compile_options(-fvisibility=default)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")  # 可执行文件通常不放在 lib 目录下，这里注释掉

# ------------------ 源文件收集 ------------------
# 使用 CONFIGURE_DEPENDS 使 CMake 在源文件改变时能自动重新生成构建系统
file(GLOB_RECURSE
        COMMON_SOURCE_FILES
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# ====================================================================
#                生成目标：jetson_shared （共享库）
# ====================================================================

# 公共源文件
add_library(jetson_shared SHARED ${COMMON_SOURCE_FILES})

# 设置目标属性
set_target_properties(jetson_shared PROPERTIES
        OUTPUT_NAME "jetson"
)

# 连接头文件
target_include_directories(jetson_shared PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)

# 连接库文件
target_link_libraries(jetson_shared PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
        ${OpenMP_LIBS} # 添加 OpenMP 链接库
        # Threads::Threads # Link Threads for std::thread support
)

# 设置编译选项
target_compile_options(jetson_shared PRIVATE
        ${OpenMP_FLAGS} # 添加 OpenMP 编译选项
)

# ====================================================================
#                生成目标：jetson_static (静态库)
# ====================================================================

# 公共源文件
add_library(jetson_static STATIC ${COMMON_SOURCE_FILES})

# 设置目标属性
set_target_properties(jetson_static PROPERTIES
        OUTPUT_NAME "jetson"
)

# 连接头文件
target_include_directories(jetson_static PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)

# 连接库文件
target_link_libraries(jetson_static PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
        ${OpenMP_LIBS} # 添加 OpenMP 链接库
        # Threads::Threads # Link Threads for std::thread support
)

# 设置编译选项
target_compile_options(jetson_static PRIVATE
        ${OpenMP_FLAGS} # 添加 OpenMP 编译选项
)

# ====================================================================
#                 function_test_pipeline (可执行文件)
# ====================================================================
set(TEST_SPECIAL_SOURCES
        "function_test_pipeline.cpp"
)

list(APPEND TEST_SPECIAL_SOURCES ${COMMON_SOURCE_FILES})

add_executable(function_test_pipeline ${TEST_SPECIAL_SOURCES})

set_target_properties(function_test_pipeline PROPERTIES
        OUTPUT_NAME "function_test_pipeline"
)

target_include_directories(function_test_pipeline PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(function_test_pipeline PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
        ${OpenMP_LIBS} # Add OpenMP link libraries
)

target_compile_options(function_test_pipeline PRIVATE
        ${OpenMP_FLAGS} # Add OpenMP compile options
)

# ====================================================================
#                trt_engine_performance_test (可执行文件)
# ====================================================================
# 测试专用源文件
set(TEST_SPECIAL_SOURCES
        "${PROJECT_SOURCE_DIR}/benchmark_performance.cpp"
)

# 合并公共源和测试专用源
list(APPEND TEST_SPECIAL_SOURCES ${COMMON_SOURCE_FILES})

# 生成可执行文件
add_executable(benchmark_performance ${TEST_SPECIAL_SOURCES})

# 设置目标属性
set_target_properties(benchmark_performance PROPERTIES
        OUTPUT_NAME "benchmark_performance"
)

# 添加包含目录
target_include_directories(benchmark_performance PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${SIMPLE_CUDA_TOOLKITS_INC}
        ${TENSORRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
)

# 链接所需的库
target_link_libraries(benchmark_performance PRIVATE
        ${CUDA_LIBRARIES}
        "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
        ${TENSORRT_LIBS}
        ${OpenCV_LIBRARIES}
        ${OpenMP_LIBS} # 添加 OpenMP 链接库
)

# 设置编译选项
target_compile_options(benchmark_performance PRIVATE
        ${OpenMP_FLAGS} # 添加 OpenMP 编译选项
)

# ------------------- 安装规则 -------------------

# 设置安装前缀。用户可以通过 -DCMAKE_INSTALL_PREFIX=/path/to/install 覆盖
if (NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "/opt/TrtEngineToolkits" CACHE PATH "Installation prefix for SimpleCudaToolkits")
endif()

# 安装头文件 (假设你的公共头文件在 PROJECT_SOURCE_DIR/include 目录下)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        OPTIONAL # OPTIONAL keyword goes before PATTERN in newer CMake versions, but for safety, if you hit an error, try moving it.
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 安装静态库
install(TARGETS jetson_static
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
)

# 安装动态库
install(TARGETS jetson_shared
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
)

# 安装 Python 封装的 API (pyengine 文件夹)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pyengine/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/pyengine"
        OPTIONAL # Use OPTIONAL if this directory might not always exist or is for specific builds
)

# 安装配置文件 (config 文件夹)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/config"
        OPTIONAL # Use OPTIONAL if this directory might not always exist or is for specific builds
)

# 安装脚本文件
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/scripts"
        OPTIONAL # Use OPTIONAL if this directory might not always exist or is for specific builds
)

# 安装资源文件
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/resources/"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/resources"
        OPTIONAL # Use OPTIONAL if this directory might not always exist or is for specific builds
)