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
file(GLOB_RECURSE COMMON_SOURCE_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/common/*.cpp"
)

# 1) jetson_test 的源文件（额外包含 jetson_test.cpp 与 jetson_apis.cpp）
set(JETSON_TEST_SOURCE_FILES
    ${COMMON_SOURCE_FILES}
    "${PROJECT_SOURCE_DIR}/jetson_test.cpp"
    "${PROJECT_SOURCE_DIR}/jetson_apis.cpp"
)

# 2) jetson_apis 的源文件（额外包含 jetson_apis.cpp）
set(JETSON_APIS_SOURCE_FILES
    ${COMMON_SOURCE_FILES}
    "${PROJECT_SOURCE_DIR}/jetson_apis.cpp"
)

# ------------------ 生成目标：jetson_test (可执行文件) ------------------
add_executable(jetson_test ${JETSON_TEST_SOURCE_FILES})

target_include_directories(jetson_test PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${SIMPLE_CUDA_TOOLKITS_INC}
    ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(jetson_test PRIVATE
    ${CUDA_LIBRARIES}
    # 建议将库文件路径用引号括起来，防止路径中出现空格问题
    "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
    ${TENSORRT_LIBS}
    ${OpenCV_LIBRARIES}
)

# ------------------ 生成目标：jetson_shared (共享库) ------------------
add_library(jetson_shared SHARED ${JETSON_APIS_SOURCE_FILES})
set_target_properties(jetson_shared PROPERTIES
    OUTPUT_NAME "jetson"
    # 默认在 Linux 下共享库后缀为 .so，可省略 SUFFIX 设置
)

target_include_directories(jetson_shared PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${SIMPLE_CUDA_TOOLKITS_INC}
    ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(jetson_shared PRIVATE
    ${CUDA_LIBRARIES}
    "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
    ${TENSORRT_LIBS}
    ${OpenCV_LIBRARIES}
)

# ------------------ 生成目标：jetson_static (静态库) ------------------
add_library(jetson_static STATIC ${JETSON_APIS_SOURCE_FILES})
set_target_properties(jetson_static PROPERTIES
    OUTPUT_NAME "jetson"
    # 默认在 Linux 下静态库后缀为 .a，可省略 SUFFIX 设置
)

target_include_directories(jetson_static PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${SIMPLE_CUDA_TOOLKITS_INC}
    ${TENSORRT_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(jetson_static PRIVATE
    ${CUDA_LIBRARIES}
    "${SIMPLE_CUDA_TOOLKITS_LIB}/libsimplecuda.a"
    ${TENSORRT_LIBS}
    ${OpenCV_LIBRARIES}
)

# ------------------ 安装规则 (可选) ------------------
install(TARGETS
    jetson_shared
    jetson_static
    RUNTIME DESTINATION bin        # 针对可执行文件
    LIBRARY DESTINATION lib        # 针对动态库
    ARCHIVE DESTINATION lib        # 针对静态库
)

install(TARGETS jetson_test
    RUNTIME DESTINATION bin
)

install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION include
)
