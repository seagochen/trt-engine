cmake_minimum_required(VERSION 3.16)

# Define the project
project(adapter)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)              # C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # C++标准是必须的
set(CMAKE_CXX_EXTENSIONS OFF)           # C++标准不是扩展

# Enable OpenCV
find_package(OpenCV REQUIRED)

# Enable Mosquitto
find_package(PkgConfig REQUIRED)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)

# Enable YAML
find_package(yaml-cpp REQUIRED)

# Find Protobuf
find_package(Protobuf REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${MOSQUITTO_INCLUDE_DIRS})
include_directories(${YAML_CPP_INCLUDE_DIRS})
include_directories(${Protobuf_INCLUDE_DIRS})

# Add source files (make sure to include all necessary directories)
file(GLOB_RECURSE SRC_FILES common/*.cpp protobufs/*.pb.cc)
add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES})

# Add common, protobufs headers
target_include_directories(adapter PRIVATE ${PROJECT_SOURCE_DIR})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${MOSQUITTO_LIBRARIES})
target_link_libraries(${PROJECT_NAME} yaml-cpp)
target_link_libraries(${PROJECT_NAME} ${Protobuf_LIBRARIES})