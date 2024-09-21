#!/bin/bash

# 卸载 OpenCV
echo "Uninstalling OpenCV..."
sudo apt remove --purge -y libopencv-dev

# 卸载 Protobuf
echo "Uninstalling Protobuf..."
sudo apt remove --purge -y protobuf-compiler libprotobuf-dev

# 卸载 YAML 库
echo "Uninstalling YAML library..."
sudo apt remove --purge -y libyaml-cpp-dev

# 卸载 Mosquitto
echo "Uninstalling Mosquitto..."
sudo apt remove --purge -y mosquitto mosquitto-clients libmosquitto-dev

# 清理不再需要的软件包
echo "Cleaning up unused packages..."
sudo apt autoremove -y

# 清理缓存
echo "Cleaning up cache..."
sudo apt clean

echo "Uninstallation complete."

