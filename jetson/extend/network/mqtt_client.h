//
// Created by ubuntu on 9/9/24.
//

#ifndef MQTT_CLIENT_H
#define MQTT_CLIENT_H

#include <string>
#include <iostream>
#include <functional>
#include <mosquitto.h>
#include <atomic>
#include <optional>

class MQTTClient {
public:
    MQTTClient(const std::string& address, int port, const std::string& clientId, const std::string& username = "", const std::string& password = "");
    ~MQTTClient();

    // 设置接收消息的回调函数
    void setMessageCallback(std::function<void(const std::string&, const void*, size_t)> callback);

    bool connect();
    void disconnect();
    bool publish(const std::string& topic, const void* payload, size_t payloadlen);
    bool subscribe(const std::string& topic);

private:
    std::string address;
    std::string clientId;
    std::optional<std::string> username;  // 使用 std::optional 来管理可选的用户名
    std::optional<std::string> password;  // 使用 std::optional 来管理可选的密码
    mosquitto* mosq;
    int port;
    std::function<void(const std::string&, const void*, const size_t)> messageCallback;

    std::atomic<bool> isConnected;  // 标志位：表示是否连接上

    static void on_message(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg);
    static void on_connect(struct mosquitto* mosq, void* obj, int rc);
    static void on_disconnect(struct mosquitto* mosq, void* obj, int rc);

    // 用于统一处理Mosquitto库调用的结果，减少代码重复
    static bool checkError(int resultCode, const std::string& operation);
};

#endif // MQTT_CLIENT_H

