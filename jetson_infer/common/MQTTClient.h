//
// Created by ubuntu on 9/9/24.
//

#ifndef VIDEO_ADP_MQTTCLIENT_H
#define VIDEO_ADP_MQTTCLIENT_H

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

    // 手动处理Mosquitto事件循环
    bool listen(int timeout);

private:
    std::string address_;
    std::string clientId_;
    std::optional<std::string> username_;  // 使用 std::optional 来管理可选的用户名
    std::optional<std::string> password_;  // 使用 std::optional 来管理可选的密码
    mosquitto* mosq_;
    int port_;
    std::function<void(const std::string&, const void*, size_t)> messageCallback_;

    std::atomic<bool> isConnected_;  // 标志位：表示是否连接上
    static void on_message(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg);
    static void on_connect(struct mosquitto* mosq, void* obj, int rc);
    static void on_disconnect(struct mosquitto* mosq, void* obj, int rc);

    // 用于统一处理Mosquitto库调用的结果，减少代码重复
    static bool checkError(int resultCode, const std::string& operation);
};

#endif // VIDEO_ADP_MQTTCLIENT_H

