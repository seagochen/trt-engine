#include "MQTTClient.h"
#include <iostream>

MQTTClient::MQTTClient(const std::string& address, int port, const std::string& clientId, const std::string& username, const std::string& password)
        : address_(address), clientId_(clientId), port_(port), isConnected_(false) {

    mosquitto_lib_init();
    mosq_ = mosquitto_new(clientId_.c_str(), true, this);
    if (!mosq_) {
        throw std::runtime_error("Failed to initialize Mosquitto client.");
    }

    if (!username.empty() && !password.empty()) {
        mosquitto_username_pw_set(mosq_, username.c_str(), password.c_str());
    }

    mosquitto_connect_callback_set(mosq_, MQTTClient::on_connect);
    mosquitto_disconnect_callback_set(mosq_, MQTTClient::on_disconnect);
    mosquitto_message_callback_set(mosq_, MQTTClient::on_message);
}

MQTTClient::~MQTTClient() {
    disconnect();
    mosquitto_destroy(mosq_);
    mosquitto_lib_cleanup();
}

void MQTTClient::setMessageCallback(std::function<void(const std::string&, const void*, size_t)> callback) {
    messageCallback_ = std::move(callback);
}

bool MQTTClient::connect() {
    int ret = mosquitto_connect(mosq_, address_.c_str(), port_, 60);
    return checkError(ret, "connect");
}

void MQTTClient::disconnect() {
    if (mosq_ && isConnected_) {
        int ret = mosquitto_disconnect(mosq_);
        if (checkError(ret, "disconnect")) {
            isConnected_ = false;  // 标记为未连接
        }
    }
}

bool MQTTClient::publish(const std::string& topic, const void* payload, size_t payloadlen) {
    int ret = mosquitto_publish(mosq_, nullptr, topic.c_str(), payloadlen, payload, 0, false);
    return checkError(ret, "publish");
}

bool MQTTClient::subscribe(const std::string& topic) {
    int ret = mosquitto_subscribe(mosq_, nullptr, topic.c_str(), 0);
    return checkError(ret, "subscribe");
}

// 手动处理事件循环
bool MQTTClient::listen(int timeout) {
    int ret = mosquitto_loop(mosq_, timeout, 1);  // 使用非阻塞模式
    return checkError(ret, "listen");
}

// Static callback functions
void MQTTClient::on_message(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg) {
    auto* client = static_cast<MQTTClient*>(obj);
    if (client->messageCallback_) {
        // 传递主题、payload（未转换为字符串）和长度
        client->messageCallback_(msg->topic, msg->payload, msg->payloadlen);
    }
}

void MQTTClient::on_connect(struct mosquitto* mosq, void* obj, int rc) {
    if (rc == 0) {
        std::cout << "Connected to the MQTT broker successfully." << std::endl;
        auto* client = static_cast<MQTTClient*>(obj);
        client->isConnected_ = true;  // 设置连接成功
    } else {
        std::cerr << "Failed to connect: " << mosquitto_strerror(rc) << std::endl;
    }
}

void MQTTClient::on_disconnect(struct mosquitto* mosq, void* obj, int rc) {
    auto* client = static_cast<MQTTClient*>(obj);
    if (rc == 0) {
        std::cout << "Disconnected from the MQTT broker gracefully." << std::endl;
    } else {
        std::cerr << "Unexpected disconnection from the MQTT broker: " << mosquitto_strerror(rc) << std::endl;
    }
    client->isConnected_ = false;
}

// 统一错误检查和日志输出
bool MQTTClient::checkError(int resultCode, const std::string& operation) {
    if (resultCode != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to " << operation << ": " << mosquitto_strerror(resultCode) << std::endl;
        return false;
    }
    return true;
}
