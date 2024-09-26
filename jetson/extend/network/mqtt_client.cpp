//
// Created by ubuntu on 9/9/24.
//

#include "mqtt_client.h"
#include <iostream>


MQTTClient::MQTTClient(const std::string& address, int port, const std::string& clientId, const std::string& username, const std::string& password)
        : address(address), clientId(clientId), port(port), isConnected(false) {

    mosquitto_lib_init();
    mosq = mosquitto_new(clientId.c_str(), true, this);
    if (!mosq) {
        throw std::runtime_error("Failed to initialize Mosquitto client.");
    }

    if (!username.empty() && !password.empty()) {
        mosquitto_username_pw_set(mosq, username.c_str(), password.c_str());
    }

    mosquitto_connect_callback_set(mosq, MQTTClient::on_connect);
    mosquitto_disconnect_callback_set(mosq, MQTTClient::on_disconnect);
    mosquitto_message_callback_set(mosq, MQTTClient::on_message);
}

MQTTClient::~MQTTClient() {
    disconnect();
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
}

void MQTTClient::setMessageCallback(std::function<void(const std::string&, const void*, size_t)> callback) {
    messageCallback = std::move(callback);
}

bool MQTTClient::connect() {
    int ret = mosquitto_connect(mosq, address.c_str(), port, 60);
    if (checkError(ret, "connect")) {
        // 启动Mosquitto库自带的异步事件循环线程
        ret = mosquitto_loop_start(mosq);
        return checkError(ret, "start event loop");
    }
    return false;
}

void MQTTClient::disconnect() {
    if (mosq && isConnected) {
        int ret = mosquitto_disconnect(mosq);
        if (checkError(ret, "disconnect")) {
            isConnected = false;
            // 停止事件循环线程
            mosquitto_loop_stop(mosq, true);
        }
    }
}

bool MQTTClient::publish(const std::string& topic, const void* payload, size_t payloadlen) {
    int ret = mosquitto_publish(mosq, nullptr, topic.c_str(), payloadlen, payload, 0, false);
    return checkError(ret, "publish");
}

bool MQTTClient::subscribe(const std::string& topic) {
    int ret = mosquitto_subscribe(mosq, nullptr, topic.c_str(), 0);
    return checkError(ret, "subscribe");
}

// Static callback functions
void MQTTClient::on_message(struct mosquitto* mosq, void* obj, const struct mosquitto_message* msg) {
    auto* client = static_cast<MQTTClient*>(obj);
    if (client->messageCallback) {
        client->messageCallback(msg->topic, msg->payload, msg->payloadlen);
    }
}

void MQTTClient::on_connect(struct mosquitto* mosq, void* obj, int rc) {
    if (rc == 0) {
        std::cout << "Connected to the MQTT broker successfully." << std::endl;
        auto* client = static_cast<MQTTClient*>(obj);
        client->isConnected = true;
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
    client->isConnected = false;
}

// 统一错误检查和日志输出
bool MQTTClient::checkError(int resultCode, const std::string& operation) {
    if (resultCode != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to " << operation << ": " << mosquitto_strerror(resultCode) << std::endl;
        return false;
    }
    return true;
}
