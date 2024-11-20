#!/bin/bash

# 检查参数数量是否正确
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <device_id> <detect_code> <local_img_file>"
    exit 1
fi

# 检查文件是否存在
if [ ! -f $3 ]; then
    echo "File not found: $3"
    exit 1
fi

# 获取参数
DEVICE_ID=$1
DETECT_CODE=$2
LOCAL_FILENAME=$3

# 自动生成远程文件名
REMOTE_FILENAME=$(basename $LOCAL_FILENAME)

# 连接参数
HOSTNAME="ee-iothub-001.azure-devices.net"
SHARED_ACCESS_KEY="Cxw7v1hcnYYlSzrHXRs07Ppe1LcdP5p8yAIoTL7pcCM="
DETECT_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 获取 SAS URL
SAS_URL=$(curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"key":"value"}' \
  "https://azjsfunction006.azurewebsites.net/api/sasurl?deviceid=$DEVICE_ID&filename=$REMOTE_FILENAME")

# 输出获取的 SAS URL
echo "SAS URL: $SAS_URL"

# 上传图片文件到远程服务器
curl -X PUT \
  -T $LOCAL_FILENAME \
  $SAS_URL \
  -H "x-ms-blob-type: BlockBlob"

# 创建消息负载
MSG_TXT="{\"local_camera_id\":\"$DEVICE_ID\",\"detect_time\":\"$DETECT_TIME\",\"detect_code\":\"$DETECT_CODE\",\"detect_imgfile\":\"$REMOTE_FILENAME\"}"

# 生成 SAS 令牌（授权）（注意：需要实现安全的 SAS 令牌生成逻辑）
EXPIRY=$(($(date +%s) + 3600)) # SAS Token 有效期设置为 1 小时
STRING_TO_SIGN=$(printf "%s\n%s" "$HOSTNAME/devices/$DEVICE_ID" "$EXPIRY" | openssl dgst -sha256 -hmac "$SHARED_ACCESS_KEY" -binary | base64)

# 获取 SAS Token
SAS_TOKEN=$(curl -s "https://azjsfunction006.azurewebsites.net/api/index?deviceid=$DEVICE_ID")

# 输出 SAS Token
echo "SAS Token: $SAS_TOKEN"

# 输出消息内容
echo "Message Payload: $MSG_TXT"

# 发送消息到 IoT Hub
curl -X POST \
  "https://$HOSTNAME/devices/$DEVICE_ID/messages/events?api-version=2018-06-30" \
  -H "Authorization: $SAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d "$MSG_TXT"

# 检查发送结果
if [ $? -eq 0 ]; then
    echo "Message successfully sent"
else
    echo "Failed to send message"
fi
