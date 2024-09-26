import json

import cv2
import numpy as np
import pandas as pd


class YoloInferenceResults:
    def __init__(self):
        self.frame_number = None
        self.frame_data = None
        self.frame_ready = False
        self.results = None
        self.results_ready = False

    def __lt__(self, other):
        return self.frame_number < other.frame_number

    def is_ready(self):
        return self.frame_ready and self.results_ready


def apply_nms(df, conf_threshold=0.5, nms_threshold=0.4):
    df = df[df['conf'] > conf_threshold]

    if df.empty:
        return df

    boxes = df[['lx', 'ly', 'rx', 'ry']].values
    confidences = df['conf'].values

    boxes_cv = np.array([
        [x, y, (rx - x), (ry - y)]
        for x, y, rx, ry in boxes
    ], dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(boxes_cv.tolist(), confidences.tolist(), conf_threshold, nms_threshold)

    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        filtered_df = df.iloc[indices]
    else:
        filtered_df = pd.DataFrame(columns=df.columns)

    return filtered_df


def parse_yolo_str(yolo_result: str):
    """
    json.loads(yolo_result)

    output of json.loads(yolo_result):

    [{'cls': 2, 'conf': 0.11083984375, 'keypoints': [], 'lx': 21, 'ly': 335, 'rx': 75, 'ry': 356}, {'cls': 2, 'conf': 0.4111328125, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 172, 'ry': 387}, {'cls': 2, 'conf': 0.385009765625, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 172, 'ry': 387}, {'cls': 2, 'conf': 0.43212890625, 'keypoints': [], 'lx': 234, 'ly': 351, 'rx': 264, 'ry': 369}, {'cls': 2, 'conf': 0.42822265625, 'keypoints': [], 'lx': 234, 'ly': 350, 'rx': 263, 'ry': 369}, {'cls': 2, 'conf': 0.2568359375, 'keypoints': [], 'lx': 234, 'ly': 350, 'rx': 264, 'ry': 370}, {'cls': 2, 'conf': 0.1607666015625, 'keypoints': [], 'lx': 369, 'ly': 352, 'rx': 393, 'ry': 369}, {'cls': 2, 'conf': 0.267333984375, 'keypoints': [], 'lx': 370, 'ly': 352, 'rx': 393, 'ry': 369}, {'cls': 2, 'conf': 0.56591796875, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 172, 'ry': 388}, {'cls': 2, 'conf': 0.515625, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 173, 'ry': 388}, {'cls': 2, 'conf': 0.127197265625, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 195, 'ry': 389}, {'cls': 2, 'conf': 0.264404296875, 'keypoints': [], 'lx': 234, 'ly': 351, 'rx': 263, 'ry': 369}, {'cls': 2, 'conf': 0.26904296875, 'keypoints': [], 'lx': 234, 'ly': 350, 'rx': 263, 'ry': 370}, {'cls': 2, 'conf': 0.2451171875, 'keypoints': [], 'lx': 234, 'ly': 350, 'rx': 263, 'ry': 370}, {'cls': 0, 'conf': 0.158203125, 'keypoints': [], 'lx': 366, 'ly': 359, 'rx': 380, 'ry': 376}, {'cls': 2, 'conf': 0.287841796875, 'keypoints': [], 'lx': 374, 'ly': 353, 'rx': 393, 'ry': 369}, {'cls': 0, 'conf': 0.1298828125, 'keypoints': [], 'lx': 483, 'ly': 354, 'rx': 500, 'ry': 398}, {'cls': 2, 'conf': 0.640625, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 172, 'ry': 387}, {'cls': 2, 'conf': 0.5927734375, 'keypoints': [], 'lx': 148, 'ly': 347, 'rx': 173, 'ry': 388}, {'cls': 0, 'conf': 0.138427734375, 'keypoints': [], 'lx': 365, 'ly': 360, 'rx': 382, 'ry': 380}, {'cls': 0, 'conf': 0.1556396484375, 'keypoints': [], 'lx': 487, 'ly': 358, 'rx': 499, 'ry': 399}, {'cls': 2, 'conf': 0.352294921875, 'keypoints': [], 'lx': 149, 'ly': 348, 'rx': 171, 'ry': 388}, {'cls': 2, 'conf': 0.138427734375, 'keypoints': [], 'lx': 148, 'ly': 348, 'rx': 174, 'ry': 388}, {'cls': 0, 'conf': 0.1466064453125, 'keypoints': [], 'lx': 486, 'ly': 359, 'rx': 499, 'ry': 400}, {'cls': 2, 'conf': 0.24658203125, 'keypoints': [], 'lx': 160, 'ly': 354, 'rx': 217, 'ry': 413}, {'cls': 0, 'conf': 0.173828125, 'keypoints': [], 'lx': 486, 'ly': 358, 'rx': 499, 'ry': 402}, {'cls': 0, 'conf': 0.12420654296875, 'keypoints': [], 'lx': 486, 'ly': 359, 'rx': 498, 'ry': 403}, {'cls': 5, 'conf': 0.61328125, 'keypoints': [], 'lx': 286, 'ly': 295, 'rx': 355, 'ry': 404}, {'cls': 5, 'conf': 0.638671875, 'keypoints': [], 'lx': 286, 'ly': 294, 'rx': 355, 'ry': 403}, {'cls': 5, 'conf': 0.57177734375, 'keypoints': [], 'lx': 286, 'ly': 294, 'rx': 355, 'ry': 402}, {'cls': 5, 'conf': 0.638671875, 'keypoints': [], 'lx': 286, 'ly': 295, 'rx': 355, 'ry': 402}, {'cls': 2, 'conf': 0.1163330078125, 'keypoints': [], 'lx': 1, 'ly': 335, 'rx': 74, 'ry': 369}, {'cls': 2, 'conf': 0.123779296875, 'keypoints': [], 'lx': 3, 'ly': 335, 'rx': 75, 'ry': 369}, {'cls': 5, 'conf': 0.60400390625, 'keypoints': [], 'lx': 286, 'ly': 295, 'rx': 357, 'ry': 403}, {'cls': 5, 'conf': 0.6279296875, 'keypoints': [], 'lx': 286, 'ly': 294, 'rx': 356, 'ry': 404}, {'cls': 5, 'conf': 0.66552734375, 'keypoints': [], 'lx': 286, 'ly': 295, 'rx': 356, 'ry': 403}, {'cls': 7, 'conf': 0.21337890625, 'keypoints': [], 'lx': 414, 'ly': 314, 'rx': 493, 'ry': 387}, {'cls': 7, 'conf': 0.20556640625, 'keypoints': [], 'lx': 415, 'ly': 313, 'rx': 495, 'ry': 394}, {'cls': 7, 'conf': 0.17333984375, 'keypoints': [], 'lx': 416, 'ly': 314, 'rx': 495, 'ry': 396}, {'cls': 7, 'conf': 0.24365234375, 'keypoints': [], 'lx': 416, 'ly': 314, 'rx': 496, 'ry': 400}, {'cls': 2, 'conf': 0.43212890625, 'keypoints': [], 'lx': 148, 'ly': 349, 'rx': 172, 'ry': 388}, {'cls': 2, 'conf': 0.32421875, 'keypoints': [], 'lx': 148, 'ly': 349, 'rx': 173, 'ry': 388}, {'cls': 2, 'conf': 0.4091796875, 'keypoints': [], 'lx': 159, 'ly': 352, 'rx': 217, 'ry': 413}, {'cls': 2, 'conf': 0.544921875, 'keypoints': [], 'lx': 159, 'ly': 352, 'rx': 218, 'ry': 413}, {'cls': 2, 'conf': 0.11676025390625, 'keypoints': [], 'lx': 233, 'ly': 351, 'rx': 263, 'ry': 369}, {'cls': 5, 'conf': 0.4072265625, 'keypoints': [], 'lx': 286, 'ly': 295, 'rx': 357, 'ry': 404}, {'cls': 5, 'conf': 0.59814453125, 'keypoints': [], 'lx': 286, 'ly': 294, 'rx': 357, 'ry': 405}, {'cls': 5, 'conf': 0.640625, 'keypoints': [], 'lx': 286, 'ly': 294, 'rx': 356, 'ry': 405}, {'cls': 7, 'conf': 0.236572265625, 'keypoints': [], 'lx': 416, 'ly': 313, 'rx': 495, 'ry': 398}, {'cls': 7, 'conf': 0.1480712890625, 'keypoints': [], 'lx': 416, 'ly': 313, 'rx': 496, 'ry': 397}, {'cls': 7, 'conf': 0.1124267578125, 'keypoints': [], 'lx': 417, 'ly': 313, 'rx': 495, 'ry': 400}, {'cls': 2, 'conf': 0.8447265625, 'keypoints': [], 'lx': 0, 'ly': 344, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.8291015625, 'keypoints': [], 'lx': 0, 'ly': 344, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.279052734375, 'keypoints': [], 'lx': 148, 'ly': 349, 'rx': 171, 'ry': 388}, {'cls': 2, 'conf': 0.74755859375, 'keypoints': [], 'lx': 159, 'ly': 353, 'rx': 218, 'ry': 413}, {'cls': 2, 'conf': 0.71240234375, 'keypoints': [], 'lx': 160, 'ly': 353, 'rx': 218, 'ry': 413}, {'cls': 2, 'conf': 0.875, 'keypoints': [], 'lx': 0, 'ly': 345, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.83984375, 'keypoints': [], 'lx': 0, 'ly': 345, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.1307373046875, 'keypoints': [], 'lx': 0, 'ly': 345, 'rx': 129, 'ry': 432}, {'cls': 2, 'conf': 0.71533203125, 'keypoints': [], 'lx': 160, 'ly': 353, 'rx': 218, 'ry': 412}, {'cls': 2, 'conf': 0.75634765625, 'keypoints': [], 'lx': 160, 'ly': 353, 'rx': 218, 'ry': 412}, {'cls': 2, 'conf': 0.69775390625, 'keypoints': [], 'lx': 160, 'ly': 353, 'rx': 219, 'ry': 413}, {'cls': 2, 'conf': 0.276611328125, 'keypoints': [], 'lx': 0, 'ly': 344, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.22265625, 'keypoints': [], 'lx': 0, 'ly': 345, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.72509765625, 'keypoints': [], 'lx': 160, 'ly': 354, 'rx': 219, 'ry': 413}, {'cls': 2, 'conf': 0.65478515625, 'keypoints': [], 'lx': 160, 'ly': 354, 'rx': 219, 'ry': 413}, {'cls': 7, 'conf': 0.18896484375, 'keypoints': [], 'lx': 415, 'ly': 314, 'rx': 496, 'ry': 403}, {'cls': 7, 'conf': 0.1749267578125, 'keypoints': [], 'lx': 415, 'ly': 314, 'rx': 497, 'ry': 403}, {'cls': 2, 'conf': 0.810546875, 'keypoints': [], 'lx': 0, 'ly': 343, 'rx': 129, 'ry': 434}, {'cls': 2, 'conf': 0.87646484375, 'keypoints': [], 'lx': 0, 'ly': 343, 'rx': 129, 'ry': 434}, {'cls': 7, 'conf': 0.1422119140625, 'keypoints': [], 'lx': 415, 'ly': 313, 'rx': 496, 'ry': 404}, {'cls': 7, 'conf': 0.1357421875, 'keypoints': [], 'lx': 416, 'ly': 313, 'rx': 496, 'ry': 405}, {'cls': 2, 'conf': 0.74462890625, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 503}, {'cls': 2, 'conf': 0.76220703125, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 503}, {'cls': 2, 'conf': 0.84375, 'keypoints': [], 'lx': 0, 'ly': 344, 'rx': 129, 'ry': 434}, {'cls': 2, 'conf': 0.865234375, 'keypoints': [], 'lx': 0, 'ly': 343, 'rx': 129, 'ry': 434}, {'cls': 2, 'conf': 0.1251220703125, 'keypoints': [], 'lx': 0, 'ly': 344, 'rx': 129, 'ry': 433}, {'cls': 2, 'conf': 0.86865234375, 'keypoints': [], 'lx': 295, 'ly': 362, 'rx': 410, 'ry': 526}, {'cls': 2, 'conf': 0.8798828125, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 410, 'ry': 525}, {'cls': 2, 'conf': 0.7216796875, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 501}, {'cls': 2, 'conf': 0.708984375, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 502}, {'cls': 2, 'conf': 0.75048828125, 'keypoints': [], 'lx': 551, 'ly': 329, 'rx': 640, 'ry': 502}, {'cls': 2, 'conf': 0.88330078125, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 410, 'ry': 525}, {'cls': 2, 'conf': 0.900390625, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 410, 'ry': 525}, {'cls': 2, 'conf': 0.9033203125, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 410, 'ry': 525}, {'cls': 2, 'conf': 0.708984375, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 502}, {'cls': 2, 'conf': 0.69091796875, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 502}, {'cls': 2, 'conf': 0.71875, 'keypoints': [], 'lx': 551, 'ly': 329, 'rx': 640, 'ry': 502}, {'cls': 2, 'conf': 0.873046875, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 410, 'ry': 526}, {'cls': 2, 'conf': 0.8896484375, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 410, 'ry': 526}, {'cls': 2, 'conf': 0.89697265625, 'keypoints': [], 'lx': 296, 'ly': 363, 'rx': 410, 'ry': 526}, {'cls': 2, 'conf': 0.6943359375, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 502}, {'cls': 2, 'conf': 0.658203125, 'keypoints': [], 'lx': 550, 'ly': 329, 'rx': 639, 'ry': 502}, {'cls': 2, 'conf': 0.91162109375, 'keypoints': [], 'lx': 296, 'ly': 362, 'rx': 409, 'ry': 526}, {'cls': 2, 'conf': 0.91650390625, 'keypoints': [], 'lx': 297, 'ly': 362, 'rx': 410, 'ry': 526}, {'cls': 2, 'conf': 0.236572265625, 'keypoints': [], 'lx': 277, 'ly': 596, 'rx': 617, 'ry': 640}, {'cls': 2, 'conf': 0.56787109375, 'keypoints': [], 'lx': 290, 'ly': 596, 'rx': 626, 'ry': 639}, {'cls': 2, 'conf': 0.63525390625, 'keypoints': [], 'lx': 293, 'ly': 597, 'rx': 635, 'ry': 640}, {'cls': 2, 'conf': 0.7138671875, 'keypoints': [], 'lx': 293, 'ly': 598, 'rx': 639, 'ry': 640}, {'cls': 2, 'conf': 0.5908203125, 'keypoints': [], 'lx': 292, 'ly': 598, 'rx': 637, 'ry': 640}, {'cls': 2, 'conf': 0.451416015625, 'keypoints': [], 'lx': 293, 'ly': 597, 'rx': 638, 'ry': 640}, {'cls': 2, 'conf': 0.3486328125, 'keypoints': [], 'lx': 292, 'ly': 597, 'rx': 638, 'ry': 640}, {'cls': 2, 'conf': 0.173828125, 'keypoints': [], 'lx': 271, 'ly': 597, 'rx': 639, 'ry': 639}]Disconnected from the MQTT broker gracefully.
    """

    # Step 1: Parse YOLO result string
    yolo_results = json.loads(yolo_result)

    # Step 2: Convert to DataFrame
    df = pd.DataFrame(yolo_results)

    # Step 3: Apply NMS
    filtered_df = apply_nms(df, conf_threshold=0.5, nms_threshold=0.4)

    return filtered_df