def iou(box1, box2):
    """计算两个边界框的交并比（IoU）。"""
    x1_inter = max(box1.lx, box2.lx)
    y1_inter = max(box1.ly, box2.ly)
    x2_inter = min(box1.rx, box2.rx)
    y2_inter = min(box1.ry, box2.ry)

    # 计算相交区域的面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 计算两个边界框的面积
    box1_area = (box1.rx - box1.lx) * (box1.ry - box1.ly)
    box2_area = (box2.rx - box2.lx) * (box2.ry - box2.ly)

    # 计算IoU
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-6)
    return iou


def for_yolo(yolo_objects, iou_threshold=0.5, confidence_threshold=0.0):
    """对Yolo对象列表执行非极大值抑制（NMS），根据类别进行抑制，并加入置信度过滤。"""
    # 首先过滤掉低于置信度阈值的目标
    yolo_objects = [obj for obj in yolo_objects if obj.conf >= confidence_threshold]

    # 按照置信度从高到低排序
    yolo_objects = sorted(yolo_objects, key=lambda x: x.conf, reverse=True)
    suppressed = []

    while yolo_objects:
        # 选择当前置信度最高的框
        current = yolo_objects.pop(0)
        suppressed.append(current)

        # 临时列表用于保存剩余的框
        remaining_objects = []

        for obj in yolo_objects:
            # 只在类别相同时，才对其进行NMS判断
            if obj.cls == current.cls:
                iou_value = iou(current, obj)
                if iou_value < iou_threshold:
                    remaining_objects.append(obj)
            else:
                remaining_objects.append(obj)

        yolo_objects = remaining_objects

    return suppressed


def for_pose(yolo_pose_objects, iou_threshold=0.5, confidence_threshold=0.0):
    """对YoloPose对象列表执行非极大值抑制（NMS），不考虑类别，并加入置信度过滤。"""
    # 首先过滤掉低于置信度阈值的目标
    yolo_pose_objects = [obj for obj in yolo_pose_objects if obj.conf >= confidence_threshold]

    # 按照置信度从高到低排序
    yolo_pose_objects = sorted(yolo_pose_objects, key=lambda x: x.conf, reverse=True)
    suppressed = []

    while yolo_pose_objects:
        # 选择当前置信度最高的框
        current = yolo_pose_objects.pop(0)
        suppressed.append(current)

        # 临时列表用于保存剩余的框
        remaining_objects = []

        for obj in yolo_pose_objects:
            # NMS基于IoU，但不考虑类别
            iou_value = iou(current, obj)
            if iou_value < iou_threshold:
                remaining_objects.append(obj)

        yolo_pose_objects = remaining_objects

    return suppressed