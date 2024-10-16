from common.yolo.simple_structs import Yolo, YoloPose, YoloPoint


def process_yolo(results) -> list:
    # A list to store the results
    yolo_objects = []

    # Loop through the results
    for det in results.boxes:
        # Get the bounding box
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        # Get the confidence
        # conf = det.conf # CUDA tensor
        conf = det.conf.cpu().numpy().item()

        # Get the class
        cls = int(det.cls[0])

        # Create a yolo object to store the results
        yolo = Yolo(x1, y1, x2, y2, cls, conf)

        # Append the yolo object to the list
        yolo_objects.append(yolo)

    return yolo_objects


def process_pose(results) -> list:
    # A list to store the results
    yolo_objects = []

    # Loop through the results
    for box, kpts in zip(results.boxes, results.keypoints):
        # Get the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the confidence
        # conf = det.conf # CUDA tensor
        conf = box.conf.cpu().numpy().item()

        # Create a list to store the keypoints
        keypoints = []

        # Squeeze the keypoints
        kpts_xy = kpts.xy.cpu().numpy().squeeze()  # (17, 2)
        kpts_conf = kpts.conf.cpu().numpy().squeeze()  # (17,)

        # Loop through the keypoints
        for (x, y), c in zip(kpts_xy, kpts_conf):
            # Create a yolo point object
            point = YoloPoint(int(x), int(y), c)

            # Append the yolo point object to the list
            keypoints.append(point)

        # Create a yolo pose object to store the results
        pose = YoloPose(x1, y1, x2, y2, conf, keypoints)

        # Append the yolo pose object to the list
        yolo_objects.append(pose)

    return yolo_objects
