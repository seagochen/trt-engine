import cv2
import numpy as np

from pyengine.inference.yolo import YoloWrapper
from pyengine.inference.yolo.infer_drawer import InferenceDrawer
from pyengine.inference.yolo.schema_loader import SchemaLoader
from pyengine.utils.load_labels import read_labels_from_file


def test_load_yolo_v8(image: np.ndarray, drawer: InferenceDrawer, coco_labels: list):

    # Load the model
    object_model = YoloWrapper("/opt/models/yolov8n.engine", use_pose=False)

    # Add image to the model
    object_model.add_image(0, img)

    # Run inference
    object_model.inference()

    # Get the number of results
    count = object_model.available_results(0, 0.5, 0.5)

    # Iterate through the results and put them in a list
    results = []
    for i in range(count):
        result = object_model.get_result(i)
        results.append(result)

    # Release the model
    object_model.release()

    # Draw the inference results on the image
    updated_img = drawer.draw_objects(image, results, labels=coco_labels)

    # Save the image
    cv2.imwrite("py_wrapper_yolo_object_test.png", updated_img)


def test_load_yolo_pose_v8(image: np.ndarray, drawer: InferenceDrawer):

    # Load the model
    object_model = YoloWrapper("/opt/models/yolov8n-pose.engine", use_pose=True)

    # Add image to the model
    object_model.add_image(0, img)

    # Run inference
    object_model.inference()

    # Get the number of results
    count = object_model.available_results(0, 0.5, 0.5)

    # Iterate through the results and put them in a list
    results = []
    for i in range(count):
        result = object_model.get_result(i)
        results.append(result)

    # Release the model
    object_model.release()

    # Draw the inference results on the image
    updated_img = drawer.draw_skeletons(image, results, show_skeletons=True, show_pts=True, show_pts_name=False)

    # Save the image
    cv2.imwrite("py_wrapper_yolo_pose_test.png", updated_img)


if __name__ == "__main__":
    # Load an image
    img = cv2.imread("res/test.png")

    # Resize the image to 640x640
    img = cv2.resize(img, (640, 640))

    # Create a SchemaLoader object and load the schema
    drawer = InferenceDrawer(SchemaLoader())

    # Load the coco labels
    coco_labels = read_labels_from_file("/opt/TrtEngineToolkits/res/coco_labels.txt")

    # Draw the inference results on the image
    test_load_yolo_v8(img, drawer, coco_labels)
    test_load_yolo_pose_v8(img, drawer)
