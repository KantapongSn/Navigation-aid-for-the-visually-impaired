import cv2
import numpy as np
import os

def load_yolo_model():
    home = os.path.expanduser("~")
    labelsPath = os.path.join(home, "yolo-coco/custom.names")
    weightsPath = os.path.join(home, "yolo-coco/stairs-yolov3-tiny_last(5).weights")
    configPath = os.path.join(home, "yolo-coco/stairs-yolov3-tiny.cfg")

    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return net, LABELS, COLORS

