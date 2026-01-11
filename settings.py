YOLO_MODEL_PATH = "yolo11m.pt"
FEATURE_MODEL_PATH = "mars-small128.pb"

IMAGE_WIDTH = 416
IMAGE_HEIGHT = 416
DETECTION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = 0.8

COSINE_DISTANCE_THRESHOLD = 0.7
EUCLIDEAN_DISTANCE_THRESHOLD = 0.7
FEATURE_BUDGET = None

IOU_THRESHOLD = 0.7
MAX_TRACK_AGE = 30
TRACK_CONFIRMATION_HITS = 3

POSITION_WEIGHT = 1.0 / 20
VELOCITY_WEIGHT = 1.0 / 160

CHI_SQUARE_95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
    5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}

OBJECT_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic-light',
    10: 'fire-hydrant', 11: 'stop-sign', 12: 'parking-meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports-ball', 33: 'kite',
    34: 'baseball-bat', 35: 'baseball-glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis-racket', 39: 'bottle', 40: 'wine-glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot-dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed', 60: 'diningtable',
    61: 'toilet', 62: 'tvmonitor', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell-phone', 68: 'microwave', 69: 'oven',
    70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy-bear', 78: 'hair-drier',
    79: 'toothbrush'
}
