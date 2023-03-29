"""
Performs inference on a given video.
Parameters:
    --image: path to the image.

example:
    python predict.py --image <path-to-image>

"""

import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from copy import deepcopy
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from statistics import mode
import csv
import os
from tqdm.auto import tqdm

class_to_description = {0: "Right Scissors",
                        1: "Left Scissors",
                        2: "Right Needle driver",
                        3: "Left Needle driver",
                        4: "Right Forceps",
                        5: "Left Forceps",
                        6: "Right Empty",
                        7: "Left Empty"}

class_to_color = {0: (31, 119, 180),
                  1: (255, 127, 14),
                  2: (44, 160, 44),
                  3: (214, 39, 40),
                  4: (148, 103, 189),
                  5: (140, 86, 75),
                  6: (227, 119, 194),
                  7: (127, 127, 127)}

right_tool_to_class = {'T0': 6,
                       'T1': 2,
                       'T2': 4,
                       'T3': 0}

left_tool_to_class = {'T0': 7,
                      'T1': 3,
                      'T2': 5,
                      'T3': 1}

RIGHT = 'right'
LEFT = 'left'

CLASS = 'class'
XYXY = 'xyxy'
CONF = 'conf'

model = None


def draw_bboxes(img, classes, bboxes):
    for cls, box in zip(classes, bboxes):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), class_to_color[cls], 2)


def draw_labels(img, classes, bboxes, confs):
    for cls, box, conf in zip(classes, bboxes, confs):
        xmin, ymin, xmax, ymax = box
        label = f"{class_to_description[cls]} {conf:.2f}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        img = cv2.rectangle(img, (xmin - 1, ymin - 20), (xmin + w, ymin - 1), class_to_color[cls], -1)
        img = cv2.putText(img, label, (xmin, ymin - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


def create_bboxes(img, img_predictions):
    new_img = deepcopy(img)

    classes = []
    bboxes = []
    confs = []

    for hand in img_predictions:
        if img_predictions[hand] is not None:
            classes.append(img_predictions[hand][CLASS])
            confs.append(img_predictions[hand][CONF])
            bboxes.append([round(num) for num in img_predictions[hand][XYXY]])

    draw_bboxes(new_img, classes, bboxes)
    draw_labels(new_img, classes, bboxes, confs)

    return new_img


def extract_from_result(result):
    extracted = {RIGHT: [],
                 LEFT: []}
    for pred in result:
        *xyxy, conf, cls = pred.tolist()
        cls = int(cls)
        hand = RIGHT if cls % 2 == 0 else LEFT
        extracted[hand].append({CLASS: cls, CONF: conf, XYXY: xyxy})
    return extracted


def extract_maximal_confidence(extracted_result):
    new_extracted = {}
    if len(extracted_result[RIGHT]) == 0:
        new_extracted[RIGHT] = None
    else:
        new_extracted[RIGHT] = max(extracted_result[RIGHT], key=lambda ext: ext[CONF])

    if len(extracted_result[LEFT]) == 0:
        new_extracted[LEFT] = None
    else:
        new_extracted[LEFT] = max(extracted_result[LEFT], key=lambda ext: ext[CONF])
    return new_extracted


def get_raw_result(img_path):
    original_img = cv2.imread(img_path)
    RGB_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    result = model(RGB_img)
    return RGB_img, result


def predict_img(img_path):
    img_path_no_ext, _ = os.path.splitext(img_path)

    rgb_img, result = get_raw_result(img_path)
    extracted_result = extract_from_result(result.xyxy[0])
    maximal_conf_result = extract_maximal_confidence(extracted_result)

    labeled_img = create_bboxes(rgb_img, maximal_conf_result)
    labeled_img_BGR = cv2.cvtColor(labeled_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"{img_path_no_ext}_labeled.jpg", labeled_img_BGR)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='image path')

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        parser.error(f"--image is invalid: no such file or directory {args.video}")

    return args


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/large_SGD/weights/best.pt',
                           force_reload=True)
    args = create_parser()
    predict_img(args.image)
