"""
Performs inference on a given video.
Parameters:
    --video: path to the video.
    --right: path to right hand label (optional in order to create metrics file, but required if --left is specified).
    --left: path to left hand label (optional in order to create metrics file, but required if --right is specified).
    --label_window: label smoothing window size (>=0)
    --bbox_window: bounding box smoothing window size (>=0)

examples:
    python video.py --video <path-to-video>
    python video.py --video <path-to-video> --right <path-to-right-label> --left <path-to-left-label>

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

# metrics
ACCURACY = 'accuracy'
F1_MACRO = 'f1 macro'
F1 = 'f1'
RECALL = 'recall'
PRECISION = 'precision'

model = None


def convert_cv2_to_pil(cv_image, is_BGR=True):
    if is_BGR:
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    else:
        img = cv_image
    return Image.fromarray(img)


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


def create_bboxes_one_frame(frame, frame_predictions):
    new_img = deepcopy(frame)

    classes = []
    bboxes = []
    confs = []

    for hand in frame_predictions:
        classes.append(frame_predictions[hand][CLASS])
        confs.append(frame_predictions[hand][CONF])
        bboxes.append([round(num) for num in frame_predictions[hand][XYXY]])

    draw_bboxes(new_img, classes, bboxes)
    draw_labels(new_img, classes, bboxes, confs)

    return new_img


def get_number_of_frames(video_path):
    cap2 = cv2.VideoCapture(video_path)
    length = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def get_raw_results(video_path):
    frames = []
    raw_results = []
    count = 0
    cap = cv2.VideoCapture(video_path)
    frames_count = get_number_of_frames(video_path)

    with tqdm(total=frames_count, desc='Reading video') as p_bar:
        while (cap.isOpened()):
            succ, frame = cap.read()
            if succ:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                result = model(frame_rgb)
                s = result.xyxy[0].shape[0]
                frames.append(frame_rgb)
                raw_results.append(result.xyxy[0])

                count += 1
                p_bar.update()
                if count == frames_count:  # End of video
                    break
    return frames, raw_results


def extract_from_result(result):
    extracted = {RIGHT: [],
                 LEFT: []}
    for pred in result:
        *xyxy, conf, cls = pred.tolist()
        cls = int(cls)
        hand = RIGHT if cls % 2 == 0 else LEFT
        extracted[hand].append({CLASS: cls, CONF: conf, XYXY: xyxy})

    return extracted


def extract_all_results(results):
    all_extracted = []
    for result in results:
        all_extracted.append(extract_from_result(result))
    return all_extracted


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


def extract_only_classes(extracted_results):
    classes = {RIGHT: [],
               LEFT: []}
    for extracted_result in extracted_results:
        right = extracted_result[RIGHT][CLASS] if extracted_result[RIGHT] is not None else -1
        left = extracted_result[LEFT][CLASS] if extracted_result[LEFT] is not None else -1
        classes[RIGHT].append(right)
        classes[LEFT].append(left)
    return classes


def create_ground_truth(gt_path, hand):
    gt = []
    with open(gt_path, 'r') as file:
        for line in file:
            trimmed = line.strip()
            start, stop, tool = trimmed.split()
            start, stop = int(start), int(stop)
            for i in range(start, stop + 1):
                gt.append(right_tool_to_class[tool] if hand == RIGHT else left_tool_to_class[tool])

    return gt


def calculate_classes_metric_one_hand(metric, ground_truth, predictions):
    unique_gt_labels = list(sorted(set(ground_truth)))
    metric_result = metric(ground_truth, predictions, average=None, labels=unique_gt_labels)
    return dict(zip(unique_gt_labels, metric_result))


def calculate_metrics(ground_truth, predictions):
    metrics = {}
    metrics[ACCURACY] = accuracy_score(ground_truth, predictions)
    metrics[F1_MACRO] = f1_score(ground_truth, predictions, average='macro')
    metrics[F1] = calculate_classes_metric_one_hand(f1_score, ground_truth, predictions)
    metrics[PRECISION] = calculate_classes_metric_one_hand(precision_score, ground_truth, predictions)
    metrics[RECALL] = calculate_classes_metric_one_hand(recall_score, ground_truth, predictions)
    return metrics


def find_one_different(predictions, window_size=1):
    for i in range(window_size, len(predictions) - window_size):
        window = predictions[i - window_size: i] + predictions[i + 1: i + window_size + 1]
        if predictions[i] not in window:
            return i, window
    return -1, None


def find_label_to_replace_negative_one(predictions, idx, negative_one_window):
    window_size = negative_one_window
    window = predictions[
             idx - (negative_one_window // 2): idx + (negative_one_window // 2) + 1]  # window of w/2 for each side
    mode_label = mode(window)

    while mode_label == -1:
        window_size += 2
        window = predictions[idx - (window_size // 2): idx + (window_size // 2) + 1]  # window of w/2 for each side
        mode_label = mode(window)
    return mode_label


def fill_missing_labels_one_hand(predictions, negative_one_window=5):
    new_predictions = [pred for pred in predictions]
    if negative_one_window > 0:
        while -1 in new_predictions:
            idx = new_predictions.index(-1)
            new_predictions[idx] = find_label_to_replace_negative_one(new_predictions, idx, negative_one_window)
    return new_predictions


def fill_missing_labels(only_classes):
    right = only_classes[RIGHT]
    left = only_classes[LEFT]
    return {RIGHT: fill_missing_labels_one_hand(right),
            LEFT: fill_missing_labels_one_hand(left)}


def smooth_labels_one_hand(predictions, anomaly_window):
    new_predictions = [pred for pred in predictions]
    if anomaly_window > 0:
        diff_idx, window = find_one_different(new_predictions, anomaly_window)
        while diff_idx != -1:
            new_predictions[diff_idx] = mode(window)
            diff_idx, window = find_one_different(new_predictions, anomaly_window)
    return new_predictions


def smooth_labels(only_classes, anomaly_window):
    right = only_classes[RIGHT]
    left = only_classes[LEFT]
    return {RIGHT: smooth_labels_one_hand(right, anomaly_window=anomaly_window),
            LEFT: smooth_labels_one_hand(left, anomaly_window=anomaly_window)}


def predict_one_missing_bounding_box(past_bboxes):
    '''
    Predics a missing bounding box based on two previous bounding boxes and simple kinematics.
    '''
    prev_prev, prev = past_bboxes
    prev_prev, prev = np.array(prev_prev), np.array(prev)
    velocity = prev - prev_prev  # v = dr/dt
    new_box = prev + velocity  # r = r0 + v*t
    return new_box.tolist()


def find_index(lst, var):
    try:
        return lst.index(var)
    except ValueError:
        return -1


def fill_missing_bounding_boxes_one_hand(bboxes):
    idx = find_index(bboxes, None)
    while idx != -1:
        if idx <= 1:  # No past 2 frames
            bboxes[idx] = deepcopy(bboxes[1 - idx])
        else:
            bboxes[idx] = predict_one_missing_bounding_box([bboxes[idx - 2], bboxes[idx - 1]])
        idx = find_index(bboxes, None)


def fill_missing_bounding_boxes(all_predictions):
    new_predictions = deepcopy(all_predictions)
    right_bboxes = [pred[RIGHT][XYXY] for pred in all_predictions]
    left_bboxes = [pred[LEFT][XYXY] for pred in all_predictions]
    fill_missing_bounding_boxes_one_hand(right_bboxes)  # Inplace
    fill_missing_bounding_boxes_one_hand(left_bboxes)  # Inplace
    for i in range(len(new_predictions)):
        new_predictions[i][RIGHT][XYXY] = right_bboxes[i]
        new_predictions[i][LEFT][XYXY] = left_bboxes[i]
    return new_predictions


def moving_average(bboxes, window_size):
    bboxes_np = [np.array(bbox) for bbox in bboxes]
    ret = []
    half_window_size = window_size // 2
    for i in range(len(bboxes)):
        window = bboxes_np[max(i - half_window_size, 0): min(i + half_window_size + 1, len(bboxes_np))]
        window_actual_size = len(window)
        if window_actual_size == window_size and window_size > 3:
            weights = []
            for i in range(1, half_window_size):
                weights.insert(0, 0.7 ** i)
            for i in range(3):
                weights.append(1)
            for i in range(1, half_window_size):
                weights.append(0.7 ** i)
            total_weights = sum(weights)
            normalized_weight = [w / total_weights for w in weights]
            ret.append(np.average(window, weights=normalized_weight, axis=0).tolist())
        else:
            window_sum = np.sum(window, axis=0)
            ret.append((window_sum / window_actual_size).tolist())
    return ret


def smooth_bboxes(all_predictions, window_size):
    new_predictions = deepcopy(all_predictions)
    right_bboxes = [pred[RIGHT][XYXY] for pred in all_predictions]
    left_bboxes = [pred[LEFT][XYXY] for pred in all_predictions]
    right_smoothed = moving_average(right_bboxes, window_size=window_size)
    left_smoothed = moving_average(left_bboxes, window_size=window_size)
    for i in range(len(new_predictions)):
        new_predictions[i][RIGHT][XYXY] = right_smoothed[i]
        new_predictions[i][LEFT][XYXY] = left_smoothed[i]
    return new_predictions


def create_video_util(video_path, frames, final_predictions):
    video_path_no_ext, _ = os.path.splitext(video_path)
    video = cv2.VideoWriter(f"{video_path_no_ext}_visualized.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for frame, prediction in zip(frames, final_predictions):
        new_frame = create_bboxes_one_frame(frame, prediction)
        cv_img = cv2.cvtColor(np.array(new_frame), cv2.COLOR_RGB2BGR)
        video.write(cv_img)
    cv2.destroyAllWindows()
    video.release()
    return cv_img


def create_metric_row(metric_val):
    row = []
    for i in range(len(class_to_description)):
        if i in metric_val:
            row.append(f"{metric_val[i]}")
        else:
            row.append(None)
    return row


def create_metrics_file(video_path, predictions_list, predictions_prefixes, gt_right_path, gt_left_path):
    video_path_no_ext, _ = os.path.splitext(video_path)
    with open(f"{video_path_no_ext}_prediction_metrics.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', *[class_to_description[i] for i in range(len(class_to_description))]])

        gt_right = create_ground_truth(gt_right_path, RIGHT)[:len(predictions_list[0])]
        gt_left = create_ground_truth(gt_left_path, LEFT)[:len(predictions_list[0])]
        gt_all = gt_right + gt_left  # Concat both hands ground truth

        for predictions, prefix in zip(predictions_list, predictions_prefixes):
            classes = extract_only_classes(predictions)
            classes_all = classes[RIGHT] + classes[LEFT]  # Concat both hands predictions
            metrics = calculate_metrics(gt_all, classes_all)
            writer.writerow([f'{prefix}{"/" if prefix else ""}accuracy', f"{metrics[ACCURACY]}"])
            writer.writerow([f'{prefix}{"/" if prefix else ""}f1-macro', f"{metrics[F1_MACRO]}"])
            for metric_name, metric_val in metrics.items():
                if metric_name not in (ACCURACY, F1_MACRO):
                    partial_row = create_metric_row(metric_val)
                    writer.writerow([f'{prefix}{"/" if prefix else ""}{metric_name}', *partial_row])


def create_video(video_path, anomaly_window, bbox_window_size, gt_right_path, gt_left_path):
    # Create raw results
    frames, raw_results = get_raw_results(video_path)

    # Extraction
    extracted_results = extract_all_results(raw_results)

    # Maximal confidence
    max_conf_predictions = deepcopy(extracted_results)
    for i in range(len(max_conf_predictions)):
        max_conf_predictions[i] = extract_maximal_confidence(max_conf_predictions[i])

    # Fill in missing classes
    only_classes = extract_only_classes(max_conf_predictions)
    filled_classes = fill_missing_labels(only_classes)
    filled_predictions = deepcopy(max_conf_predictions)
    for i in range(len(filled_predictions)):
        if filled_predictions[i][RIGHT] is not None:
            filled_predictions[i][RIGHT][CLASS] = filled_classes[RIGHT][i]
        else:
            filled_predictions[i][RIGHT] = {CLASS: filled_classes[RIGHT][i],
                                            CONF: filled_predictions[i - 1][RIGHT][CONF] if i > 0 else 0,
                                            XYXY: None}
        if filled_predictions[i][LEFT] is not None:
            filled_predictions[i][LEFT][CLASS] = filled_classes[LEFT][i]
        else:
            filled_predictions[i][LEFT] = {CLASS: filled_classes[LEFT][i],
                                           CONF: filled_predictions[i - 1][LEFT][CONF] if i > 0 else 0,
                                           XYXY: None}

    # Fill in missing bounding boxes
    filled_predictions = fill_missing_bounding_boxes(filled_predictions)

    # Smooth labels
    smoothed_classes = smooth_labels(filled_classes, anomaly_window=anomaly_window)
    smoothed_predictions = deepcopy(filled_predictions)
    for i in range(len(smoothed_predictions)):
        smoothed_predictions[i][RIGHT][CLASS] = smoothed_classes[RIGHT][i]
        smoothed_predictions[i][LEFT][CLASS] = smoothed_classes[LEFT][i]

    # Smooth bounding boxes
    smoothed_predictions = smooth_bboxes(smoothed_predictions, window_size=bbox_window_size)

    # Video creation
    create_video_util(video_path, frames, smoothed_predictions)

    if gt_right_path is not None:  # Both are not None
        create_metrics_file(video_path,
                            [max_conf_predictions, filled_predictions, smoothed_predictions],
                            ['max-conf', 'filled', 'smoothed'], gt_right_path, gt_left_path)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=False, help='video path')
    parser.add_argument('--left', type=str, required=False, help='left hand labels')
    parser.add_argument('--right', type=str, required=False, help='right hand labels')
    parser.add_argument('--label_window', type=int, default=3, help='label smoothing window size (>=0)')
    parser.add_argument('--bbox_window', type=int, default=5, help='bounding box smoothing window size (>=0)')
    args = parser.parse_args()
    if args.left is not None and args.right is None:
        parser.error("--right is required if --left is specified")
    if args.right is not None and args.left is None:
        parser.error("--left is required if --right is specified")
    if args.label_window < 0:
        parser.error(f"--label_window must be non-negative, but got {args.label_window}")
    if args.bbox_window < 0:
        parser.error(f"--bbox_window must be non-negative, but got {args.bbox_window}")
    if (args.video is not None) and (not os.path.isfile(args.video)):
        parser.error(f"--video is invalid: no such file or directory {args.video}")
    if (args.right is not None) and (not os.path.isfile(args.right)):
        parser.error(f"--right is invalid: no such file or directory {args.right}")
    if (args.left is not None) and (not os.path.isfile(args.left)):
        parser.error(f"--left is invalid: no such file or directory {args.left}")
    return args


if __name__ == '__main__':
    args = create_parser()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/large_SGD/weights/best.pt',
                           force_reload=True)

    create_video(args.video, args.label_window, args.bbox_window, args.right, args.left)
