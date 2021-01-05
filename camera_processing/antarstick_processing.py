import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2 as cv
import joblib
import numpy as np
from disjoint_set import DisjointSet
from sklearn.pipeline import Pipeline

from camera import Camera, WeatherCondition
from camera_processing.stick_detection import boxes_intersection, line_upright_bbox, detect_lines, \
    merge_lines_with_same_orientation, length_of_line, fit_into_length, line_vector
from stick import Stick

STICK_HOG_SVC_FILE = Path(sys.argv[0]).parent / 'camera_processing/stick_hog_desc'
SNOW_CLASSIFIER_FILE = Path(sys.argv[0]).parent / 'camera_processing/snow_classifier.joblib'

hog_desc = cv.HOGDescriptor()
hog_desc.load(str(STICK_HOG_SVC_FILE))

snow_classifier: Pipeline = joblib.load(SNOW_CLASSIFIER_FILE)


class Reason(IntEnum):
    FinishedQueue = 0,
    SticksMoved = 1,
    Update = 2


@dataclass
class Measurement:
    camera: Optional[Camera] = None
    reason: Reason = Reason.FinishedQueue
    measurements: Dict[str, Dict[str, List[Stick]]] = field(default_factory=dict)
    last_valid_sticks: List[Stick] = field(default_factory=list)
    last_img: str = ''
    current_img: str = ''
    remaining_photos: Optional[List[str]] = None

    sticks_to_confirm: Optional[Stick] = None
    im: Optional[np.ndarray] = None
    snow_pic_count: int = 0


def iou(box1: List[int], box2: List[int]) -> float:
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    left_box = box1 if box1[0] < box2[0] else box2
    right_box = box2 if left_box == box1 else box1

    top_box = box1 if box1[1] < box2[1] else box2
    bottom_box = box2 if top_box == box1 else box1

    hor_length = left_box[0] + left_box[2] - right_box[0]
    ver_length = top_box[1] + top_box[3] - bottom_box[1]

    if hor_length <= 0 or ver_length <= 0:
        return 0.0
    intersection = hor_length * ver_length
    return intersection / (area1 + area2 - intersection + 0.0001)


def merge_det_boxes(boxes: List[List[int]], img_width: int = 2000) -> List[np.ndarray]:
    s = DisjointSet()

    step = int(round(0.1 * img_width))

    boxeses = [[] for _ in range(10)]

    for boxid, box in enumerate(boxes):
        idd = int(box[0] / step)
        boxeses[idd].append(boxid)

    processed = [False for _ in boxes]

    for bid in range(9):
        for boxid in boxeses[bid]:
            box = boxes[boxid]
            if processed[boxid]:
                continue
            processed[boxid] = True
            for o_boxid in boxeses[bid]:
                if o_boxid == boxid:
                    continue
                other_box = boxes[o_boxid]
                boxes_iou = iou(box, other_box)

                if boxes_iou > 0.5:
                    s.union(boxid, o_boxid)
                    processed[o_boxid] = True
            for o_boxid in boxeses[bid+1]:
                if o_boxid == boxid:
                    continue
                other_box = boxes[o_boxid]
                boxes_iou = iou(box, other_box)

                if boxes_iou > 0.5:
                    s.union(boxid, o_boxid)
                    processed[o_boxid] = True
    final_rects = []

    for group in s.itersets():
        if len(group) < 3:
            continue
        left = 9000
        right = -9000
        top = 9000
        bottom = -9000

        for box_id in group:
            box = boxes[box_id]
            if box[0] < left:
                left = box[0]
            if box[1] < top:
                top = box[1]
            if box[0] + box[2] > right:
                right = box[0] + box[2]
            if box[1] + box[3] > bottom:
                bottom = box[1] + box[3]
        final_rects.append(np.array([[left, top], [right, bottom]]))

    return final_rects


def match_new_sticks_with_old(old_sticks: List[Stick], new_sticks: List[np.ndarray]) -> List[Tuple[Stick, List[int]]]:
    possible_matches = []
    bboxes = list(map(lambda s: line_upright_bbox(s.line(), 5000, 5000, 35), old_sticks))
    for old_stick in old_sticks:
        dist_from_camera = np.linalg.norm(old_stick.bottom - np.array([1000, 1500]))
        for new_i, new in enumerate(new_sticks):
            vector = 0.5 * (new[0] + new[1]) - 0.5 * (old_stick.top + old_stick.bottom)
            matching = [(None, n, 0, np.array([0, 0])) for n in new_sticks]
            matching[new_i] = (old_stick, new, 999999, vector) # `old` and `new` are trivially matched and hence their distance is 0.0
            for l, old_ in enumerate(old_sticks):
                if old_ == old_stick:
                    continue
                old_box = np.round(bboxes[l] + vector).astype(np.int32)
                last_match = -1
                for ik, candidate in enumerate(new_sticks):
                    new_box = line_upright_bbox(candidate, 5000, 5000, 35)
                    inters = boxes_intersection(old_box, new_box)
                    if inters is not None:
                        area = (inters[1, 0] - inters[0, 0]) * (inters[1, 1] - inters[0, 1])
                        if matching[ik][0] is None or matching[ik][2] < area:
                            if last_match >= 0:
                                conflict_match = matching[last_match]
                                if conflict_match[2] < area:
                                    matching[last_match] = (None, conflict_match[1], 0)
                                    matching[ik] = (old_, candidate, area, vector)
                                    last_match = ik
                            else:
                                matching[ik] = (old_, candidate, area, vector)
                                last_match = ik
            matching = list(filter(lambda mm: mm[0] is not None, matching))
            possible_matches.append((matching, vector * (200.0 / dist_from_camera), 0 * len(matching) + 1 * sum(map(lambda match_: match_[2], matching))))
    to_return = max(possible_matches, key=lambda m: m[2], default=(None, None))[:2]
    return to_return


def get_stick_area(sticks: List[Stick], width: int = -1) -> np.ndarray: #List[int]:
    left = 0
    right = width
    top = 9000
    bottom = -1

    for stick in sticks:
        left = min(left, min(stick.top[0], stick.bottom[0]))
        right = max(right, max(stick.top[0], stick.bottom[0]))
        top = min(top, min(stick.top[1], stick.bottom[1]))
        bottom = max(bottom, max(stick.top[1], stick.bottom[1]))
    return np.array([[max(left - 100, 0), top - 100], [right + 100, bottom + 100]])


def analyze_photos_with_stick_tracking(images: List[Tuple[str, WeatherCondition]], folder: Path, sticks: List[Stick], standard_size: Tuple[int, int], process_nighttime: bool = True, snow_pic_count: int = 0) -> Measurement:
    update_step = 100
    win_size = (48, 96)
    stick_box = get_stick_area(sticks)
    quality_sticks = len(list(filter(lambda s: s.determines_quality, sticks)))

    measurement = Measurement()
    measurement.reason = Reason.Update
    measurement.snow_pic_count = snow_pic_count

    for img_name, condition in images[:update_step]:
        sticks_: List[Stick] = list(map(lambda s: s.copy(), sticks))
        found_sticks = {stick: False for stick in sticks}
        for s in sticks_:
            s.view = img_name
            s.is_visible = False
            s.set_snow_height_px(0)
        if condition == WeatherCondition.Snow:
            gray = cv.pyrDown(cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE))
            snowy = True
        else:
            bgr = cv.pyrDown(cv.imread(str(folder / img_name)))
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
            bgr = cv.pyrDown(bgr)
            avg_color = np.mean(bgr[370:720, :], axis=(0, 1))
            snowy = snow_classifier.predict([avg_color])[0] > 0
        daytime = is_day(gray)

        if snowy and daytime:
            measurement.snow_pic_count = min(measurement.snow_pic_count + 3, 5)
        elif daytime:
            measurement.snow_pic_count = max(measurement.snow_pic_count - 1, 0)

        if not daytime and not process_nighttime:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': False,
                                                  'state': -1,
                                                  'is_snowy': False}
            continue

        if not snowy and measurement.snow_pic_count < 3 and daytime:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': True,
                                                  'state': 1,
                                                  'is_snowy': False}
            continue

        if not daytime and measurement.snow_pic_count < 3:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': False,
                                                  'state': 1,
                                                  'is_snowy': False}
            continue

        gray_crop = cv.GaussianBlur(gray, (3, 3), sigmaX=1.0)[stick_box[0, 1]:stick_box[1, 1], :]
        gray_h = cv.resize(gray_crop, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        found, weights = hog_desc.detect(gray_crop)
        found_, weights_ = hog_desc.detect(gray_h)
        if len(weights) > 0:
            rects = list(map(lambda f: [max(int(f[0]), 0), int(f[1] + 1 * stick_box[0, 1]), int(win_size[0]), int(win_size[1])], found))
        else:
            rects = []
            weights = []
        if len(weights_) > 0:
            rects_ = list(map(lambda f: [int(2 * f[0]) + 24, int(2 * f[1] + 1 * stick_box[0, 1]) + 48, int(win_size[0]), int(win_size[1])], found_))
        else:
            rects_ = []
            weights_ = []
        rects.extend(rects_)
        if len(weights) == 0:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 0.0,
                                                  'is_day': daytime,
                                                  'state': 1,
                                                  'is_snowy': True}
            continue
        grouped = merge_det_boxes(rects, gray.shape[1])
        matches, vector = match_new_sticks_with_old(sticks_, grouped)
        detected_sticks = 0
        if matches is not None:
            if condition != WeatherCondition.Snow:
                bgr = cv.pyrUp(bgr)
            for match in matches:
                if match[0] is None:
                    continue
                stick = match[0]
                found_sticks[stick] = True
                box = match[1]
                height_diff = max((stick.bottom[1] - stick.top[1]) - ((box[1, 1] - box[0, 1]) - win_size[1] // 2), 0)
                if height_diff > 0:
                    height_diff += 0.5 * (box[1, 1] - box[0, 1]) + win_size[1] // 2
                    vec = line_vector(stick.line())
                    vec /= vec[1]
                    vec *= height_diff
                    line__ = np.array([0.5 * (box[0] + box[1]), 0.5 * (box[0] + box[1])])
                    line__[0] -= vec
                    line__[1] += vec
                    line__ = np.round(line__).astype(np.int32)
                    box = line_upright_bbox(line__, gray.shape[1], gray.shape[0])

                roi = linear_stretch(gray[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0]])
                if stick.width > 9:
                    lines = merge_lines_with_same_orientation(detect_lines(cv.pyrDown(roi)), max_gap=-1)
                    lines = list(map(lambda l: 2 * l, lines))
                else:
                    lines = merge_lines_with_same_orientation(detect_lines(roi), max_gap=-1)
                line: np.ndarray = max(lines, key=lambda l: length_of_line(l), default=None)
                if line is not None:
                    stick.is_visible = True
                    detected_sticks += 1
                    line_ = fit_into_length(line, stick.length_px)
                    stick.set_top(line_[0] + box[0])
                    stick.set_bottom(line_[1] + box[0])
                    stick.set_snow_height_px(max(stick.length_px - length_of_line(line), 0))
                    if condition != WeatherCondition.Snow:
                        bgr_roi = bgr[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0]]
                        mean_color = np.mean(bgr_roi[int(np.round(0.7 * bgr_roi.shape[0])):, :], axis=(0, 1))
                        if daytime and not (snow_classifier.predict([mean_color])[0] > 0):
                            stick.set_snow_height_px(0)
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': min(detected_sticks / quality_sticks, 1.0),
                                                  'is_day': daytime,
                                                  'state': 1,
                                                  'is_snowy': True}
        else:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 0.0,
                                                  'is_day': daytime,
                                                  'state': -1,
                                                  'is_snowy': False}

    measurement.remaining_photos = images[update_step:]
    return measurement


def linear_stretch(gray: np.ndarray) -> np.ndarray:
    return cv.convertScaleAbs((gray - np.min(gray)) * (1.0 / (np.max(gray) - np.min(gray) + 0.0001)), alpha=255)


def is_day(gray: np.ndarray) -> bool:
    return np.mean(gray[:int(0.07 * gray.shape[0]), :]) > 40.0


