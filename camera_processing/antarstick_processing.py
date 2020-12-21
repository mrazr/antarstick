import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2 as cv
import numpy as np
from disjoint_set import DisjointSet

from camera import Camera
from camera_processing.stick_detection import boxes_intersection, line_upright_bbox, detect_sticks, \
    merge_lines_with_same_orientation, length_of_line, fit_into_length, fit_into_length_from_bottom, line_vector, \
    find_sticks
from stick import Stick

STICK_HOG_SVC_FILE = Path(sys.argv[0]).parent / 'camera_processing/stick_hog_desc'

hog_desc = cv.HOGDescriptor()
hog_desc.load(str(STICK_HOG_SVC_FILE))


@dataclass
class StickDetection:
    model: Optional[Stick] = None
    valid: bool = False
    old_stick: Optional[Stick] = None
    old_orientation: float = -1.0
    old_corrected_endpoint: Optional[np.ndarray] = None

    new_stick: Optional[Stick] = None
    new_orientation: float = -1.0
    new_corrected_endpoint: Optional[np.ndarray] = None

    stick_to_use: Optional[Stick] = None

    orientation_diff: float = -1.0
    top_diff: np.ndarray = np.array([np.nan, np.nan])
    bottom_diff: np.ndarray = np.array([np.nan, np.nan])
    bbox: np.ndarray = np.array([])
    roi: np.ndarray = np.array([])
    likely_snow_point: np.ndarray = np.array([0, 0], np.int32)
    image: str = ''

    # query fields
    top_is_same: bool = False
    bottom_is_same: bool = False
    orientation_is_same: bool = False
    new_lies_on_old: bool = False
    old_corrected_matches_old: bool = False
    new_corrected_matches_new: bool = False
    in_frame: bool = False
    top_out_of_frame: bool = False
    bottom_out_of_frame: bool = False
    needs_to_measure: bool = False

    def reset(self):
        self.model = None
        self.valid = False
        self.old_stick = None
        self.old_orientation = -1.0
        self.old_corrected_endpoint = None
        self.image = ''

        self.new_stick = None
        self.new_orientation = -1.0
        self.new_corrected_endpoint = None

        self.orientation_diff = -1.0
        self.top_diff = np.array([np.nan, np.nan])
        self.bottom_diff = np.array([np.nan, np.nan])
        self.bbox: np.ndarray = np.array([])
        self.likely_snow_point = np.array([0, 0], np.int32)

        # query fields
        self.top_is_same = False
        self.bottom_is_same = False
        self.orientation_is_same = False
        self.new_lies_on_old = False
        self.old_corrected_matches_old = False
        self.new_corrected_matches_new = False
        self.in_frame = False
        self.top_out_of_frame: bool = False
        self.bottom_out_of_frame: bool = False
        self.needs_to_measure: bool = True

    def __hash__(self):
        return self.old_stick.__hash__()

    def __eq__(self, other):
        return self.old_stick == other.old_stick


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


def iou(box1: List[int], box2: List[int]) -> float:
    # area1 = abs((box1[0] - box1[2]) * (box1[1] - box1[3]))
    # area2 = abs((box2[0] - box2[2]) * (box2[1] - box2[3]))

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
    #print(f'boxes is {len(boxes)} items')

    processed = [False for _ in boxes]

    for bid in range(9):
        #print(f'b {bid} contains {len(boxeses[bid])} items')
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
    #processed = [False for _ in boxes]

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

    #for boxid in range(len(boxes)):
    #    rep = s.find(boxid)
    #    group = s.get_set(rep)
    #    if len(group) < 3 or processed[rep]:
    #        continue
    #    processed[rep] = True

    #    left = 9000
    #    right = -9000
    #    top = 9000
    #    bottom = -9000

    #    for box_id in group:
    #        box = boxes[box_id]
    #        if box[0] < left:
    #            left = box[0]
    #        if box[1] < top:
    #            top = box[1]
    #        if box[0] + box[2] > right:
    #            right = box[0] + box[2]
    #        if box[1] + box[3] > bottom:
    #            bottom = box[1] + box[3]
    #    final_rects.append(np.array([[left, top], [right, bottom]]))

    return final_rects


def match_new_sticks_with_old(old_sticks: List[Stick], new_sticks: List[np.ndarray]) -> List[Tuple[Stick, List[int]]]:
    N = len(new_sticks)
    possible_matches = []
    bboxes = list(map(lambda s: line_upright_bbox(s.line(), 5000, 5000, 35), old_sticks))
    for old_stick in old_sticks:
        # We're going to be matching `old` stick with every new stick from `new_sticks`
        # Because, mostly, big stick misalignments are caused by camera rotation around the stake the camera is fixed on,
        # we have to work with the model that each stick lies on a circle with the center being at the camera position.
        # And then, we assume that after camera rotation, each stick moves by the same angular distance, which results in
        # euclidean distance proportional to the distance from the camera.
        dist_from_camera = np.linalg.norm(old_stick.bottom - np.array([1000, 1500]))
        for new_i, new in enumerate(new_sticks):
            vector = 0.5 * (new[0] + new[1]) - 0.5 * (old_stick.top + old_stick.bottom)
            matching = [(None, n, 0, np.array([0, 0])) for n in new_sticks]
            matching[new_i] = (old_stick, new, 999999, vector) # `old` and `new` are trivially matched and hence their distance is 0.0
            for l, old_ in enumerate(old_sticks):
                if old_ == old_stick:
                    continue
                dist_from_camera2 = np.linalg.norm(old_.bottom - np.array([1000, 1500]))
                factor = dist_from_camera2 / dist_from_camera
                # Correction due to distance from camera
                corrected_vector = np.round(factor * vector).astype(np.int32)
                #print(f'corrected vector is {corrected_vector}')
                #offsetted = old_.line() + corrected_vector
                #old_box = line_upright_bbox(offsetted, 5000, 5000, 35)
                old_box = np.round(bboxes[l] + corrected_vector).astype(np.int32)
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
                                    matching[ik] = (old_, candidate, area, corrected_vector)
                                    last_match = ik
                            else:
                                matching[ik] = (old_, candidate, area, corrected_vector)
                                last_match = ik
            matching = list(filter(lambda mm: mm[0] is not None, matching))
            possible_matches.append((matching, vector * (200.0 / dist_from_camera), 0 * len(matching) + 1 * sum(map(lambda match_: match_[2], matching))))
            #print(f'len match is {len(matching)} with sum = {sum(map(lambda mat_: mat_[2], matching))}')
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


    # print(f'match took {time() - ma_start} secs')
    # if matches is None or len(matches) < 0.3 * len(sticks):
    #    continue
    ##print(f'took {time() - start} secs')
    # dr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # for match in matches:
    #    if match[0] is None:
    #        continue
    #    stick = match[0]
    #    vec = match[3]
    #    new_box = match[1]
    #    color = [randint(0, 255), randint(0, 255), randint(0, 255)]

    #    cv.rectangle(dr, (new_box[0, 0], new_box[0, 1] - stick_box[1]), (new_box[1, 0], new_box[1, 1] - stick_box[1]), color, 3)
    #    cv.line(dr, (int(stick.top[0]), int(stick.top[1] - stick_box[1])), (int(stick.bottom[0]), int(stick.bottom[1] - stick_box[1])), color, 2)
    # cv.imshow('dr', dr)
    # if cv.waitKey(0) == ord('q'):
    #    break


def analyze_photos_with_stick_tracking(images: List[str], folder: Path, sticks: List[Stick], standard_size: Tuple[int, int]) -> Measurement:
    update_step = 100
    win_size = (48, 96)
    stick_box = get_stick_area(sticks)
    #stick_box = [0, 0, 0, 0]
    quality_sticks = len(list(filter(lambda s: s.determines_quality, sticks)))
    measurement = Measurement()
    measurement.reason = Reason.Update
    for img_name in images[:update_step]:
        sticks_: List[Stick] = list(map(lambda s: s.copy(), sticks))
        found_sticks = {stick: False for stick in sticks}
        for s in sticks_:
            s.view = img_name
            s.is_visible = False
            s.set_snow_height_px(0)
        gray = cv.pyrDown(cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE))
        #gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        #nighttime = is_night(bgr)
        daytime = is_day(gray)
        gray_crop = cv.GaussianBlur(gray, (3, 3), sigmaX=1.0)[stick_box[0, 1]:stick_box[1, 1], :]
        #gray_crop = cv.copyMakeBorder(gray_crop, 0, 0, 24, 24, cv.BORDER_REPLICATE)
        gray_h = cv.resize(gray_crop, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
        snowy = is_snow(gray)
        if not snowy and daytime:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': True,
                                                  'state': 1,
                                                  'is_snowy': False}
            continue
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
                    lines = merge_lines_with_same_orientation(detect_sticks(cv.pyrDown(roi), False, False), max_gap=-1)
                    lines = list(map(lambda l: 2 * l, lines))
                else:
                    lines = merge_lines_with_same_orientation(detect_sticks(roi, False, False), max_gap=-1)
                line: np.ndarray = max(lines, key=lambda l: length_of_line(l), default=None)
                if line is not None:
                    stick.is_visible = True
                    detected_sticks += 1
                    line_ = fit_into_length(line, stick.length_px)
                    stick.set_top(line_[0] + box[0])
                    stick.set_bottom(line_[1] + box[0])
                    stick.set_snow_height_px(stick.length_px - length_of_line(line))
                    #line = np.round((1.0 / scale) * line).astype(np.int32) + box[0]
                    #snow_height = np.linalg.norm(line[1] - stick.bottom)
                    #stick.is_visible = True
                    #detected_sticks += 1
                    ## line_ = fit_into_length(line, stick.length_px)
                    #line_ = fit_into_length_from_bottom(line, stick.length_px - snow_height)
                    #line_ = fit_into_length(line_, stick.length_px)
                    #stick.set_top(line_[0])
                    #stick.set_bottom(line_[1])
                    #stick.set_snow_height_px(snow_height)  # stick.length_px - length_of_line(line))
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
    if len(measurement.remaining_photos) == 0:
        measurement.reason = Reason.FinishedQueue
    return measurement


def analyze_photos(images: List[str], folder: Path, sticks: List[Stick], standard_size: Tuple[int, int],  update_step: int = 100) -> Measurement:
    measurement = Measurement()
    measurement.reason = Reason.Update
    quality_sticks = len(list(filter(lambda s: s.determines_quality, sticks)))
    for img_name in images[:update_step]:
        gray = cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE)
        gray_h = cv.pyrDown(gray)
        sticks_: List[Stick] = list(map(lambda s: s.copy(), sticks))
        print(img_name)
        for s in sticks_:
            s.view = img_name
            s.is_visible = False
            s.set_snow_height_px(0)
        if gray.shape[0] != standard_size[1] or gray.shape[1] != standard_size[0]:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': True,
                                                  'state': -1,
                                                  'is_snowy': False}
            continue
        daytime = is_day(gray_h)
        snowy = is_snow(gray_h)
        if not snowy and daytime:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': True,
                                                  'state': 1,
                                                  'is_snowy': False}
            continue
        detected_sticks = 0
        for stick in sticks_:
            box = line_upright_bbox(2 * stick.line(), gray.shape[1], gray.shape[0], width=45)
            roi = linear_stretch(cv.GaussianBlur(gray[box[0, 1]:box[1, 1], box[0, 0]: box[1, 0]], (3, 3), sigmaX=1.0))
            if stick.width < 5:
                scale = 2.0
                h = 21
            elif stick.width > 9:
                scale = 0.5
                roi = cv.pyrDown(cv.pyrDown(roi))
                #box = 2 * box
                h = 9
            else:
                scale = 1.0
                roi = cv.pyrDown(roi)
                h = 9
                #h = 21
                #box = np.round(0.5 * box).astype(np.int32)
            box = np.round(0.5 * box).astype(np.int32)

            lines = detect_sticks(roi, False, False, h=h)
            lines = merge_lines_with_same_orientation(lines, max_gap=-1)
            line: np.ndarray = max(lines, key=lambda l: length_of_line(l), default=None)
            if line is not None:
                line = np.round((1.0 / scale) * line).astype(np.int32) + box[0]
                snow_height = np.linalg.norm(line[1] - stick.bottom)
                stick.is_visible = True
                detected_sticks += 1
                #line_ = fit_into_length(line, stick.length_px)
                line_ = fit_into_length_from_bottom(line, stick.length_px - snow_height)
                line_ = fit_into_length(line_, stick.length_px)
                stick.set_top(line_[0])
                stick.set_bottom(line_[1])
                stick.set_snow_height_px(snow_height)#stick.length_px - length_of_line(line))
        measurement.measurements[img_name] = {'sticks': sticks_,
                                              'image_quality': min(detected_sticks / quality_sticks, 1.0),
                                              'is_day': daytime,
                                              'state': 1,
                                              'is_snowy': True}

    measurement.remaining_photos = images[update_step:]
    if len(measurement.remaining_photos) == 0:
        measurement.reason = Reason.FinishedQueue
    return measurement


def linear_stretch(gray: np.ndarray) -> np.ndarray:
    return cv.convertScaleAbs((gray - np.min(gray)) * (1.0 / (np.max(gray) - np.min(gray) + 0.0001)), alpha=255)


def analyze_photos_ip(images: List[str], folder: Path, sticks: List[Stick], standard_size: Tuple[int, int], update_step: int = 20) -> Measurement:
    measurement = Measurement()
    measurement.reason = Reason.Update
    quality_sticks = len(list(filter(lambda s: s.determines_quality, sticks)))
    stick_box = 2 * get_stick_area(sticks, standard_size[0])
    for img_name in images[:update_step]:
        sticks_: List[Stick] = list(map(lambda s: s.copy(), sticks))
        for st in sticks_:
            st.is_visible = False
            st.set_snow_height_px(-1)
            st.view = img_name
        gray = cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE)
        if gray.shape[0] != standard_size[1] or gray.shape[1] != standard_size[0]:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': True,
                                                  'state': -1,
                                                  'is_snowy': False}
            continue
        daytime = is_day(gray)
        snowy = is_snow(gray)
        if not snowy and daytime:
            measurement.measurements[img_name] = {'sticks': sticks_,
                                                  'image_quality': 1.0,
                                                  'is_day': True,
                                                  'state': 1,
                                                  'is_snowy': False}
            continue
        gray = gray[stick_box[0, 1]: stick_box[1, 1],
               stick_box[0, 0]: stick_box[1, 0]]
        gray = cv.GaussianBlur(gray, (3, 3), sigmaX=1.0)
        gray_h = cv.pyrDown(gray)

        gray_q = cv.pyrDown(gray_h)

        final_lines = None
        lines = detect_sticks(gray, False, False, h=19)
        lines_h = detect_sticks(gray_h, False, False)
        lines_q = detect_sticks(gray_q, False, False)

        if len(lines) > 0:
            final_lines = np.round(0.5 * lines).astype(np.int32)
        if len(lines_h) > 0:
            final_lines = lines_h if final_lines is None else np.vstack((final_lines, lines_h))
        if len(lines_q) > 0:
            lines_q = 2 * lines_q
            final_lines = lines_q if final_lines is None else np.vstack((final_lines, lines_q))
        detected_sticks = 0
        if final_lines is not None:
            box = np.round(0.5 * stick_box).astype(np.int32)
            final_lines = merge_lines_with_same_orientation(final_lines, max_gap=25)
            matches, offset_vector = match_new_sticks_with_old(sticks_, final_lines + box[0])
            if matches is not None:
                for match in matches:
                    old_stick: Stick = match[0]
                    new_stick: np.ndarray = match[1]
                    snow_height = old_stick.length_px - length_of_line(new_stick)
                    line_ = fit_into_length_from_bottom(new_stick, old_stick.length_px - snow_height)
                    line_ = fit_into_length(line_, old_stick.length_px)
                    old_stick.set_top(line_[0])
                    old_stick.set_bottom(line_[1])
                    old_stick.set_snow_height_px(int(round(snow_height)))
                    old_stick.is_visible = True
                    if old_stick.determines_quality:
                        detected_sticks += 1
        measurement.measurements[img_name] = {'sticks': sticks_,
                                              'image_quality': min(detected_sticks / quality_sticks, 1.0),
                                              'is_day': daytime,
                                              'state': 1,
                                              'is_snowy': True}
    if len(images) <= update_step:
        measurement.reason = Reason.FinishedQueue
    measurement.remaining_photos = images[update_step:]
    return measurement


def is_day(gray: np.ndarray) -> bool:
    return np.mean(gray[:int(0.07 * gray.shape[0]), :]) > 40.0


def is_snow(gray: np.ndarray) -> bool:
    return np.mean(gray[:int(0.13 * gray.shape[0]), :]) - np.mean(gray[int(0.5 * gray.shape[0]):int(0.8 * gray.shape[0]), :]) < 25


def handle_big_camera_movement(img: np.ndarray, half: np.ndarray, quart: np.ndarray, stick_detections: List[
    StickDetection]) -> Optional[List[StickDetection]]:
    new_sticks = find_sticks(img, img, True)
    new_sticks = list(map(lambda s: s[0], filter(lambda s: s[1], new_sticks)))
    if len(new_sticks) == 0:
        return None, None
    #matches = match_detections_to_sticks(stick_detections, new_sticks)
    matches, vector = match_new_sticks_with_old(stick_detections, new_sticks)
    if matches is None or len(matches) < 0.5 * len(stick_detections):
        return None, None
    for d in stick_detections:
        d.valid = False
    dd = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for ns in new_sticks:
        box = line_upright_bbox(ns, 5000, 5000, 35)
        cv.rectangle(dd, (box[0, 0], box[0, 1]), (box[1, 0], box[1, 1]), [0, 255, 255], 4)
    for match in matches:
        if match[0] is None:
            continue
        #det: StickDetection = match[0][0]
        det: StickDetection = match[0]
        vec = match[3]
        n_stick = det.old_stick.copy()
        new_line = match[1]

        nbox = line_upright_bbox(new_line, 5000, 5000, 35)

        cv.rectangle(dd, (nbox[0, 0], nbox[0, 1]), (nbox[1, 0], nbox[1, 1]), [0, 255, 0], 3)

        vec_to_old = det.old_stick.top - np.array([1000, 1500])
        #vec_to_old = vec_to_old / np.linalg.norm(vec_to_old)
        #smallest_length = det.old_stick.length_px * np.dot(np.array([0, -1.0]), vec_to_old)

        vec_to_new = new_line[0] - np.array([1000, 1500])
        factor = vec_to_old[1] / (vec_to_new[1] + 0.000001)
        new_length = factor * det.old_stick.length_px
        #vec_to_new = vec_to_new / np.linalg.norm(vec_to_new)
        #new_length = smallest_length / np.dot(np.array([0, -1.0]), vec_to_new)

        if new_line[0, 0] < 10 or new_line[0, 0] > img.shape[1] - 10:
            new_line = fit_into_length_from_bottom(new_line, det.old_stick.length_px)
        elif new_line[1, 0] < 10 or new_line[1, 0] > img.shape[1] - 10:
            new_line = fit_into_length(new_line, det.old_stick.length_px)
        else:
            new_line = fit_into_length(new_line, det.old_stick.length_px)
        n_stick.set_top(new_line[0])
        n_stick.set_bottom(new_line[1])

        obox = line_upright_bbox(det.old_stick.line(), 5000, 5000, 35)
        obox_t = obox + vec
        cv.rectangle(dd, (obox[0, 0], obox[0, 1]), (obox[1, 0], obox[1, 1]), [255, 0, 0], 2)
        cv.rectangle(dd, (obox_t[0, 0], obox_t[0, 1]), (obox_t[1, 0], obox_t[1, 1]), [255, 0, 255], 2)

        det.new_stick = n_stick
        det.stick_to_use = det.new_stick
        det.bottom_diff = det.stick_to_use.bottom - det.old_stick.bottom
        det.top_diff = det.stick_to_use.top - det.old_stick.top

        det.valid = True

    #cv.imshow('matches', dd)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return stick_detections, dd


def match_new_sticks_with_old_(old_sticks: List[StickDetection], new_sticks: List[np.ndarray]) -> List[Tuple[
    StickDetection, np.ndarray]]:
    N = len(new_sticks)
    possible_matches = []
    for old in old_sticks:
        # We're going to be matching `old` stick with every new stick from `new_sticks`
        old_stick = old.old_stick
        # Because, mostly, big stick misalignments are caused by camera rotation around the stake the camera is fixed on,
        # we have to work with the model that each stick lies on a circle with the center being at the camera position.
        # And then, we assume that after camera rotation, each stick moves by the same angular distance, which results in
        # euclidean distance proportional to the distance from the camera.
        dist_from_camera = np.linalg.norm(old_stick.bottom - np.array([1000, 1500]))
        for new_i, new in enumerate(new_sticks):
            vector = new[0] - old_stick.top
            matching = [(None, n, 0, np.array([0, 0])) for n in new_sticks]
            matching[new_i] = (old, new, 999999, vector) # `old` and `new` are trivially matched and hence their distance is 0.0
            for old_ in old_sticks:
                if old_ == old:
                    continue
                dist_from_camera2 = np.linalg.norm(old_.old_stick.bottom - np.array([1000, 1500]))
                factor = dist_from_camera2 / dist_from_camera
                # Correction due to distance from camera
                corrected_vector = np.round(factor * vector).astype(np.int32)
                offsetted = old_.old_stick.line() + corrected_vector
                old_box = line_upright_bbox(offsetted, 5000, 5000, 35)
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
                                    matching[ik] = (old_, candidate, area, corrected_vector)
                                    last_match = ik
                            else:
                                matching[ik] = (old_, candidate, area, corrected_vector)
                                last_match = ik
            matching = list(filter(lambda mm: mm[0] is not None, matching))
            possible_matches.append((matching, vector, 0 * len(matching) + 1 * sum(map(lambda match_: match_[2], matching))))
    to_return = max(possible_matches, key=lambda m: m[2], default=(None, None))[:2]
    return to_return