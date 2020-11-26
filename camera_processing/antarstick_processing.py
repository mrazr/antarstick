#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from functools import reduce
from itertools import tee
from os import scandir
from pathlib import Path
from queue import Queue
from random import randint
from typing import Dict, List, Optional, Tuple, Union

import cv2 as cv
import joblib
import numpy as np
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform
from sklearn.pipeline import Pipeline

from camera import Camera
from stick import Stick

STICK_PIPELINE_FILE = Path(sys.argv[0]).parent / 'camera_processing/stick_verification_pipeline4.joblib'

try:
    with open(STICK_PIPELINE_FILE, 'rb') as f:
        stick_pipeline: Pipeline = joblib.load(f)
except FileNotFoundError:
    print(f'Could not load file {STICK_PIPELINE_FILE}')  # TODO show error message dialog
    exit(-1)

clahe = cv.createCLAHE()
clahe.setTilesGridSize((8, 8))

FEATURES = ['DIFF_MEAN', 'DIFF_STD', 'LEFT_STD', 'RIGHT_STD', 'LR_MEAN_DIFF', 'I_STD', 'DIFF_FROM_ANGLE', 'LEFT_PER',
            'RIGHT_PER', 'LEFT_RIGHT_DIFF']


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
        self.needs_to_measure: bool = False

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


def show_imgs_(images: List[np.ndarray], names: List[str]):
    for image, name in zip(images, names):
        cv.imshow(name, image)


def is_non_snow(hsv_img: np.ndarray) -> bool:
    roi_x = int(0.2 * hsv_img.shape[1])
    roi_y = int(0.4 * hsv_img.shape[0])
    roi_w = int(0.6 * hsv_img.shape[1])
    roi_h = int(0.4 * hsv_img.shape[0])

    return np.mean(hsv_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, 2]) < 100


dbd_se1 = np.ones((1, 1), dtype=np.int8)
dbd_se1 = np.pad(dbd_se1, ((0, 0), (2, 2)), 'constant', constant_values=-1)
dbd_se1 = np.pad(dbd_se1, ((0, 0), (3, 3)), 'constant', constant_values=0)

dbd_se3 = np.ones((1, 3), dtype=np.int8)
dbd_se3 = np.pad(dbd_se3, ((0, 0), (3, 3)), 'constant', constant_values=-1)
dbd_se3 = np.pad(dbd_se3, ((0, 0), (3, 3)), 'constant', constant_values=0)

dbd_se5 = np.ones((1, 5), dtype=np.int8)
dbd_se5 = np.pad(dbd_se5, ((0, 0), (3, 3)), 'constant', constant_values=-1)
dbd_se5 = np.pad(dbd_se5, ((0, 0), (5, 5)), 'constant', constant_values=0)

dbd_se7 = np.ones((1, 7), dtype=np.int8)
dbd_se7 = np.pad(dbd_se7, ((0, 0), (3, 3)), 'constant', constant_values=-1)
dbd_se7 = np.pad(dbd_se7, ((0, 0), (5, 5)), 'constant', constant_values=0)

dbd_se9 = np.ones((1, 9), dtype=np.int8)
dbd_se9 = np.pad(dbd_se9, ((0, 0), (3, 3)), 'constant', constant_values=-1)
dbd_se9 = np.pad(dbd_se9, ((0, 0), (5, 5)), 'constant', constant_values=0)

top_end_se = np.pad(dbd_se5, ((1, 0), (0, 0)), 'constant', constant_values=-1)
top_end_se = np.pad(top_end_se, ((3, 0), (0, 0)), 'constant', constant_values=0)

bdb_se1 = np.zeros((1, 1), dtype=np.int8)
bdb_se1 = np.pad(bdb_se1, ((0, 0), (2, 2)), 'constant', constant_values=-1)
bdb_se1 = np.pad(bdb_se1, ((0, 0), (3, 3)), 'constant', constant_values=1)

bdb_se3 = np.zeros((1, 3), dtype=np.int8)
bdb_se3 = np.pad(bdb_se3, ((0, 0), (3, 3)), 'constant', constant_values=-1)
bdb_se3 = np.pad(bdb_se3, ((0, 0), (5, 5)), 'constant', constant_values=1)

bdb_se5 = np.zeros((1, 5), dtype=np.int8)
bdb_se5 = np.pad(bdb_se5, ((0, 0), (3, 3)), 'constant', constant_values=-1)
bdb_se5 = np.pad(bdb_se5, ((0, 0), (5, 5)), 'constant', constant_values=1)

bdb_se7 = np.zeros((1, 7), dtype=np.int8)
bdb_se7 = np.pad(bdb_se7, ((0, 0), (3, 3)), 'constant', constant_values=-1)
bdb_se7 = np.pad(bdb_se7, ((0, 0), (5, 5)), 'constant', constant_values=1)

bdb_se9 = np.zeros((1, 9), dtype=np.int8)
bdb_se9 = np.pad(bdb_se9, ((0, 0), (3, 3)), 'constant', constant_values=-1)
bdb_se9 = np.pad(bdb_se9, ((0, 0), (5, 5)), 'constant', constant_values=1)

hmt_selems = [
    [
        dbd_se1,
        dbd_se3,
        dbd_se5,
        dbd_se7,
        #dbd_se9,
    ],
    [
        bdb_se1,
        bdb_se3,
        bdb_se5,
        bdb_se7,
        #bdb_se9,
    ]
]


def uhmt(img: np.ndarray, se: np.ndarray, anchor: Tuple[int] = (-1, -1)) -> Tuple[np.ndarray, np.ndarray]:
    e_se = (1 * (se == 1)).astype(np.uint8)
    d_se = (1 * (se == 0)).astype(np.uint8)
    e: np.ndarray = cv.erode(img, e_se, borderType=cv.BORDER_REPLICATE, anchor=anchor)
    d: np.ndarray = cv.dilate(img, d_se, borderType=cv.BORDER_REPLICATE, anchor=anchor)
    mask: np.ndarray = e > d
    diff = (e.astype(np.int16) - d.astype(np.int16))
    diff[diff < 0] = 0
    return cv.convertScaleAbs(diff), (1 * mask).astype(np.uint8)


def is_night_(img: np.ndarray) -> bool:
    g = img[:, :, 1]
    return np.abs(np.mean(img[:, :, 2] - g)) < 10.0 and np.abs(np.mean(img[:, :, 0] - g)) < 10.0


def is_night(img: np.ndarray) -> bool:
    return False
    small = cv.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    return np.mean(small[:, :, 2] - small[:, :, 0]) < 1.0


def get_non_snow_images(path: Path, queue: Queue, count: int = 9) -> None:
    nights = 0
    images_loaded = 0
    for file in scandir(path):
        if file.name[-3:].lower() != "jpg":
            continue
        img = cv.imread(str(file.path))
        img = cv.pyrDown(img)
        if is_night(img):
            nights += 1
            continue
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        if is_non_snow(hsv):
            queue.put_nowait((file.path, img))
            images_loaded += 1
            if images_loaded == count:
                break
    queue.put_nowait(None)


asf_strels = [np.array([]) for _ in range(36)]

for sz in range(3, 35 + 1, 2):
    asf_strels[sz] = cv.getStructuringElement(cv.MORPH_RECT, (1, sz))


def asf(img: np.ndarray, size: int, w, mode: str = 'co') -> np.ndarray:
    c = img
    op3 = None
    if mode == 'co':
        op1 = cv.MORPH_OPEN
        op2 = cv.MORPH_CLOSE
    elif mode == 'oco':
        op1 = cv.MORPH_OPEN
        op2 = cv.MORPH_CLOSE
        op3 = cv.MORPH_OPEN
    elif mode == 'coc':
        op1 = cv.MORPH_CLOSE
        op2 = cv.MORPH_OPEN
        op3 = cv.MORPH_CLOSE
    elif mode == 'oc':
        op1 = cv.MORPH_CLOSE
        op2 = cv.MORPH_OPEN
    else:
        op1 = cv.MORPH_OPEN
        op2 = cv.MORPH_CLOSE

    for sz in range(3, size + 1, 2):
        strel = cv.getStructuringElement(cv.MORPH_RECT, (w, sz))
        c = cv.morphologyEx(c, op1, strel)
        c = cv.morphologyEx(c, op2, strel)
        if op3 is not None:
            c = cv.morphologyEx(c, op3, strel)
    return c


def ground_part(img: np.ndarray) -> np.ndarray:
    return img[int(0.4 * img.shape[0]):int(0.95 * img.shape[0]), int(0.2 * img.shape[1]):int(0.8 * img.shape[1])]


def is_snow(gray: np.ndarray, img: np.ndarray, sigma: float = 0.5, threshold: int = 5) -> bool:
    ground = ground_part(gray)
    bgr = ground_part(img)
    bl = cv.GaussianBlur(ground, (5, 5), sigmaX=sigma)
    diff = ground - bl
    diff_mean = np.array([[np.count_nonzero(diff > threshold) / (ground.shape[0] * ground.shape[1]),
                           np.mean(bgr[:, :, 0])]])
    # return snow_svc.predict(snow_scaler.transform(diff_mean))[0] > 0.0
    return True


def line_upright_bbox(line: np.ndarray, max_width: int, max_height: int, width=25, out_of_bounds: bool=False) -> np.ndarray:
    w = max(1, int((width - 1) / 2))
    if not out_of_bounds:
        top_left = np.maximum(np.min(line, axis=0) - w, 0)
    else:
        top_left = np.min(line, axis=0) - w
    bottom_right = np.max(line, axis=0) + w
    if not out_of_bounds:
        bottom_right[0] = np.minimum(bottom_right[0], max_width - 1)
        bottom_right[1] = np.minimum(bottom_right[1], max_height - 1)
    return np.array([top_left, bottom_right])


def angle_of_line(line: np.ndarray, normal: bool = False) -> float:
    v = line[1] - line[0]
    return (math.degrees(math.atan2(v[1], v[0]))) % 180


def line_vector(line: np.ndarray) -> np.ndarray:
    v = line[1] - line[0]
    return v / (np.linalg.norm(v) + 0.00001)


def line_edge_offsets(line: np.ndarray, mag: np.ndarray, w: int = 19) -> Tuple[List[int], np.ndarray]:
    # bbox = line_upright_bbox(line, 17)
    h_w = (w - 1) // 2
    mag_lines = skimage.measure.profile_line(mag, line[0, ::-1], line[1, ::-1], linewidth=w, reduce_func=None,
                                             mode='constant')
    line_sums = np.sum(mag_lines, axis=0)
    left_edge_idx = np.argmax(line_sums[::-1][h_w + 1:]) + 1
    right_edge_idx = np.argmax(line_sums[h_w + 1:]) + 1

    return [left_edge_idx, right_edge_idx], mag_lines


params = {
    'C': -5,
    'blockSize': 15,
    'method': 'mean',
    'op1_h': 5,
    'op2_h': 17,
    'cl1_h': 5,
    'cl2_h': 15,
    'asf_sz': 25,
    'asf_mode': 'oco',
    'hough_th': 40,
    'line_length': 30,
    'line_gap': 25,
    'hyst_low': 5,
    'hyst_high': 10,
    'hmt_se_size': 3,
    'f': 0.5,
}


def detect_sticks(gray: np.ndarray, align_endpoints: bool = False, equalize: bool = True):
    clahe.setClipLimit(5.0)
    #if equalize and False:
    #    gray_eq = clahe.apply(gray)
    #else:
    #    gray_eq = gray
    #gray_eq = cv.convertScaleAbs(gray_eq, None, 0.2)
    gray_eq = gray
    line_count = 140

    while line_count > 100:
        hmt_dbd1, _ = uhmt(gray_eq, hmt_selems[0][0])
        hmt_dbd3, _ = uhmt(gray_eq, hmt_selems[0][1])
        hmt_dbd5, _ = uhmt(gray_eq, hmt_selems[0][2])
        hmt_dbd7, _ = uhmt(gray_eq, hmt_selems[0][3])
        #hmt_dbd9, _ = uhmt(gray_eq, hmt_selems[0][4])
        hmt_bdb1, _ = uhmt(gray_eq, hmt_selems[1][0])
        hmt_bdb3, _ = uhmt(gray_eq, hmt_selems[1][1])
        hmt_bdb5, _ = uhmt(gray_eq, hmt_selems[1][2])
        hmt_bdb7, _ = uhmt(gray_eq, hmt_selems[1][3])

        threshold = 9.0

        _, hmt_dbd1_ = cv.threshold(hmt_dbd1, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_dbd3_ = cv.threshold(hmt_dbd3, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_dbd5_ = cv.threshold(hmt_dbd5, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_dbd7_ = cv.threshold(hmt_dbd7, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        #_, hmt_dbd9_ = cv.threshold(hmt_dbd9, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_bdb1_ = cv.threshold(hmt_bdb1, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_bdb3_ = cv.threshold(hmt_bdb3, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_bdb5_ = cv.threshold(hmt_bdb5, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        _, hmt_bdb7_ = cv.threshold(hmt_bdb7, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        #_, hmt_bdb9_ = cv.threshold(hmt_bdb9, threshold, 1.0, cv.THRESH_BINARY)  # | cv.THRESH_OTSU)
        th1 = asf(hmt_dbd1_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_dbd1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th2 = asf(hmt_dbd3_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_dbd3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th3 = asf(hmt_bdb1_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_dbd5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th4 = asf(hmt_bdb3_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_dbd7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th5 = asf(hmt_bdb5_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_bdb1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th6 = asf(hmt_dbd5_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_bdb3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th7 = asf(hmt_dbd7_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_bdb5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
        th8 = asf(hmt_bdb7_, 9, 1,
                  'co')  # cv.morphologyEx(hmt_bdb7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))

        th = np.bitwise_or(th1, th2)
        th = np.bitwise_or(th, th3)
        th = np.bitwise_or(th, th4)
        th = np.bitwise_or(th, th5)
        th = np.bitwise_or(th, th6)
        th = np.bitwise_or(th, th7)
        th = np.bitwise_or(th, th8)

        th_ = cv.erode(th, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), anchor=(0, 0))
        th_ = cv.dilate(th_, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), anchor=(1, 0))

        lines = cv.HoughLinesP(th_, 1.0, np.pi / 180.0, params['hough_th'], None, params['line_length'], params['line_gap'])
        line_count = 0 if lines is None else len(lines)
        if line_count > 100:
            gray_eq = cv.convertScaleAbs(gray_eq, None, 0.5)
    if lines is None:
        return []
    lines = np.reshape(lines, (-1, 2, 2))
    for line in lines:
        if line[0, 1] > line[1, 1]:
            t = line[1].copy()
            line[1] = line[0]
            line[0] = t
    return lines


def apply_multi_hmt(img: np.ndarray, height: int = 1, width: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    dbd_hmts = []
    bdb_hmts = []

    for selem in hmt_selems[0][1:]:
        s = selem
        if height > 1:
            s = np.tile(selem, (height, 1))
        dbd_hmts.append(uhmt(img, s, anchor=(selem.shape[1] // 2, 0))[0])

    for selem in hmt_selems[1][1:]:
        s = selem
        if height > 1:
            s = np.tile(selem, (height, 1))
        bdb_hmts.append(uhmt(img, s, anchor=(selem.shape[1] // 2, 0))[0])

    return reduce(np.bitwise_or, dbd_hmts), reduce(np.bitwise_or, bdb_hmts)


def extend_line(line: np.ndarray, amount: int, endpoints: str = 'both') -> np.ndarray:
    """
    Extends `line` by `amount` units from top and/or bottom. If the line length < 1, then returns `line`.

    Parameters
    ----------
    line : (2, 2) np.ndarray
        The line to be extended.
    amount : int
        Number of units to extend the line by.
    endpoints : str
        if 'both' then the line is extended from both top and bottom therefore adding 2 * `amount`

    Returns
    -------
    np.ndarray
        the extended line
    """
    vec = (line[1] - line[0]).astype(np.float32)
    line_len = np.linalg.norm(vec)
    if line_len < 1:
        return line
    vec /= line_len
    new_line = line.copy()

    if endpoints == 'top' or endpoints == 'both':
        new_line[0] -= (amount * vec).astype(np.int32)
    if endpoints == 'bottom' or endpoints == 'both':
        new_line[1] += (amount * vec).astype(np.int32)

    return new_line


def find_sticks(gray: np.ndarray, bgr: np.ndarray, equalize: bool = True) -> List[np.ndarray]:
    gray = gray[:-100, :]
    dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
    mag = cv.magnitude(dx, dy)
    angles = cv.phase(dx, dy, None, angleInDegrees=True)
    angles = np.mod(angles - 90, 180)

    lines1 = detect_sticks(gray, equalize=equalize)
    lines1 = list(lines1)
    lines1 = merge_lines_with_same_orientation(lines1)
    lines1 = np.array(lines1)

    lines05 = detect_sticks(cv.pyrDown(gray), equalize=equalize)
    lines05 = 2 * lines05
    lines05 = list(lines05)
    lines05 = merge_lines_with_same_orientation(list(lines05))
    lines05 = np.array(lines05)

    if len(lines1) > 0 and len(lines05) > 0:
        lines = np.array(merge_lines_with_same_orientation(list(np.vstack((lines1, lines05)))))
    else:
        lines = lines1 if len(lines1) > 0 else lines05
    valid_flags: List[bool] = []
    final_lines = []


    for line in lines:
        line_vec = (line[1] - line[0]).astype(np.float32)
        if np.linalg.norm(line_vec) < 2:
            continue
        line_vec /= np.linalg.norm(line_vec)
        a = 20
        w = 25
        w_h = (w - 1) // 2
        line_ = extend_line(line, amount=a, endpoints='both')
        line_[1, 1] = np.minimum(line_[1, 1], gray.shape[0] - 1)

        edge_offsets, mag_line = line_edge_offsets(line_, mag, w)

        left = w_h - edge_offsets[0]
        right = w_h + edge_offsets[1]

        line_angle = angle_of_line(line)
        line_sample = skimage.measure.profile_line(gray, line[0, ::-1], line[1, ::-1], linewidth=w, reduce_func=None,
                                                   mode='constant')
        angle_sample = skimage.measure.profile_line(angles, line[0, ::-1], line[1, ::-1], linewidth=w,
                                                    reduce_func=None, mode='constant')

        f_vec = extract_feature_vector(line_sample, None, angle_sample, line_angle, left, right)
        valid_flags.append(stick_pipeline.predict([f_vec]) > 0.0)
        width = max(int(edge_offsets[0] + edge_offsets[1]), 3)
        final_lines.append((line, valid_flags[-1], width))

    return final_lines


def extract_feature_vector(gray: np.ndarray, mag: np.ndarray, angle: np.ndarray, line_angle: float, left_edge: int,
                           right_edge: int) -> np.ndarray:
    off = 2
    left_angles = angle[:, max(0, left_edge - off):left_edge + off + 1]
    left_angles = np.where(left_angles > 180, left_angles - 180, left_angles)

    right_angles = angle[:, right_edge - off:min(angle.shape[1], right_edge + off + 1)]
    right_angles = np.where(right_angles > 180, right_angles - 180, right_angles)

    right_angle_diff = np.abs(right_angles - line_angle)
    right_angle_diff_count = np.count_nonzero(right_angle_diff < 11) / 3.0
    left_angle_diff = np.abs(left_angles - line_angle)
    left_angle_diff_count = np.count_nonzero(left_angle_diff < 11) / 3.0

    diff = np.abs(left_angles[:, 1] - right_angles[:, 1])
    intensity = gray[:, (gray.shape[1]) // 2]

    return np.array([
        np.mean(diff),
        np.std(diff),
        np.std(left_angles),
        np.std(right_angles),
        np.abs(np.mean(left_angles) - np.mean(right_angles)),
        np.std(intensity),
        np.mean(left_angle_diff),
        left_angle_diff_count / float(left_angles.shape[0]),
        right_angle_diff_count / float(right_angles.shape[0]),
    ])


kern_width_ = 1
zeros = np.zeros((4, kern_width_), np.uint8)
kern_bot = np.ones((1, kern_width_), np.uint8)
kern_bot = np.tile(kern_bot, (5, 1))
kern_bot = np.vstack((zeros, kern_bot))
kern_top = kern_bot[::-1, :]


def get_top_endpoint_r(stick: Stick, full: np.ndarray, half: np.ndarray, quart: np.ndarray, offset: np.ndarray = np.array([0, 0])) -> Optional[np.ndarray]:
    scale = 1
    width = stick.width
    stick_ = stick.scale(1.0)
    img = full

    if width < 3:
        width = 3

    amount = 50
    line = stick_.line()
    line_e = extend_line(line, amount, endpoints='top')
    offset = offset[::-1]
    line_sample = skimage.measure.profile_line(img, line_e[0, ::-1] + offset, line_e[1, ::-1] + offset,
                                               linewidth=5 * width, reduce_func=None,
                                               mode='constant')
    x = int(2.5 * width)
    y_off = amount
    if width < 5:
        line_sample = cv.pyrUp(line_sample)
        x *= 2
        y_off *= 2
        width *= 2
        scale = 0.5
    elif width > 10:
        line_sample = cv.pyrDown(line_sample)
        x = int(x * 0.5)
        y_off = int(0.5 * amount)
        width = int(width * 0.5)
        scale = 2
    if width % 2 == 0:
        width += 1

    dbd, bdb = apply_multi_hmt(line_sample, height=3)
    dbd, bdb = 1 * (dbd > 9).astype(np.uint8), 1 * (bdb > 9).astype(np.uint8)

    hmts = np.bitwise_or(dbd, bdb)
    hmts = asf(hmts, 5, w=1, mode='co')

    angle = math.atan2(*(line[1] - line[0])[::-1]) - 0.5 * np.pi

    coords: List[List[int]] = []

    fil_bot = cv.filter2D(hmts, cv.CV_8U, kern_bot, None, borderType=cv.BORDER_CONSTANT)
    fil_top = cv.filter2D(hmts, cv.CV_8U, kern_top, None, borderType=cv.BORDER_CONSTANT, anchor=(0, 5))

    for y_ in range(y_off - 3, y_off + 4):
        for x_ in range(x - 2 * kern_width_, x + 2 * kern_width_ + 1):
            is_e2 = fil_bot[y_, x_] == 5
            is_e2 = is_e2 and fil_top[y_, x_] == 0 and hmts[y_, x_] > 0
            if is_e2:
                coords.append([x_, y_])

    if len(coords) > 0:
        coords = np.array(coords)
        mean = np.round(coords.mean(axis=0)).astype(np.int32)
        line_orig = stick.line()
        new_top = coords[np.abs(coords - mean).sum(axis=1).argmin()] - np.array([x, y_off], np.int32)
        glob_x = np.round(scale * (new_top[0] * math.cos(angle) - new_top[1] * math.sin(angle)) + line_orig[0, 0])
        glob_y = np.round(scale * (new_top[0] * math.sin(angle) + new_top[1] * math.cos(angle)) + line_orig[0, 1])
        result = np.array([glob_x, glob_y], np.int32)
    else:
        result = None

    return result


def length_of_line(line: np.ndarray) -> float:
    return np.linalg.norm(line[0] - line[1])


def merge_lines_with_same_orientation(lines: List[np.ndarray], max_gap: int = 25) -> List[np.ndarray]:
    lines_ = sorted(lines, key=length_of_line, reverse=True)
    lines_ = list(map(lambda line: (line, angle_of_line(line), False), lines_))

    final_lines = []
    for i, line_tuple in enumerate(lines_):
        line = line_tuple[0]
        line_orient = line_tuple[1]
        line_vec = line_vector(line)

        for j in range(i + 1, len(lines_)):
            if abs(line_orient - lines_[j][1]) > 5 or lines_[j][2]:
                continue
            line2 = lines_[j][0]
            line2_orient = lines_[j][1]
            v = line2[0] - line[0]
            dot = np.dot(line_vec, v)
            alpha = math.acos(dot / (np.linalg.norm(v) + 0.0001))
            if line_to_line_distance(line, line2) > 15:
                continue
            if max_gap > 0 and (line[1, 1] - line2[0, 1] < -max_gap or line[0, 1] - line2[1, 1] > max_gap):
                continue
            if line2[1, 1] > line[1, 1]:
                v_ = line2[1] - line[0]
                v_proj = line_vec * np.dot(line_vec, v_)
                new_bottom = line[0] + v_proj
                line[1] = np.round(new_bottom).astype(np.int32)
            elif line2[0, 1] < line[0, 1]:
                v_ = line2[0] - line[0]
                v_proj = line_vec * np.dot(line_vec, v_)
                new_top = line[0] + v_proj
                line[0] = np.round(new_top).astype(np.int32)
            lines_[j] = (line2, line2_orient, True)
        if not lines_[i][2]:
            final_lines.append(line)
        lines_[i] = (line, line_orient, True)

    return final_lines


def fit_into_length(line: np.ndarray, length: float, anchor: str = 'top') -> np.ndarray:
    vec = line_vector(line)
    bottom = line[0] + length * vec
    bottom = np.round(bottom).astype(np.int32)

    line_to_return = np.array([line[0], bottom])
    return np.array([line[0], bottom])


def fit_into_length_from_bottom(line: np.ndarray, length: float) -> np.ndarray:
    vec = -1.0 * line_vector(line)
    top = line[1] + length * vec
    top = np.round(top).astype(np.int32)

    line_to_return = np.array([top, line[1]])
    return line_to_return


def deb_draw_lines(img: np.ndarray, lines: Union[List[np.ndarray], np.ndarray]):
    dr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for l in lines:
        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
        cv.line(dr, (int(l[0, 0]), int(l[0, 1])), (int(l[1, 0]), int(l[1, 1])), color)
    cv.imshow('dr', dr)
    cv.waitKey(0)
    cv.destroyWindow('dr')


imag = None
my_debug = False

def match_detections_to_sticks(sticks: List[StickDetection], new_sticks: List[np.ndarray]) -> List[Tuple[StickDetection, np.ndarray]]:
    global imag
    possible_matches: List[Tuple[int, List[Tuple[StickDetection, np.ndarray]]]] = []

    for i, stick in enumerate(sticks):
        for j, n_stick in enumerate(new_sticks):
            vec = stick.old_stick.top - n_stick[0]
            num_matches = 0
            matches: List[Tuple[StickDetection, np.ndarray]] = []
            matches2: List[List[Tuple[float, StickDetection]]] = [[] for _ in range(len(new_sticks))]
            match_map: Dict[StickDetection, Tuple[int, float]] = {}
            for k, n_stick_ in enumerate(new_sticks):
                n_stick_offset = n_stick_ + vec
                n_bbox = line_upright_bbox(n_stick_offset, 5000, 5000, out_of_bounds=True)
                n_bbox_area = box_area(n_bbox)
                top = n_stick_offset[0]
                for stick_d in sticks:
                    stick_ = stick_d.old_stick
                    dist = np.linalg.norm(stick_.top - top)
                    #if dist < 20:
                    #    other_match = match_map.setdefault(stick_d, (-1, 999999))
                    #    if other_match[1] > dist:
                    #        if other_match[0] >= 0:
                    #            other_matches = matches2[other_match[0]]
                    #            other_matches = list(filter(lambda m: m[1] != stick_d, other_matches))
                    #            matches2[other_match[0]] = other_matches
                    #        match_map[stick_d] = (k, dist)
                    #    matches2[k].append((dist, stick_d, None, None, None))
                    bbox = line_upright_bbox(stick_.line(), 5000, 5000, out_of_bounds=True)
                    intersection = boxes_intersection(n_bbox, bbox)
                    #if intersection is None:
                    #    intersection = bbox
                    vec_ = stick_.top - n_stick_[0]
                    if intersection is not None:
                        intersection_area = box_area(intersection)
                        bbox_area = box_area(bbox)
                        if intersection_area / float(bbox_area) > 0.1 and length_of_line(n_stick_) < 1.2 * stick_.length_px:
                            num_matches += 1
                            mm = matches2[k]
                            bbox_area = box_area(bbox)
                            intersection_area = box_area(intersection)
                            mm.append((bbox_area / intersection_area, stick_d, intersection,
                                       np.linalg.norm(stick_.top - n_stick_[0])))
                            matches.append((stick_d, n_stick_))
            for mm in matches2:
                #if len(mm) > 1:
                #    avg = np.mean(list(map(lambda ts: ts[3], mm)))
                if len(mm) > 1:
                    mm.sort(key=lambda s: s[0])
            matches2 = zip(matches2, new_sticks)
            matches2 = list(filter(lambda zipped: len(zipped[0]) > 0, matches2))
            #matches = list(map(lambda mm: (mm[0][0][1:], mm[1], mm[0][0][0]), matches2))
            matches = list(map(lambda mm: (mm[0][0][1:], mm[1]), matches2))
            #total_diff_inverse = sum(map(lambda mm: 1.0 / (mm[2] + 0.00001), matches))
            #matches = list(map(lambda mm: mm[:2], matches))
            if my_debug:
                dd = imag.copy()
                for match in matches:
                    s1 = match[0][0].old_stick
                    bax = match[0][1]
                    s2 = match[1]
                    b1 = line_upright_bbox(s1.line(), 5000, 5000)
                    b2 = line_upright_bbox(s2, 5000, 5000)
                    col = [randint(0, 255), randint(0, 255), randint(0, 255)]
                    cv.rectangle(dd, (int(b1[0, 0]), int(b1[0, 1])), (int(b1[1, 0]), int(b1[1,1])), col, 2)
                    cv.rectangle(dd, (int(b2[0, 0]), int(b2[0, 1])), (int(b2[1, 0]), int(b2[1, 1])), col, 2)
                    cv.rectangle(dd, (int(bax[0, 0]), int(bax[0, 1])), (int(bax[1, 0]), int(bax[1, 1])), col, 1)
                b1_ = line_upright_bbox(stick.old_stick.line(), 5000, 5000)
                b2_ = line_upright_bbox(n_stick, 5000, 5000)
                cv.rectangle(dd, (int(b1_[0, 0]), int(b1_[0, 1])), (int(b1_[1, 0]), int(b1_[1, 1])), [0, 0, 0], 4)
                cv.rectangle(dd, (int(b2_[0, 0]), int(b2_[0, 1])), (int(b2_[1, 0]), int(b2_[1, 1])), [0, 0, 0], 4)
                cv.imshow(f'match{len(matches)}', cv.resize(dd, (0, 0), fx=0.5, fy=0.5))
                cv.waitKey(0)
                cv.destroyWindow(f'match{len(matches)}')
            #possible_matches.append((len(matches) + total_diff_inverse, matches))
            possible_matches.append((len(matches), matches))
    possible_matches.sort(key=lambda e: e[0], reverse=True)
    return list(map(lambda t: t[1], possible_matches))[0]


def process_detection(det: StickDetection):
    if 1.1 * det.old_stick.length_px < det.new_stick.length_px:
        det.stick_to_use = det.old_stick.copy()
        det.valid = False
        return
    if not det.old_corrected_matches_old and det.new_corrected_matches_new:
        if det.old_stick.length_px > det.new_stick.length_px or True:
            new_line_ = det.new_stick.line()
            if np.linalg.norm(det.old_stick.bottom - det.new_stick.bottom) < 7:
                new_line_ = fit_into_length_from_bottom(new_line_, det.old_stick.length_px)
            else:
                new_line_ = fit_into_length(new_line_, det.old_stick.length_px)
            det.new_stick.set_top(new_line_[0])
            det.new_stick.set_bottom(new_line_[1])
            det.top_diff = det.new_stick.line()[0] - det.old_stick.line()[0]
        det.stick_to_use = det.new_stick
        det.bottom_diff = det.new_stick.bottom - det.old_stick.bottom
    elif det.top_is_same:
        new_line_ = det.new_stick.line()
        new_line_ = fit_into_length(new_line_, det.old_stick.length_px)
        det.new_stick.set_top(new_line_[0])
        det.new_stick.set_bottom(new_line_[1])
        det.bottom_diff = det.new_stick.bottom - det.old_stick.bottom
        det.stick_to_use = det.new_stick
    else:
        det.bottom_diff = det.new_stick.bottom - det.old_stick.bottom
        det.stick_to_use = det.old_stick.copy()


def handle_big_camera_movement(img: np.ndarray, half: np.ndarray, quart: np.ndarray, stick_detections: List[StickDetection]) -> Optional[List[StickDetection]]:
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

        new_line = fit_into_length(new_line, det.old_stick.length_px)
        n_stick.set_top(new_line[0])
        n_stick.set_bottom(new_line[1])

        obox = line_upright_bbox(det.old_stick.line(), 5000, 5000, 35)
        obox_t = obox + vec
        cv.rectangle(dd, (obox[0, 0], obox[0, 1]), (obox[1, 0], obox[1, 1]), [255, 0, 0], 2)
        cv.rectangle(dd, (obox_t[0, 0], obox_t[0, 1]), (obox_t[1, 0], obox_t[1, 1]), [255, 0, 255], 2)

        det.new_stick = n_stick
        det.stick_to_use = det.new_stick
        det.valid = True

    #cv.imshow('matches', dd)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return stick_detections, dd


def validate_positions(sticks: List[StickDetection], stick_to_stick: Dict[Stick, Dict[Stick, np.ndarray]]) -> Tuple[List[StickDetection], List[StickDetection]]:
    matches: List[Tuple[int, StickDetection]] = []
    for det in sticks:
        match_count = 0
        for det2 in sticks:
            if det == det2:
                continue
            old_vec = stick_to_stick[det.old_stick][det2.old_stick]
            old_len = np.linalg.norm(old_vec)
            new_vec = det2.stick_to_use.bottom - det.stick_to_use.bottom
            new_len = np.linalg.norm(new_vec)
            if abs(old_len - new_len) < 1.3 * (det.old_stick.width + det2.old_stick.width):
                cos_angle = np.dot(old_vec / old_len, new_vec / (new_len + 0.00001))
                if math.sqrt(new_len * new_len + old_len * old_len - 2 * new_len * old_len * cos_angle) < 10:
                    match_count += 1
        matches.append((match_count, det))
    #matches.sort(key=lambda e: e[0])
    it1, it2 = tee(matches)
    n = int(0.25 * len(sticks))
    return list(map(lambda e: e[1], filter(lambda e: e[0] >= 3, it1))), list(map(lambda e: e[1], filter(lambda e: e[0] < 3, it2)))


def deb_draw(img: np.ndarray, dets: Union[List[Stick], List[StickDetection], List[np.ndarray]], color: List[int], im_id: str, which: str='old'):
    if len(dets) == 0:
        return
    if isinstance(dets[0], StickDetection):
        if which == 'old':
            sticks = list(map(lambda d: d.old_stick, dets))
        elif which == 'use':
            sticks = list(map(lambda d: d.stick_to_use, dets))
        else:
            sticks = list(map(lambda d: d.new_stick, dets))
    else:
        sticks = dets

    for stick in sticks:
        if isinstance(stick, Stick):
            top = stick.top
            bottom = stick.bottom
            cv.line(img, (int(top[0]), int(top[1])), (int(bottom[0]), int(bottom[1])), color, 2)
        else:
            top = stick[0]
            bottom = stick[1]
            cv.line(img, (int(top[0]), int(top[1])), (int(bottom[0]), int(bottom[1])), color, 2)

    cv.imshow(im_id, cv.resize(img, (0, 0), fx=0.5, fy=0.5))


def boxes_intersection(box1: np.ndarray, box2: np.ndarray) -> Optional[np.ndarray]:
    x1 = max(box1[0, 0], box2[0, 0])
    y1 = max(box1[0, 1], box2[0, 1])
    x2 = min(box1[1, 0], box2[1, 0])
    y2 = min(box1[1, 1], box2[1, 1])

    if x1 >= x2 or y1 >= y2:
        return None
    return np.array([[x1, y1], [x2, y2]], np.int32)


def box_area(box: np.ndarray) -> int:
    return (box[1, 0] - box[0, 0]) * (box[1, 1] - box[0, 1])


def process_photoss_deb(images: List[str], folder: Path, sticks: List[Stick], stick_to_stick: Dict[Stick, Dict[Stick, np.ndarray]]) -> Measurement:
    measurement_struct = Measurement()
    measurements: Dict[str, List[Stick]] = {}
    sticks_ = list(map(lambda stick: stick.copy(), sticks))
    detections: List[StickDetection] = [StickDetection() for _ in range(len(sticks))]
    found_sticks_: List[StickDetection] = []
    not_found_sticks: List[StickDetection] = []
    to_process = images.copy()
    stickS = list(zip(sticks_, sticks))
    dict_sticks = {stick: stick for stick in sticks}

    for img_name in images:
        print(f'processing {img_name}')
        orig = cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE)
        orig = cv.pyrDown(orig)
        half = cv.pyrDown(orig)
        quart = cv.pyrDown(half)

        measured_sticks: List[Stick] = []

        out_of_frame_sticks: int = len(sticks)
        small_movements: int = 0
        big_movements: int = 0
        no_movements: int = 0
        found_sticks: int = 0
        more_than_one: int = 0
        found_sticks_.clear()
        not_found_sticks.clear()
        measurement_struct.current_img = img_name

        for i, stick in enumerate(sticks_):

            scale = 1
            stick_line = stick.line()

            old_stick = stick.copy()
            new_stick = stick.copy()

            new_stick.view = img_name
            old_line = old_stick.line()

            bbox = line_upright_bbox(stick_line, orig.shape[1], orig.shape[0], width=35)

            detection = detections[i]
            detection.reset()
            old_top_endpoint = get_top_endpoint_r(stick, orig, half, quart)
            detection.old_stick = stick
            detection.old_orientation = angle_of_line(old_line)
            detection.old_corrected_endpoint = old_top_endpoint
            detection.old_corrected_matches_old = detection.old_corrected_endpoint is not None and \
                                                  np.linalg.norm(stick.line()[0] - old_top_endpoint) < 4
            detection.bbox = bbox

            old_line_length = old_stick.length_px

            old_line_angle = angle_of_line(old_line)


            line_roi = orig[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
            line_roi_h = cv.pyrDown(line_roi)

            lines = list(detect_sticks(line_roi, equalize=True))
            lines2 = list(2 * detect_sticks(line_roi_h, equalize=True))
            w = max(5, stick.width)

            lines = np.array(merge_lines_with_same_orientation(lines, max_gap=-1))
            lines2 = np.array(merge_lines_with_same_orientation(lines2, max_gap=-1))

            if len(lines) > 0 and len(lines2) > 0:
                lines = np.array(merge_lines_with_same_orientation(list(np.vstack((lines, lines2))),
                                                                   max_gap=-1))
            else:
                lines = lines if len(lines) > 0 else lines2

            lines = list(lines)
            lines = merge_lines_with_same_orientation(lines)

            if len(lines) == 1:
                found_sticks += 1
                new_line = lines[0] + bbox[0]

                new_stick.set_top(new_line[0])
                new_stick.set_bottom(new_line[1])

                new_stick_full = new_stick
                new_line_length = np.linalg.norm(new_line[0] - new_line[1])
                new_line_angle = angle_of_line(new_line)

                new_line_vec = (new_line[1] - new_line[0]) / np.linalg.norm(new_line[1] - new_line[0])
                new_top_endpoint = get_top_endpoint_r(new_stick_full, orig, half,
                                                      quart)

                top_diff = new_line[0] - old_line[0]
                top_diff_mag = np.linalg.norm(top_diff)
                bottom_diff = new_line[1] - old_line[1]
                bottom_diff_mag = np.linalg.norm(bottom_diff)

                new_stick.set_snow_height_px(0)

                detection.valid = True

                detection.new_stick = new_stick_full
                detection.new_orientation = angle_of_line(new_stick_full.line())
                detection.new_corrected_endpoint = new_top_endpoint

                detection.top_diff = top_diff

                detection.orientation_diff = detection.old_orientation - detection.new_orientation

                detection.orientation_is_same = abs(detection.orientation_diff) < 2
                detection.top_is_same = np.linalg.norm(top_diff) <= stick.width * 1.0
                old_new_vec = (new_line[1] - old_line[0]) / np.linalg.norm(new_line[1] - old_line[0])
                detection.new_lies_on_old = detection.orientation_is_same and \
                                            np.abs(np.dot(new_line_vec, old_new_vec)) > 0.997
                detection.old_corrected_matches_old = detection.old_corrected_endpoint is not None and \
                                                      np.linalg.norm(stick.line()[0] - old_top_endpoint) < 4
                detection.new_corrected_matches_new = detection.new_corrected_endpoint is not None and \
                                                      np.linalg.norm(new_stick_full.line()[0] - new_top_endpoint) < 4

                bw = bbox[1, 0] - bbox[0, 0] - 5
                bh = bbox[1, 1] - bbox[0, 1] - 5
                line_ = lines[0]
                detection.in_frame = 5 < line_[0, 0] < bw and 5 < line_[1, 0] < bw and 5 < line_[0, 1] < bh \
                                     and 5 < line_[1, 1] < bh

                process_detection(detection)
                assert abs(length_of_line(detection.stick_to_use.line()) - length_of_line(detection.old_stick.line())) < 5

                if not np.isnan(detection.bottom_diff[0]):
                    no_movements += 1
                    if abs(detection.bottom_diff[0]) > detection.old_stick.width:  # np.linalg.norm(detection.bottom_diff) > 5:
                        no_movements -= 1
                        if detection.in_frame:
                            small_movements += 1
                if detection.in_frame:
                    out_of_frame_sticks -= 1
                else:
                    big_movements += 1
                found_sticks_.append(detection)
            elif len(lines) > 1:
                more_than_one += 1
                not_found_sticks.append(detection)
            else:
                not_found_sticks.append(detection)

        assert len(found_sticks_) <= len(sticks)
        assert len(found_sticks_) + len(not_found_sticks) == len(sticks)
        dd = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
        dd_ = dd.copy()
        deb_draw(dd_, found_sticks_, [0, 255, 0], "found_pre", which='use')
        #found_sticks_, not_found_sticks_ = truly_valid_detections(found_sticks_, camera)
        #found_sticks = len(found_sticks_)
        #for nf in not_found_sticks_:
        #    not_found_sticks.append(nf)
        dd1 = dd.copy()
        deb_draw(dd1, found_sticks_, [0, 255, 0], "found", which='use')
        dd2 = dd.copy()
        deb_draw(dd2, not_found_sticks, [0, 255, 0], "not_found", which='old')

        if found_sticks <= int(0.3 * len(sticks)) or out_of_frame_sticks >= int(0.5 * len(sticks)): #  or big_movements >= int(0.8 * len(sticks)):
            print('BIG')
            print(f'no = {no_movements}, small = {small_movements}, big = {big_movements}, found = {found_sticks} / {len(sticks)}')
            measured_sticks.clear()
            sd, mat = handle_big_camera_movement_deb(orig, half, quart, detections)
            mat = mat[0]
            found_sticks_ = list(filter(lambda d: d.valid, sd))
            dd2_ = dd.copy()
            deb_draw(dd2_, found_sticks_, [0, 255, 0], "big_found_raw", which='use')
            dr1 = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
            for d in found_sticks_:
                #measured_sticks.append(d.stick_to_use)
                col = [randint(0, 255), randint(0, 255), randint(0, 255)]
                cv.line(dr1, (int(d.stick_to_use.top[0]), int(d.stick_to_use.top[1])),
                        (int(d.stick_to_use.bottom[0]), int(d.stick_to_use.bottom[1])), col, 3)
            not_found_sticks = list( filter(lambda d: not d.valid, sd))
            print(len(not_found_sticks))
            #found_sticks_, not_found_sticks_ = truly_valid_detections(found_sticks_, camera)
            ##found_sticks = len(found_sticks_)
            #for nf in not_found_sticks_:
            #    not_found_sticks.append(nf)
            dd3_ = dd.copy()
            dd4_ = dd.copy()
            deb_draw(dd3_, found_sticks_, [0, 255, 0], "big_found", which='use')
            deb_draw(dd4_, not_found_sticks, [0, 255, 0], "big_not_found", which='old')

            #assert len(not_found_sticks) + len(found_sticks_) == len(sticks)
            cv.imshow('old', cv.resize(dr1, (0,0), fx=0.5, fy=0.5))
            #cv.imshow('new', cv.resize(dr2, (0, 0), fx=0.5, fy=0.5))
            cv.waitKey(0)
            #cv.destroyAllWindows()
            #measurement_struct.reason = Reason.SticksMoved
            measurement_struct.remaining_photos = to_process
            measurement_struct.last_valid_sticks = sticks_

        assert len(measured_sticks) <= len(sticks)
        for d in found_sticks_:
            measured_sticks.append(d.stick_to_use)
        assert len(measured_sticks) <= len(sticks)
        assert len(found_sticks_) + len(not_found_sticks) == len(sticks)
        for nf in not_found_sticks:
            nf_stick = nf.old_stick
            new_vecs: List[np.ndarray] = []
            for det_ in found_sticks_:
                vec = det_.stick_to_use.bottom - det_.old_stick.bottom
                new_vecs.append(vec)
            new_vec = np.mean(new_vecs, axis=0).astype(np.int32)
            new_stick = None
            if nf.new_stick is not None:
                old_to_new = nf.new_stick.bottom - nf.old_stick.bottom
                if abs(np.linalg.norm(old_to_new) - np.linalg.norm(new_vec)) < 20:
                    if np.dot(old_to_new, new_vec) > 0:
                        new_stick = nf.new_stick
            if new_stick is None:
                new_stick = nf_stick.copy()
                new_stick.translate(new_vec)
            measured_sticks.append(new_stick)
        dd5 = dd.copy()
        deb_draw(dd5, measured_sticks, [0, 255, 0], "final_sticks", which='use')
        measurements[img_name] = measured_sticks
        for mstick in measured_sticks:
            model = dict_sticks[mstick]
            assert abs(length_of_line(mstick.line()) - length_of_line(model.line())) < 3
        if measurement_struct.reason == Reason.SticksMoved:
            measurement_struct.sticks_to_confirm = measured_sticks
            measurement_struct.current_img = img_name
            break
        measurement_struct.measurements[img_name] = measured_sticks
        measurement_struct.last_img = img_name
        to_process.remove(img_name)
        sticks_ = measured_sticks
        assert len(sticks_) == len(sticks)
        cv.waitKey(0)
        cv.destroyAllWindows()
    cv.destroyAllWindows()
    return measurement_struct


def handle_big_camera_movement_deb(img: np.ndarray, half: np.ndarray, quart: np.ndarray, stick_detections: List[StickDetection]) -> Tuple[List[StickDetection], np.ndarray]:
    global imag
    new_sticks = find_sticks(img, img, True)
    new_sticks = list(map(lambda s: s[0], filter(lambda s: s[1], new_sticks)))
    imag = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    imga = imag.copy()
    deb_draw(imga, new_sticks, [0, 255, 0], "thisjustnow")
    cv.waitKey(0)
    #matches = match_detections_to_sticks(stick_detections, new_sticks)
    matches = match_new_sticks_with_old(stick_detections, new_sticks)
    dr1 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    dr2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for det in stick_detections:
        det.valid = False
    X_points: List[List[int, int]] = []
    x_points: List[List[int, int]] = []
    for match in matches:
        det: StickDetection = match[0][0]
        n_stick = det.old_stick.copy()
        new_line = match[1]
        new_line = fit_into_length(new_line, det.old_stick.length_px)
        n_stick.set_top(new_line[0])
        n_stick.set_bottom(new_line[1])

        det.new_stick = n_stick
        new_top_endpoint = get_top_endpoint_r(det.new_stick, img, half,
                                              quart)
        det.new_orientation = angle_of_line(det.new_stick.line())
        det.new_corrected_endpoint = new_top_endpoint
        det.new_corrected_matches_new = det.new_corrected_endpoint is not None and \
                                        np.linalg.norm(det.new_stick.line()[0] - new_top_endpoint) < 4
        det.new_corrected_matches_new = True
        det.old_corrected_matches_old = False
        process_detection(det)
        assert abs(length_of_line(det.old_stick.line()) - length_of_line(det.new_stick.line())) < 3
        #det.stick_to_use = det.new_stick
        det.valid = True
        X_points.append([int(det.old_stick.bottom[0]), int(det.old_stick.bottom[1])])
        x_points.append([int(det.stick_to_use.bottom[0]), int(det.stick_to_use.bottom[1])])
        os = det.old_stick
        col = [randint(0, 255), randint(0, 255), randint(0, 255)]
        cv.line(dr1, (int(os.top[0]), int(os.top[1])),
                (int(os.bottom[0]), int(os.bottom[1])), col, 3)
        cv.line(dr2, (int(n_stick.top[0]), int(os.top[1])),
                (int(n_stick.bottom[0]), int(os.bottom[1])), col, 3)

    affine_matrix = cv.estimateAffine2D(np.array(X_points), np.array(x_points))
    #cv.imshow('old', cv.resize(dr1, (0,0), fx=0.5, fy=0.5))
    #cv.imshow('new', cv.resize(dr2, (0, 0), fx=0.5, fy=0.5))
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return stick_detections, affine_matrix


def analyze_photos(images: List[str], folder: Path, sticks: List[Stick]) -> Measurement:
    writeout = False
    measurement = Measurement()
    measurement.reason = Reason.Update
    #print(f'process started with {images[0]}')

    sticks_ = sticks
    detections: List[StickDetection] = [StickDetection() for _ in range(len(sticks))]
    detected_sticks: List[StickDetection] = []
    sticks_to_infer: List[StickDetection] = []
    photos_to_process = images[::-1]
    update_step = 50

    for img_name in images[:update_step]:
        if img_name == 'IMAG2950.JPG':
            writeout = True
        if writeout:
            print(img_name)
        bgr = cv.pyrDown(cv.imread(str(folder / img_name)))
        orig = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        measured_sticks: List[Stick] = []
        no_movements: int = 0
        big_movements: int = 0
        detected_sticks.clear()
        sticks_to_infer.clear()

        for i, stick in enumerate(sticks_):
            detection = detections[i]
            detection.reset()
            detection.image = img_name
            detection.old_stick = stick.copy()
            detection.old_stick.view = img_name

            box = line_upright_bbox(detection.old_stick.line(), orig.shape[1], orig.shape[0], width=35)
            if box[1, 0] < 10 or box[0, 0] > orig.shape[0] - 10:
                sticks_to_infer.append(detection)
                continue
            detection.bbox = box
            line_roi = orig[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0]]
            line_roi_half = cv.pyrDown(line_roi)

            detection.roi = line_roi

            lines = list(detect_sticks(line_roi, equalize=True))
            lines2 = list(2 * detect_sticks(line_roi_half, equalize=True))

            lines = np.array(merge_lines_with_same_orientation(lines, max_gap=-1))
            lines2 = np.array(merge_lines_with_same_orientation(lines2, max_gap=-1))

            if len(lines) > 0 and len(lines2) > 0:
                lines = np.array(merge_lines_with_same_orientation(list(np.vstack((lines, lines2))),
                                                                   max_gap=-1))
            else:
                lines = lines if len(lines) > 0 else lines2

            lines = list(lines)
            lines = merge_lines_with_same_orientation(lines)
            #if len(lines) > 0:
            #    lines.sort(key=lambda l: length_of_line(l), reverse=True)
            #    lines = [lines[0]]
            lines = [max(lines, key=lambda l: length_of_line(l), default=None)]
            if len(lines) == 1 and lines[0] is not None:
                old_line = detection.old_stick.line()
                new_line = lines[0] + box[0]

                detection.top_diff = new_line[0] - old_line[0]
                detection.bottom_diff = new_line[1] - old_line[1]

                detection.new_stick = detection.old_stick.copy()
                detection.new_stick.set_top(new_line[0])
                detection.new_stick.set_bottom(new_line[1])

                detection.old_orientation = angle_of_line(detection.old_stick.line())
                detection.new_orientation = angle_of_line(detection.new_stick.line())

                detection.orientation_diff = detection.old_orientation - detection.new_orientation
                detection.orientation_is_same = abs(detection.orientation_diff) < 2

                detection.top_is_same = np.linalg.norm(detection.top_diff) < detection.old_stick.width
                detection.bottom_is_same = np.linalg.norm(detection.bottom_diff) < detection.old_stick.width

                if detection.orientation_is_same:
                    detection.new_lies_on_old = line_to_line_distance(old_line, new_line) < detection.old_stick.width

                detection.top_is_same = np.linalg.norm(detection.top_diff) <= stick.width * 1.0
                detection.bottom_is_same = np.linalg.norm(detection.bottom_diff) < stick.width * 1.0

                bw = box[1, 0] - box[0, 0] - 5
                bh = box[1, 1] - box[0, 1] - 5
                loc_line = lines[0]
                detection.top_out_of_frame = 5 > loc_line[0, 0] or loc_line[0, 0] > bw or loc_line[0, 1] < 5
                detection.bottom_out_of_frame = 5 > loc_line[1, 0] or loc_line[1, 0] > bw or loc_line[1, 1] > bh
                detection.in_frame = 5 < loc_line[0, 0] < bw and 5 < loc_line[1, 0] < bw and 5 < loc_line[0, 1] < bh \
                                     and 5 < loc_line[1, 1] < bh

                if detection.in_frame:
                    no_movements += 1
                else:
                    big_movements += 1
                detected_sticks.append(detection)
            else:
                sticks_to_infer.append(detection)
        no_movements, big_movements = process_detections2(detected_sticks, no_movements, big_movements)

        sticks_to_infer.extend(filter(lambda d: not d.valid, detected_sticks))
        detected_sticks = list(filter(lambda d: d.valid, detected_sticks))

        if len(detected_sticks) < int(0.3 * len(sticks)) or big_movements > no_movements:
            if not is_night(bgr):
                measured_sticks.clear()
                sd, im = handle_big_camera_movement(orig, None, None, detections)
            else:
                #print('is night - skipping')
                sd = None
                im = None
            if sd is None:
                copied_sticks = list(map(lambda s: s.copy(), sticks_))
                for s in copied_sticks:
                    s.is_visible = False
                detected_sticks.clear()
                sticks_to_infer.clear()
                measured_sticks = copied_sticks
            else:
                detected_sticks = list(filter(lambda d: d.valid, sd))
                sticks_to_infer = list(filter(lambda d: not d.valid, sd))
                measurement.reason = Reason.SticksMoved
                measurement.last_valid_sticks = sticks_
                measurement.im = im
        if len(sticks_to_infer) > 0 and len(detected_sticks) > 0:
            # Position sticks that were not found based on the displacement of sticks that were found
            # We take into account that stick movements are caused by rotations of camera around its axis, so
            # sticks that are farther away from camera must be moved by a larger amount than closer sticks
            stick1 = detected_sticks[0]
            dist_from_camera = np.linalg.norm(stick1.old_stick.bottom - np.array([1000, 1500]))
            offset_vec = stick1.stick_to_use.bottom - stick1.old_stick.bottom
            for n_det in sticks_to_infer:
                dist_from_camera_ = np.linalg.norm(n_det.old_stick.bottom - np.array([1000, 1500]))
                factor = dist_from_camera_ / dist_from_camera  # Account for the actual rotational movement of camera
                new_stick = n_det.old_stick.copy()
                new_stick.translate(np.round(offset_vec * factor).astype(np.int32))
                new_stick.is_visible = False
                estimate_snow_height(new_stick, orig)
                measured_sticks.append(new_stick)
        for d in detected_sticks:
            if d.needs_to_measure:
                estimate_snow_height(d.stick_to_use, orig)
            measured_sticks.append(d.stick_to_use)
        assert len(measured_sticks) == len(sticks)
        if measurement.reason == Reason.SticksMoved:
            measurement.sticks_to_confirm = measured_sticks
            measurement.current_img = img_name
            break
        measurement.measurements[img_name] = {'sticks': measured_sticks, 'image_quality': len(detected_sticks) / len(sticks)}
        measurement.last_img = img_name
        photos_to_process.pop()
        sticks_ = measured_sticks
        measurement.last_valid_sticks = sticks_
    photos_to_process.reverse()
    measurement.remaining_photos = photos_to_process
    if len(measurement.remaining_photos) == 0:
        measurement.reason = Reason.FinishedQueue

    return measurement


def process_detections2(detections: List[StickDetection], no_movement: int, big_movement: int) -> Tuple[int, int]:
    for det in detections:
        det.stick_to_use = det.old_stick
        det.needs_to_measure = True
        det.likely_snow_point[0] = -1.0
        if det.top_is_same:
            det.valid = True
            if det.bottom_is_same:
                line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                det.new_stick.set_top(line[0])
                det.new_stick.set_bottom(line[1])
                det.stick_to_use = det.new_stick
                det.stick_to_use.set_snow_height_px(0)
                det.needs_to_measure = False
            else:
                det.likely_snow_point = det.new_stick.bottom
                line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                det.new_stick.set_top(line[0])
                det.new_stick.set_bottom(line[1])
                det.stick_to_use = det.new_stick
                # and perform snow measurement - likely snow point is det.new_stick.bottom
        else:
            if det.bottom_is_same:
                if no_movement > big_movement:
                    line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                    det.new_stick.set_top(line[0])
                    det.new_stick.set_bottom(line[1])
                    det.stick_to_use = det.new_stick
                    det.stick_to_use.set_snow_height_px(0)
                    det.needs_to_measure = False
                    det.valid = True
            else:
                if det.new_lies_on_old:
                    if not det.in_frame:
                        #if np.linalg.norm(det.old_stick.top - det.new_stick.top) < np.linalg.norm(det.old_stick.bottom - det.new_stick.bottom):
                        #    line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                        #else:
                        #    line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                        if det.top_out_of_frame:
                            line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                        else:
                            line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                        det.new_stick.set_top(line[0])
                        det.new_stick.set_bottom(line[1])
                        det.stick_to_use = det.new_stick
                        big_movement -= 1
                        no_movement += 1
                    else:
                        det.stick_to_use = det.old_stick
                    det.valid = True
                    # perform snow measurement
                elif det.in_frame:
                    if det.orientation_is_same:
                        #if np.linalg.norm(det.old_stick.bottom - det.new_stick.bottom) < np.linalg.norm(det.old_stick.top - det.new_stick.top):
                        #    line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                        #else:
                        #    line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                        if det.top_out_of_frame:
                            line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                        else:
                            line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                        det.new_stick.set_top(line[0])
                        det.new_stick.set_bottom(line[1])
                        det.stick_to_use = det.new_stick
                        det.valid = True
                    else:
                        det.valid = False
                else:
                    if det.orientation_is_same:
                        #if np.linalg.norm(det.old_stick.bottom - det.new_stick.bottom) < np.linalg.norm(det.old_stick.top - det.new_stick.top):
                        #    line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                        #else:
                        #    line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                        if det.top_out_of_frame:
                            line = fit_into_length_from_bottom(det.new_stick.line(), det.old_stick.length_px)
                        else:
                            line = fit_into_length(det.new_stick.line(), det.old_stick.length_px)
                        det.new_stick.set_top(line[0])
                        det.new_stick.set_bottom(line[1])
                        det.stick_to_use = det.new_stick
                        det.valid = True
                    else:
                        det.valid = False
    return no_movement, big_movement


def line_to_line_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    line_vec = line_vector(line1)
    v = line2[0] - line1[0]
    dot = np.dot(line_vec, v)
    alpha = math.acos(dot / (np.linalg.norm(v) + 0.0001))
    return math.sin(alpha) * np.linalg.norm(v)


def estimate_snow_height2(stick: Stick, img: np.ndarray):
    sample = skimage.measure.profile_line(img, stick.top[::-1], stick.bottom[::-1], linewidth=5 * stick.width,
                                          reduce_func=None, mode='constant')
    sample = cv.medianBlur(sample.astype(np.uint8), 3)
    sample = cv.equalizeHist(sample)
    sample = cv.morphologyEx(sample, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (stick.width, stick.width)))
    sample_hmt1, sample_hmt2 = apply_multi_hmt(sample.astype(np.uint8), 3, width=stick.width)
    #sample_hmt = np.maximum(sample_hmt[0], sample_hmt[1])
    _, th1 = cv.threshold(sample_hmt1.astype(np.uint8), 9, 1, cv.THRESH_BINARY)
    _, th2 = cv.threshold(sample_hmt2.astype(np.uint8), 9, 1, cv.THRESH_BINARY)
    cl1 = cv.filter2D(th1[:, 2 * stick.width:-2 * stick.width], cv.CV_8U, cv.getStructuringElement(cv.MORPH_RECT, (3, 5)), anchor=(1, 4))
    cl2 = cv.filter2D(th2[:, 2 * stick.width:-2 * stick.width], cv.CV_8U, cv.getStructuringElement(cv.MORPH_RECT, (3, 5)), anchor=(1, 4))
    ko = np.argwhere(cl1[::-1] > 16)
    if ko.shape[0] > 0:
        y = ko[0, 0]
    else:
        y = -1
    cv.imshow('th1', 255 * (cl1 > 16).astype(np.uint8))
    cv.imshow('th2', 255 * (cl2 > 16).astype(np.uint8))
    cv.imshow('eq', sample)
    cv.imshow('fil', 10 * cl1)
    cv.imshow('fil2', 10 * cl2)
    cv.waitKey(0)
    stick.set_snow_height_px(y)


def estimate_snow_height(stick: Stick, img: np.ndarray):
    l = stick.line()
    l = extend_line(l, 30, 'bottom')
    sample = skimage.measure.profile_line(img, l[0, ::-1], l[1, ::-1], linewidth=1 * stick.width,
                                          reduce_func=None, mode='constant')
    sample = sample.astype(np.uint8)
    sample = cv.medianBlur(sample, 3)
    sample = cv.morphologyEx(sample, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    if sample.shape[0] == 0 or sample.shape[1] == 0:
        print('empty sample')

    #integral = cv.integral(sample)
    #inte
    diffs = []
    for i in range(30, sample.shape[0] - 30):
        m1 = np.mean(sample[i-30:i, :])
        m2 = np.mean(sample[i+1:i+31, :])
        diffs.append(abs(np.round(m1 - m2)))
    diff_img = np.reshape(diffs, (-1, 1)).astype(np.uint8)
    #sample = cv.equalizeHist(sample)
    #sample_hmt = apply_multi_hmt(sample.astype(np.uint8), 3, width=stick.width)
    #sample_hmt = np.maximum(sample_hmt[0], sample_hmt[1])
    #_, th = cv.threshold(sample_hmt.astype(np.uint8), 9, 1, cv.THRESH_BINARY)
    #cl = cv.filter2D(th[:, 1 * stick.width:-1 * stick.width], cv.CV_8U, cv.getStructuringElement(cv.MORPH_RECT, (3, 5)),
    #                 anchor=(1, 4))
    #ko = np.argwhere(cl[::-1] > 13)
    #if ko.shape[0] > 0:
    #    y = ko[0, 0]
    #else:
    #    y = -1
    #cv.imshow('th', 255 * (cl > 10).astype(np.uint8))
    #cv.imshow('eq', sample)
    y = stick.length_px - np.argmax(diff_img) - 30
    #cv.imshow('fil', 10 * cv.resize(diff_img, (0, 0), fx=100, fy=5, interpolation=cv.INTER_NEAREST))
    #cv.imshow(stick.label, sample)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    stick.set_snow_height_px(y)


def match_new_sticks_with_old(old_sticks: List[StickDetection], new_sticks: List[np.ndarray]) -> List[Tuple[StickDetection, np.ndarray]]:
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



