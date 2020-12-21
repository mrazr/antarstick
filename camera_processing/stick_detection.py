#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
import sys
from functools import reduce
from pathlib import Path
from random import randint
from typing import List, Optional, Tuple, Union

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


def show_imgs_(images: List[np.ndarray], names: List[str]):
    for image, name in zip(images, names):
        cv.imshow(name, image)


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


def line_upright_bbox(line: np.ndarray, max_width: int, max_height: int, width=25, out_of_bounds: bool=False, left_range: int = -1, right_range: int = -1) -> np.ndarray:
    w = max(1, int((width - 1) / 2))
    lo = left_range if left_range > 0 else w
    ro = right_range if right_range > 0 else w
    if not out_of_bounds:
        top_left = np.maximum(np.min(line, axis=0) - [lo, w], 0)
    else:
        top_left = np.min(line, axis=0) - [lo, w]
    bottom_right = np.max(line, axis=0) + [ro, w]
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


def detect_sticks(gray: np.ndarray, align_endpoints: bool = False, equalize: bool = True, h: int = 9):
    clahe.setClipLimit(5.0)
    if equalize and False:
        gray_eq = clahe.apply(gray)
    else:
        gray_eq = gray
    #gray_eq = cv.convertScaleAbs(gray_eq, None, 0.2)
    #gray_eq = gray
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
        th1 = asf(hmt_dbd1_, h, 1,
                  'co')  # cv.morphologyEx(hmt_dbd1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th2 = asf(hmt_dbd3_, h, 1,
                  'co')  # cv.morphologyEx(hmt_dbd3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th3 = asf(hmt_bdb1_, h, 1,
                  'co')  # cv.morphologyEx(hmt_dbd5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th4 = asf(hmt_bdb3_, h, 1,
                  'co')  # cv.morphologyEx(hmt_dbd7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th5 = asf(hmt_bdb5_, h, 1,
                  'co')  # cv.morphologyEx(hmt_bdb1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th6 = asf(hmt_dbd5_, h, 1,
                  'co')  # cv.morphologyEx(hmt_bdb3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th7 = asf(hmt_dbd7_, h, 1,
                  'co')  # cv.morphologyEx(hmt_bdb5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))
        th8 = asf(hmt_bdb7_, h, 1,
                  'co')  # cv.morphologyEx(hmt_bdb7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, h)))

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
        #assert left > 0
        right = w_h + edge_offsets[1]

        line_angle = angle_of_line(line)
        line_sample = skimage.measure.profile_line(gray, line[0, ::-1], line[1, ::-1], linewidth=w, reduce_func=None,
                                                   mode='constant')
        angle_sample = skimage.measure.profile_line(angles, line[0, ::-1], line[1, ::-1], linewidth=w,
                                                    reduce_func=None, mode='constant')

        f_vec = extract_feature_vector(line_sample, None, angle_sample, line_angle, left, right)
        valid_flags.append(stick_pipeline.predict([f_vec]) > 0.0)

        final_lines.append((line, valid_flags[-1], int(edge_offsets[0] + edge_offsets[1])))

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


def line_to_line_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    line_vec = line_vector(line1)
    v = line2[0] - line1[0]
    dot = np.dot(line_vec, v)
    alpha = math.acos(dot / (np.linalg.norm(v) + 0.0001))
    return math.sin(alpha) * np.linalg.norm(v)

