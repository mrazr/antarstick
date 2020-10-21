#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
import multiprocessing as mp
import sys
from functools import reduce
from os import scandir
from pathlib import Path
from queue import Queue
from time import time
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import joblib
import numpy as np
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from stick import Stick

STICK_PIPELINE_FILE = Path(sys.argv[0]).parent / 'camera_processing/stick_verification_pipeline4.joblib'

try:
    with open(STICK_PIPELINE_FILE, 'rb') as f:
        stick_pipeline: Pipeline = joblib.load(f)
except FileNotFoundError:
    print(f'Could not load file {STICK_PIPELINE_FILE}') #TODO show error message dialog
    exit(-1)

clahe = cv.createCLAHE()
clahe.setTilesGridSize((8, 8))

FEATURES = ['DIFF_MEAN', 'DIFF_STD', 'LEFT_STD', 'RIGHT_STD', 'LR_MEAN_DIFF', 'I_STD', 'DIFF_FROM_ANGLE', 'LEFT_PER', 'RIGHT_PER', 'LEFT_RIGHT_DIFF']


def show_imgs_(images: List[np.ndarray], names: List[str]):
    for image, name in zip(images, names):
        cv.imshow(name, image)


def measure_snow(img: np.ndarray, sticks: List[Stick]) -> List[Tuple[Stick, float]]:
    """Measures the height of snow in the image `img` around the sticks in `sticks`

    Parameters
    ----------
    img : np.ndarray
        grayscale image
    sticks : List[Stick]
        list of sticks around which to measure the height of snow

    Returns
    -------
    Dict[int, float]
        dictionary of (Stick.id : snow_height_in_pixels)
    """

    blurred = cv.GaussianBlur(img, (3, 3), 1.5)
    measurements = []

    for stick in sticks:
        # Approximate thickness of the stick
        thickness = int(math.ceil(0.03 * stick.length_px))
        end1 = np.array([stick.top[1], stick.top[0]])
        end2 = np.array([stick.bottom[1], stick.bottom[0]])

        # Extract intesity profile underneath the line
        line_profile = skimage.measure.profile_line(blurred, end2, end1, mode='reflect')
        off_x = np.array([0, int(1.5 * thickness)])
        # Now extract intensity profiles left of and right of the stick
        left_neigh_profile = skimage.measure.profile_line(blurred, end2 - off_x, end1 - off_x, mode='reflect')
        right_neigh_profile = skimage.measure.profile_line(blurred, end2 + off_x, end1 + off_x, mode='reflect')

        # Compute the difference of the line profile and the average of the neighboring profiles
        # the idea is that, if there is snow, all the three intensity profiles will be similar enough
        diff = np.abs(line_profile - 0.5 * (left_neigh_profile + right_neigh_profile))
        diff_norm = math.sqrt(np.inner(diff, diff))
        diff_norm = 1.0 / (diff_norm + 0.00001)


        diff = diff_norm * diff

        # Find the indices where the normalized difference is greater than 0.01,
        # this ideally indicates that after seeing snow, we arrived at the stick
        dist_along_stick: np.ndarray = np.argwhere(diff > 0.01)
        if dist_along_stick.shape[0] == 0:
            measurements.append((stick, stick.length_px))
        else:
            # Map `dist_along_stick` to actual height from the ground, which is dependent on the angle of the `stick`
            vec = stick.top - stick.bottom
            vec = vec / np.linalg.norm(vec)

            height = dist_along_stick[0] * np.dot(np.array([0.0, -1.0]), vec)
            measurements.append((stick, height[0]))

    return measurements


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

hmt_selems = [
    [
        dbd_se1,
        dbd_se3,
        dbd_se5,
        dbd_se7
    ],
    [
        bdb_se1,
        bdb_se3,
        bdb_se5,
        bdb_se7
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


def is_night(img: np.ndarray) -> bool:
    g = img[:, :, 1]
    return np.abs(np.mean(img[:, :, 2] - g)) < 10.0 and np.abs(np.mean(img[:, :, 0] - g)) < 10.0


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


def process_batch(batch: List[str], folder: Path, sticks: List[Stick]) -> DataFrame:
    data = {
        'image_name': batch.copy(),
    }

    for stick in sticks:
        data[stick.label + '_top'] = [[0, 0] for _ in range(len(batch))]
        data[stick.label + '_bottom'] = [[0, 0] for _ in range(len(batch))]
        data[stick.label + '_height_px'] = [0 for _ in range(len(batch))]
        data[stick.label + '_snow_height'] = [0 for _ in range(len(batch))]

    for i, img_name in enumerate(batch):
        img = cv.resize(cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE), (0,0), fx=0.25, fy=0.25,
                        interpolation=cv.INTER_LINEAR)
        heights = measure_snow(img, sticks)

        for stick, height in heights:
            data[stick.label + '_top'][i] = list(stick.top)
            data[stick.label + '_bottom'][i] = list(stick.bottom)
            data[stick.label + '_height_px'][i] = stick.length_px
            data[stick.label + '_snow_height'][i] = height
    return DataFrame(data=data)


def load_batch(image_names: List[str], folder: Path, queue: mp.Queue):
    for img_name in image_names:
        img = cv.resize(cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE), (0, 0), fx=0.25, fy=0.25,
                        interpolation=cv.INTER_LINEAR)
        queue.put_nowait((img_name, img))
    queue.put_nowait(None)


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

    for sz in range(3, size+1, 2):
        strel = cv.getStructuringElement(cv.MORPH_RECT, (w, sz))
        c = cv.morphologyEx(c, op1, strel)
        c = cv.morphologyEx(c, op2, strel)
        if op3 is not None:
            c = cv.morphologyEx(c, op3, strel)
    return c


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
    'hough_th': 30,
    'line_length': 30,
    'line_gap': 10,
    'hyst_low': 5,
    'hyst_high': 10,
    'hmt_se_size': 3,
    'f': 0.5,
}


def ground_part(img: np.ndarray) -> np.ndarray:
    return img[int(0.4 * img.shape[0]):int(0.95 * img.shape[0]), int(0.2 * img.shape[1]):int(0.8 * img.shape[1])]


def is_snow(gray: np.ndarray, img: np.ndarray, sigma: float = 0.5, threshold: int = 5) -> bool:
    ground = ground_part(gray)
    bgr = ground_part(img)
    bl = cv.GaussianBlur(ground, (5, 5), sigmaX=sigma)
    diff = ground - bl
    diff_mean = np.array([[np.count_nonzero(diff > threshold) / (ground.shape[0] * ground.shape[1]),
                           np.mean(bgr[:, :, 0])]])
    #return snow_svc.predict(snow_scaler.transform(diff_mean))[0] > 0.0
    return True


def get_rotated_bbox(line: np.ndarray) -> np.ndarray:
    if line[0, 1] > line[1, 1]:
        v = line[0] - line[1]
    else:
        v = line[1] - line[0]
    v = v / np.linalg.norm(v)
    n = v[::-1].copy()

    if np.abs(n[0]) < 0.001:
        n[1] *= -1.0
    else:
        n[0] *= -1.0

    line_ = line.copy()
    line_[0] = line[0] - (30 * v).astype(np.int32)
    line_[1] = line[1] + (30 * v).astype(np.int32)

    points = np.vstack((line_ - 5 * n, line_ + 5 * n)).astype(np.int32)
    rect = cv.minAreaRect(points)
    return rect


def cluster_lines(lines: np.ndarray) -> np.ndarray:
    box_points = np.array(list(map(lambda idx_line: (get_rotated_bbox(idx_line[1]), idx_line[0]), enumerate(lines))))

    for i, line in enumerate(lines):
        bbox_groupid_i = box_points[i]
        bbox_i = bbox_groupid_i[0]
        group_i = bbox_groupid_i[1]

        line_i_len = np.linalg.norm(line[0] - line[1])
        line_i_angle = angle_of_line(line)

        for j in range(i+1, len(lines)):
            line_j = lines[j]
            line_j_len = np.linalg.norm(line_j[0] - line_j[1])
            bbox_groupid_j = box_points[j]
            bbox_j = bbox_groupid_j[0]

            intersection = cv.rotatedRectangleIntersection(bbox_i, bbox_j)
            if intersection[0] > 0:
                line_j_angle = angle_of_line(line_j)
                if abs(line_i_angle - line_j_angle) < 5.0:
                    if line_i_len > line_j_len:
                        src_id = bbox_groupid_j[1]
                        dst_id = group_i
                    else:
                        src_id = group_i
                        dst_id = bbox_groupid_j[1]

                    for id, p_group in enumerate(box_points):
                        group_id = p_group[1]
                        if group_id == src_id:
                            box_points[id] = (p_group[0], dst_id)


    result_lines = []
    finished_groups = [False for _ in range(len(lines))]

    for i, line in enumerate(lines):
        group = box_points[i][1]
        if finished_groups[group]:
            continue
        finished_groups[group] = True
        top = line[0] if line[0,1] < line[1,1] else line[1]
        bottom = line[0] if line[0,1] > line[1,1] else line[1]

        for j, line_j in enumerate(lines):
            group_j = box_points[j][1]
            if group_j != group:
                continue

            top_j = line_j[0] if line_j[0, 1] < line_j[1, 1] else line_j[1]
            bottom_j = line_j[0] if line_j[0, 1] > line_j[1, 1] else line_j[1]

            if top_j[1] < top[1]:
                top = top_j
            if bottom_j[1] > bottom[1]:
                bottom = bottom_j
        result_lines.append(np.array([top, bottom]))

    return np.array(result_lines)


def line_upright_bbox(line: np.ndarray, max_width: int, max_height: int, width = 25) -> np.ndarray:
    w = max(1, int((width - 1) / 2))
    top_left = np.maximum(np.min(line, axis=0) - w, 0)
    bottom_right = np.max(line, axis=0) + w
    bottom_right[0] = np.minimum(bottom_right[0], max_width-1)
    bottom_right[1] = np.minimum(bottom_right[1], max_height-1)
    return np.array([top_left, bottom_right])


def angle_of_line(line: np.ndarray, normal: bool = False) -> float:
    v = line[1] - line[0]
    return (math.degrees(math.atan2(v[1], v[0]))) % 180


def line_vector(line: np.ndarray) -> np.ndarray:
    v = line[1] - line[0]
    return v / (np.linalg.norm(v) + 0.00001)


def line_edge_offsets(line: np.ndarray, mag: np.ndarray, w: int = 19) -> Tuple[List[int], np.ndarray]:
    #bbox = line_upright_bbox(line, 17)
    h_w = (w-1) // 2
    mag_lines = skimage.measure.profile_line(mag, line[0, ::-1], line[1, ::-1], linewidth=w, reduce_func=None,
                                             mode='constant')
    line_sums = np.sum(mag_lines, axis=0)
    left_edge_idx = np.argmax(line_sums[::-1][h_w+1:]) + 1
    right_edge_idx = np.argmax(line_sums[h_w+1:]) + 1

    return [left_edge_idx, right_edge_idx], mag_lines


def detect_sticks(gray: np.ndarray, align_endpoints: bool = False, equalize: bool = True):
    clahe.setClipLimit(5.0)
    if equalize and False:
        gray_eq = clahe.apply(gray)
    else:
        gray_eq = gray

    hmt_dbd1, _ = uhmt(gray_eq, hmt_selems[0][0])
    hmt_dbd3, _ = uhmt(gray_eq, hmt_selems[0][1])
    hmt_dbd5, _ = uhmt(gray_eq, hmt_selems[0][2])
    hmt_dbd7, _ = uhmt(gray_eq, hmt_selems[0][3])
    hmt_bdb1, _ = uhmt(gray_eq, hmt_selems[1][0])
    hmt_bdb3, _ = uhmt(gray_eq, hmt_selems[1][1])
    hmt_bdb5, _ = uhmt(gray_eq, hmt_selems[1][2])
    hmt_bdb7, _ = uhmt(gray_eq, hmt_selems[1][3])

    _, hmt_dbd1_ = cv.threshold(hmt_dbd1, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd3_ = cv.threshold(hmt_dbd3, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd5_ = cv.threshold(hmt_dbd5, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd7_ = cv.threshold(hmt_dbd7, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb1_ = cv.threshold(hmt_bdb1, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb3_ = cv.threshold(hmt_bdb3, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb5_ = cv.threshold(hmt_bdb5, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb7_ = cv.threshold(hmt_bdb7, 9.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)

    th1 = asf(hmt_dbd1_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th2 = asf(hmt_dbd3_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th3 = asf(hmt_bdb1_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th4 = asf(hmt_bdb3_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th5 = asf(hmt_bdb5_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th6 = asf(hmt_dbd5_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th7 = asf(hmt_dbd7_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th8 = asf(hmt_bdb7_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))

    #ths = Parallel(n_jobs=8)(delayed(partial_t)(gray_eq, hmt_selems[i][j]) for i in [0,1] for j in range(len(hmt_selems[0])))
    #th = reduce(np.bitwise_or, ths)
    #return

    th = np.bitwise_or(th1, th2)
    th = np.bitwise_or(th, th3)
    th = np.bitwise_or(th, th4)
    th = np.bitwise_or(th, th5)
    th = np.bitwise_or(th, th6)
    th = np.bitwise_or(th, th7)
    th = np.bitwise_or(th, th8)

    th_ = cv.erode(th, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), anchor=(0,0))
    th_ = cv.dilate(th_, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), anchor=(1,0))

    lines = cv.HoughLinesP(th_, 1.0, np.pi / 180.0, params['hough_th'], None, params['line_length'], params['line_gap'])
    f = 1.0
    f = 1
    if lines is None:
        return []
    lines = np.reshape(lines, (-1, 2, 2))
    for line in lines:
        if line[0,1] > line[1,1]:
            t = line[1].copy()
            line[1] = line[0]
            line[0] = t
    return lines
    start_cluster = time()
    clustered = cluster_lines(lines)
    print(f'clustering took {time() - start_cluster} secs')
    return clustered


def find_horizontal_edges(gray: np.ndarray, bgr: np.ndarray, line: np.ndarray, look_for_top: bool = True, method='hmt') -> Tuple[List[int], List[List[int]]]:
    start_43 = time()
    #dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
    #mag = cv.magnitude(np.zeros(dy.shape, dy.dtype), dy)
    #mag = cv.normalize(mag, None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
    start_line = time()
    mag_line = skimage.measure.profile_line(gray, line[0,::-1], line[1,::-1], linewidth=1, mode='constant',
                                            reduce_func=None).astype(np.uint8)
    print(f'line took {time() - start_line} secs')
    hmt_method = []
    if method == 'hmt' or method == 'both':
        start_hmt = time()
        top_se = np.array([
            [-1],# -1, -1],
            [-1],# -1, -1],
            [-1],# -1, -1],
            [-1],# -1, -1],
            [ 0],#  0,  0],
            [ 0],#  0,  0],
            [ 0],#  0,  0],
            [ 1],#  1,  1],
            [ 1],#  1,  1],
            [ 1],#  1,  1],
            [ 1],  # 1,  1],
            [ 1],  # 1,  1],
            [ 1],  # 1,  1],
            [ 1],  # 1,  1],
            [ 1],  # 1,  1],
            [ 1],  # 1,  1],
            [ 1],  # 1,  1],
        ], dtype=np.int32)
        bottom_se = top_se[::-1].copy()
        #if look_for_top:
        #    mag_hmt = uhmt(mag_line, top_se, anchor=(0,7))[0]
        #else:
        #    mag_hmt = uhmt(mag_line, bottom_se, anchor=(0, 9))[0]
        top_hmt = uhmt(mag_line, top_se, anchor=(0, 7))[0]
        bottom_hmt = uhmt(mag_line, bottom_se, anchor=(0, 9))[0]

        #hmt_method = np.nonzero(mag_hmt > 10)[0]
        top_indices = np.nonzero(top_hmt > 10)[0]
        bottom_indices = np.nonzero(bottom_hmt > 10)[0]
        top_vals = top_hmt[top_indices]
        bottom_vals = bottom_hmt[bottom_indices]
        #hmt_method = np.array([hmt_method, np.reshape(mag_hmt[hmt_method], (-1,))])
        #hmt_method = np.hstack((np.reshape(hmt_method, (-1,1)), vals))
        hmt_method = (np.hstack((np.reshape(top_indices, (-1,1)), top_vals)),
                      np.hstack((np.reshape(bottom_indices, (-1,1)), bottom_vals)))
        print(f'hmt took {time() - start_hmt} secs')
    mean_method = []
    regions = []
    if method == 'mean' or method == 'both':
        start_for = time()
        mag_line = np.reshape(mag_line, (-1,)).astype(np.int32)
        edges = []
        region = [-1, -1]
        regions.append(region)
        sz = 20
        f = 1/float(sz)
        running_sum1 = np.sum(mag_line[:sz]).astype(np.int32)
        running_sum2 = np.sum(mag_line[sz:2*sz]).astype(np.int32)

        #means1 = [f * running_sum2]
        #means2 = [np.mean(mag_line[sz:2*sz])]
        mean_times = 0
        mean_count = 0
        run_times = 0

        if look_for_top:
            if f * (running_sum2 - running_sum1) > 30:
                edges.append([sz, f * (running_sum2 - running_sum1)])
                region[0] = region[1] = sz
        else:
            if f * (running_sum1 - running_sum2) > 30:
                edges.append([sz, f * (running_sum1 - running_sum2)])
                region[0] = region[1] = sz
        for i in range(sz+1, mag_line.shape[0] - sz):
            start = time()
            #abc = np.mean(mag_line[i:i+sz])
            #abc = np.mean(mag_line[i-1-sz:i-1])
            mean_times += (time() - start)
            start = time()
            running_sum1 += mag_line[i-1] - mag_line[i-1-sz]
            running_sum2 += mag_line[i+sz-1] - mag_line[i-1]
            mean1 = running_sum1 * f
            mean2 = running_sum2 * f
            run_times += (time() - start)
            if look_for_top:
                #if f * running_sum2 - f * running_sum1 > 30:
                if mean2 - mean1 > 30:
                    #edges.append([i, f * (running_sum2 - running_sum1)])
                    edges.append([i, mean2 - mean1])
                    if region[0] < 0:
                        region[0] = region[1] = i
                    else:
                        if i - region[1] <= 2:
                            region[1] = i
                        else:
                            region = [i, i]
                            regions.append(region)
            else:
                #if f * running_sum1 - f * running_sum2 > 30:
                if mean1 - mean2 > 30:
                    #edges.append([i, f * (running_sum1 - running_sum2)])
                    edges.append([i, mean1 - mean2])
                    if region[0] < 0:
                        region[0] = region[1] = i
                    else:
                        if i - region[1] <= 2:
                            region[1] = i
                        else:
                            region = [i, i]
                            regions.append(region)
            mean_count += 1
            #diff = np.array(means1) - np.array(means2)
            #dcd = np.sum(diff)
            mean_method = np.array(edges)
            #print(f'np.mean took {mean_times / mean_count} secs and running sums took {run_times / mean_count} secs')
        print(f'mean took {time() - start_for} secs')
    c_time = time()
    print(f'finding took {c_time - start_43} secs or {(1000 * (c_time - start_43))} ms')
    return hmt_method, mean_method, regions


def process_photos_(images: List[str], folder: Path, sticks: List[Stick]) -> Dict[str, Dict[Stick, bool]]:
    results = {}
    #cv.namedWindow('en')
    #cv.namedWindow('roi')
    for img_name in images:
        stick_results = {}
        img_path = str(folder / img_name)
        img = cv.pyrDown(cv.imread(img_path, cv.IMREAD_GRAYSCALE))
        gray_e = img #clahe.apply(img)
        half = cv.pyrDown(gray_e)
        for stick in sticks:
            line = stick.line()
            #vec = line_vector(line)
            top_end_ = get_top_endpoint_r(stick, gray_e, half)
            if top_end_ is not None:
                print(f'old = {stick.top} \tnew = {top_end_}')
                stick.top = top_end_
                stick_results[stick] = True
            else:
                stick_results[stick] = False
            continue
            width = stick.width if stick.width % 2 == 1 else stick.width + 1
            h = 2 * width
            a = 50
            line_e = extend_line(line, a, endpoints='top')
            bbox = line_upright_bbox(line_e, img.shape[1], img.shape[0])
            line_roi = gray_e[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]]
            line_sample = skimage.measure.profile_line(gray_e, line_e[0, ::-1], line_e[1, ::-1],
                                                       linewidth=5 * width, reduce_func=None,
                                                       mode='constant').astype(np.uint8)

            draw = cv.cvtColor(line_sample, cv.COLOR_GRAY2BGR)
            y_off = a
            if width > 7:
                #hmts = 255 * (apply_multi_hmt(cv.pyrDown(line_sample)) > 9).astype(np.uint8)
                dbd, bdb = apply_multi_hmt(cv.pyrDown(line_sample), height=3)
                dbd, bdb = 255 * (dbd > 9).astype(np.uint8), 255 * (bdb > 9).astype(np.uint8)
                draw = cv.pyrDown(draw)
                width = width // 2
                y_off = y_off // 2
            else:
                #hmts = 255 * (apply_multi_hmt(line_sample) > 9).astype(np.uint8)
                dbd, bdb = apply_multi_hmt(line_sample, height=3)
                dbd, bdb = 255 * (dbd > 9).astype(np.uint8), 255 * (bdb > 9).astype(np.uint8)
            hmts = np.bitwise_or(dbd, bdb)
            hmts = asf(hmts, 5, w=1, mode='co')

            x = int(2.5 * width)
            is_endpoint = np.count_nonzero(hmts[y_off:y_off+3, x-width//2:x+width//2 + 1]) >= width
            is_endpoint = is_endpoint and np.count_nonzero(hmts[y_off-3:y_off, x-width//2:x+width//2 + 1]) < width
            
            stick_results[stick] = is_endpoint
            
            for y_ in range(y_off-5, y_off+6):
                for x_ in range(x-width, x+width+1):
                    is_endpoint = np.count_nonzero(hmts[y_:y_ + 3, x_ - width // 2:x_ + width // 2 + 1]) >= width
                    is_endpoint = is_endpoint and np.count_nonzero(hmts[y_ - 3:y_, x_ - width // 2:x_ + width // 2 + 1]) < width
                    if is_endpoint:
                        draw[y_, x_, 1] = (draw[y_, x_, 1] * 0.3 + 0.7 * 255).astype(np.uint8)
                    else:
                        draw[y_, x_, 2] = (draw[y_, x_, 2] * 0.3 + 0.7 * 255).astype(np.uint8)
                        

            #hmts = asf(hmts, 9, w=1, mode='co')
            #draw = cv.cvtColor(hmts, cv.COLOR_GRAY2BGR)
            #y_ = 2 * width + (width + 1) // 2
            #y_ = 2 * h + width + 1
            #for i in range(y_ - width // 1, y_ + width // 1 + 1):
            #    for j in range(x - width // 2, x + width // 2 + 1):
            #        if is_top_corner_at(line_sample, j, i, width):
            #            color = np.array([0, 255, 0])
            #        else:
            #            color = np.array([0, 0, 255])
            #        draw[i - width // 2, j] = (0.5 * draw[i - width // 2, j] + 0.5 * color).astype(np.uint8)
            #cv.circle(draw, ((3 * width + 1) // 2, 2 * width), 5, [0, 255, 0])
            #draw[:, :, 0] = (0.5 * draw[:, :, 0] + 0.5 * hmts).astype(np.uint8)
            #ind = np.nonzero(hmts)
            #draw[ind[0], ind[1], 0] = (0.2 * draw[ind[0], ind[1], 0] + 0.8 * hmts[ind[0], ind[1]]).astype(np.uint8)
            ##cv.circle(draw, (int(1.5 * width), h), 5, [0, 255, 0])
            ##cv.imshow('sample', cv.resize(draw, (0, 0), fx=3, fy=3))
            #cv.imshow('hmts', hmts)
            cv.imshow('dr', cv.resize(draw, (0, 0), fx=4, fy=4))
            cv.waitKey(0)
        results[img_name] = stick_results

    #cv.destroyWindow('hmts')
    #cv.destroyWindow('dr')
    #cv.destroyWindow('en')
    #cv.destroyWindow('roi')
    #cv.destroyAllWindows()
    return results


def apply_multi_hmt(img: np.ndarray, height: int = 1) -> Tuple[np.ndarray]:
    #hmts = [uhmt(img, hmt_selems[i][j])[0] for i in range(2) for j in range(len(hmt_selems[0]))]
    #return reduce(np.bitwise_or, hmts)
    #dbd_hmts = [uhmt(img, selem)[0] for selem in hmt_selems[0][1:]]
    #bdb_hmts = [uhmt(img, selem)[0] for selem in hmt_selems[1][1:]]
    
    dbd_hmts = []
    bdb_hmts = []
    
    for selem in hmt_selems[0]:
        s = selem
        if height > 1:
            s = np.tile(selem, (height, 1))
        print(f's.shape = {s.shape} selem.shape = {selem.shape}')
        dbd_hmts.append(uhmt(img, s, anchor=(selem.shape[1] // 2, 0))[0])
        
    for selem in hmt_selems[1]:
        s = selem
        if height > 1:
            s = np.tile(selem, (height, 1))
        bdb_hmts.append(uhmt(img, s, anchor=(selem.shape[1] // 2, 0))[0])
    
    return reduce(np.bitwise_or, dbd_hmts), reduce(np.bitwise_or, bdb_hmts)


def process_photos(images: List[str], folder: Path, sticks: List[Stick]) -> Dict[str, List[Stick]]:
    measurements: Dict[str, List[Stick]] = {}
    sticks_ = list(map(lambda stick: stick.scale(0.5), sticks))
    loading_time = 0
    for img_name in images:
        start = time()
        img = cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE)
        loading_time += time() - start
        img = cv.pyrDown(cv.resize(img, (0, 0), fx=0.5, fy=0.5))

        measured_sticks: List[Stick] = []

        for stick in sticks_:
            #new_stick = stick.scale(0.5)
            new_stick = stick.scale(1.0)
            new_stick.view = img_name
            old_stick = stick.scale(1.0)
            old_line = np.array([old_stick.top, old_stick.bottom])
            old_line_length = old_stick.length_px
            old_line_angle = angle_of_line(old_line)
            bbox = line_upright_bbox(old_line, img.shape[1], img.shape[0], width=35)
            line_roi = img[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]
            lines = detect_sticks(line_roi, equalize=False)
            # 3 cases here - 0, 1 or multiple lines detected in the stick region
            # 0 lines case - take note and handle after all sticks went through this loop
            # 1 line case - compare angle, endpoints
            # >= 2 lines case - compare angles, find matching endpoints, merge, extend to match stick length

            if len(lines) == 1:
                # 3 cases here:
                # 1) both line's endpoints match stick's endpoints: do nothing
                # 2) one endpoint matches stick's top endpoint:
                    # it's probable that a layer of snow covers the bottom-ish part of the stick.
                    # The detected line might actually represent the visible part of the stick.
                    # To verify this, use find_horizontal_edges and compare the found edge with detected bottom endpoint
                # 2) one endpoint matches stick's bottom endpoint:
                #   a) stick's orientation changed - compare lengths, find top endpoint with `find_horizontal_edges`
                #   b) orientation matches, extend line by finding endpoint with `find_horizontal_edges`
                # 3) no endpoint matches stick's endpoints: might be camera's movement, then all lines will be misaligned.
                #   with their respective sticks. Deal with it after the loop.
                new_line = lines[0] + bbox[0]
                new_line_length = np.linalg.norm(new_line[0] - new_line[1])
                new_line_angle = angle_of_line(new_line)
                new_line_vec = (new_line[1] - new_line[0]) / np.linalg.norm(new_line[1] - new_line[0])
                top_diff = new_line[0] - old_line[0]
                top_diff_mag = np.linalg.norm(top_diff)
                bottom_diff = new_line[1] - old_line[1]
                bottom_diff_mag = np.linalg.norm(bottom_diff)

                if top_diff_mag < 2 and bottom_diff_mag < 2:
                    new_stick.set_endpoints(new_line[0, 0], new_line[0, 1],
                                            new_line[1, 0], new_line[1, 1])
                elif abs(old_line_angle - new_line_angle) < 3:
                    if abs(old_line_length - new_line_length) < 2:
                        new_stick.set_endpoints(new_line[0, 0], new_line[0, 1],
                                                new_line[1, 0], new_line[1, 1])
                    elif top_diff_mag < 2:
                        new_bottom = new_line[0] + stick.length_px * new_line_vec
                    elif bottom_diff_mag < 2:
                        new_top = new_line[1] - stick.length_px * new_line_vec

            measured_sticks.append(new_stick)

        measurements[img_name] = measured_sticks
        sticks_ = measured_sticks
    print(f'loading took {loading_time} secs, or {loading_time / float(len(images))} per image')
    return measurements


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


def extract_line_from_stick(mag_line: np.ndarray, sz: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    #mag_line = skimage.measure.profile_line(gray, line[0, ::-1], line[1, ::-1], linewidth=line_width, mode='constant',
    #                                        reduce_func=None).astype(np.uint8)
    #if len(bounds) == 2:
    #    mag_line = mag_line[:,7-bounds[0]:7+bounds[1]]
    #    mag_line = np.mean(mag_line, axis=1)

    hmt_method = []
    top_se = np.array([
        [-1],# -1, -1],
        [-1],# -1, -1],
        [-1],# -1, -1],
        [-1],# -1, -1],
        [ 0],#  0,  0],
        [ 0],#  0,  0],
        [ 0],#  0,  0],
        [ 1],#  1,  1],
        [ 1],#  1,  1],
        [ 1],#  1,  1],
        [ 1],  # 1,  1],
        [ 1],  # 1,  1],
        [ 1],  # 1,  1],
        [ 1],  # 1,  1],
        [ 1],  # 1,  1],
        [ 1],  # 1,  1],
        [ 1],  # 1,  1],
    ], dtype=np.int32)
    bottom_se = top_se[::-1].copy()
    top_hmt = uhmt(mag_line, top_se, anchor=(0, 7))[0]
    bottom_hmt = uhmt(mag_line, bottom_se, anchor=(0, 9))[0]

    hmt_method = [top_hmt, bottom_hmt]

    mag_line = np.reshape(mag_line, (-1,)).astype(np.int32)

    mean_diffs = np.nan * np.ones((len(mag_line),), np.float32)

    f = 1 / float(sz)

    running_sum1 = np.sum(mag_line[:sz]).astype(np.int32)
    running_sum2 = np.sum(mag_line[sz:2 * sz]).astype(np.int32)

    mean_diffs[sz] = f * (running_sum2 - running_sum1)

    for i in range(sz + 1, mag_line.shape[0] - sz):
        running_sum1 += mag_line[i - 1] - mag_line[i - 1 - sz]
        running_sum2 += mag_line[i + sz - 1] - mag_line[i - 1]
        mean1 = running_sum1 * f
        mean2 = running_sum2 * f
        mean_diffs[i] = mean2 - mean1

    return hmt_method, mean_diffs


def get_endpoint_regions(mean_diffs: np.ndarray) -> List[List[int]]:
    region = [mean_diffs[0], mean_diffs[0]]
    regions = [region]

    for v in mean_diffs:
        if v - region[1] <= 2:
            region[1] = v
        else:
            region = [v, v]
            regions.append(region)

    return regions


def find_sticks(gray: np.ndarray, bgr: np.ndarray, equalize: bool = True) -> List[np.ndarray]:
    vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    start_io = time()
    gray = gray[:-100,:]
    dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
    mag = cv.magnitude(dx, dy)
    angles = cv.phase(dx, dy, None, angleInDegrees=True)
    angles = np.mod(angles - 90, 180)

    lines1 = detect_sticks(gray, equalize=equalize)
    lines1 = cluster_lines(lines1)

    lines05 = detect_sticks(cv.pyrDown(gray), equalize=equalize)
    lines05 = 2 * lines05
    lines05 = cluster_lines(lines05)
    #lines1 = (0.5 * lines1).astype(np.int32)

    if len(lines1) > 0 and len(lines05) > 0:
        lines = cluster_lines(np.vstack((lines1, lines05)))
    else:
        lines = lines1 if len(lines1) > 0 else lines05
    valid_flags: List[bool] = []
    line_samples: List[np.ndarray] = []
    lojns = []

    gray_e = clahe.apply(gray)

    for line in lines:
        line_vec = (line[1] - line[0]).astype(np.float32)
        if np.linalg.norm(line_vec) < 2:
            continue
        line_vec /= np.linalg.norm(line_vec)
        a = 20
        w = 25
        w_h = (w - 1) // 2
        line_ = extend_line(line, amount=a, endpoints='both')
        line_[1,1] = np.minimum(line_[1,1], gray.shape[0]-1)
        bbox = line_upright_bbox(line_, gray.shape[1], gray.shape[0])
        line_roi = gray[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]]

        edge_offsets, mag_line = line_edge_offsets(line_, mag, w)

        left = w_h - edge_offsets[0]
        right = w_h + edge_offsets[1]

        line_angle = angle_of_line(line)
        line_sample = skimage.measure.profile_line(gray, line[0, ::-1], line[1, ::-1], linewidth=w, reduce_func=None,
                                                   mode='constant')
        angle_sample = skimage.measure.profile_line(angles, line[0, ::-1], line[1, ::-1], linewidth=w,
                                                    reduce_func=None, mode='constant')
        extended_line_sample = skimage.measure.profile_line(gray, line_[0, ::-1], line_[1, ::-1], linewidth=w,
                                                            reduce_func=None, mode='constant')

        f_vec = extract_feature_vector(line_sample, None, angle_sample, line_angle, left, right)
        #f_vec2 = extract_features_from_line(gray, line, True)
        valid_flags.append(stick_pipeline.predict([f_vec]) > 0.0)

        #cv.imshow(str(valid_flags[-1]), line_sample)
        #cv.waitKey(0)
        #cv.destroyWindow(str(valid_flags[-1]))

        #print(f'swidth = {edge_offsets[0] + edge_offsets[1]} oh no')

        corrected_endpoints = get_corrected_endpoints(line, extended_line_sample)
        lojns.append((corrected_endpoints if False and len(corrected_endpoints) > 0 else line, valid_flags[-1], int(edge_offsets[0] + edge_offsets[1])))

    #return np.array([lines1, [True] * len(lines1)])
    return lojns
    #return lines1


def get_corrected_endpoints(line: np.ndarray, stick_sample: np.ndarray) -> np.ndarray:
    line_vec = (line[1] - line[0]) / (np.linalg.norm(line[1] - line[0]) + 0.0001)
    line_ = line.copy()
    line_ -= line_[0]
    hmt, mean_diffs = extract_line_from_stick(np.mean(stick_sample, axis=1))

    top_hmt = hmt[0]
    bottom_hmt = hmt[1]

    top_hmt_indices = np.nonzero(top_hmt > 10)[0]
    top_hmt_values = top_hmt[top_hmt_indices]

    bottom_hmt_indices = np.nonzero(bottom_hmt > 10)[0]
    bottom_hmt_values = bottom_hmt[bottom_hmt_indices]

    top_indices = np.nonzero(mean_diffs > 15)[0]
    top_values = mean_diffs[top_indices]

    bottom_indices = np.nonzero(mean_diffs < -15)[0]
    bottom_values = mean_diffs[bottom_indices]

    if len(top_indices) == 0 or len(bottom_indices) == 0:
        return np.zeros((0, 0))

    top_regions = get_endpoint_regions(top_indices)
    bottom_regions = get_endpoint_regions(bottom_indices)

    regions = top_regions.copy()
    regions.extend(bottom_regions)

    for region in top_regions:
        reg = region.copy()
        reg[0] = line_[0, 1] + (reg[0] * line_vec[1]).astype(np.int32)
        reg[1] = line_[0, 1] + (reg[1] * line_vec[1]).astype(np.int32)
        if reg[0] <= line_[0, 1] <= reg[1]:
            # print('offsetting top')
            region_vals = mean_diffs[region[0]:region[1] + 1]
            reg_max = np.argmax(region_vals) + region[0]
            line_[0] = line_[0] + (reg_max * line_vec).astype(np.int32)
        elif reg[0] <= line_[1, 1] <= reg[1]:
            # print('ofsetting bottom')
            region_vals = mean_diffs[region[0]:region[1] + 1]
            reg_max = np.argmax(region_vals) + region[0]
            line_[1] = line_[0] + (reg_max * line_vec).astype(np.int32)

    for region in bottom_regions:
        reg = region.copy()
        reg[0] = line_[0, 1] + (reg[0] * line_vec[1]).astype(np.int32)
        reg[1] = line_[0, 1] + (reg[1] * line_vec[1]).astype(np.int32)
        if reg[0] <= line_[0, 1] <= reg[1]:
            # print('snow offsetting top')
            region_vals = mean_diffs[region[0]:region[1] + 1]
            reg_max = np.argmin(region_vals) + region[0]
            line_[0] = line_[0] + (reg_max * line_vec).astype(np.int32)
        elif reg[0] <= line_[1, 1] <= reg[1]:
            # print('snow ofsetting bottom')
            region_vals = mean_diffs[region[0]:region[1] + 1]
            reg_max = np.argmin(region_vals) + region[0]
            line_[1] = line_[0] + (reg_max * line_vec).astype(np.int32)

    return line_ + line[0]


def extract_feature_vector(gray: np.ndarray, mag: np.ndarray, angle: np.ndarray, line_angle: float, left_edge: int,
                           right_edge: int) -> np.ndarray:
    off = 2
    left_angles = angle[:, max(0, left_edge-off):left_edge+off+1]
    left_angles = np.where(left_angles > 180, left_angles - 180, left_angles)

    right_angles = angle[:, right_edge-off:min(angle.shape[1], right_edge+off+1)]
    right_angles = np.where(right_angles > 180, right_angles - 180, right_angles)

    right_angle_diff = np.abs(right_angles - line_angle)
    right_angle_diff_count = np.count_nonzero(right_angle_diff < 11) / 3.0
    left_angle_diff = np.abs(left_angles - line_angle)
    left_angle_diff_count = np.count_nonzero(left_angle_diff < 11) / 3.0

    diff = np.abs(left_angles[:,1] - right_angles[:,1])
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


def get_top_endpoint_r(stick: Stick, full: np.ndarray, half: np.ndarray) -> Optional[np.ndarray]:
    st = time()
    scale = 1
    width = stick.width
    stick_ = stick.scale(1.0)
    img = full
    if width > 7:
        width = int(round(0.5 * width))
        if width % 2 == 0:
            width += 1
        stick_ = stick.scale(0.5)
        img = half
    elif width < 3:
        width = 3
    amount = 50
    line = stick_.line()
    line_e = extend_line(line, amount, endpoints='top')
    line_sample = skimage.measure.profile_line(img, line_e[0, ::-1], line_e[1, ::-1],
                                               linewidth=5 * width, reduce_func=None,
                                               mode='constant').astype(np.uint8)
    dbd, bdb = apply_multi_hmt(line_sample, height=3)
    dbd, bdb = 1 * (dbd > 9).astype(np.uint8), 1 * (bdb > 9).astype(np.uint8)

    hmts = np.bitwise_or(dbd, bdb)
    hmts = asf(hmts, 5, w=1, mode='co')

    angle = math.atan2(*(line[1]-line[0])[::-1]) - 0.5 * np.pi

    coords: List[List[int]] = []
    zeros = np.zeros((2, width), np.uint8)
    kern_bot = np.ones((1, width), np.uint8)
    kern_bot = np.tile(kern_bot, (3, 1))
    kern_bot = np.vstack((zeros, kern_bot))

    kern_top = kern_bot[::-1, :]

    fil_bot = cv.filter2D(hmts, cv.CV_8U, kern_bot, None, borderType=cv.BORDER_CONSTANT)
    fil_top = cv.filter2D(hmts, cv.CV_8U, kern_top, None, borderType=cv.BORDER_CONSTANT, anchor=(width // 2, 3))

    y_off = amount
    x = int(round(2.5 * width))

    for y_ in range(y_off-5, y_off+6):
        for x_ in range(x - 2 * width, x + 2 * width+1):
            #is_endpoint = np.count_nonzero(hmts[y_:y_ + 3, x_ - width // 2:x_ + width // 2 + 1]) >= 1.5 * width
            #is_endpoint = is_endpoint and np.count_nonzero(hmts[y_ - 3:y_, x_ - width // 2:x_ + width // 2 + 1]) < width
            is_e2 = fil_bot[y_, x_] >= 1.5 * width
            is_e2 = is_e2 and fil_top[y_, x_] < width
            if is_e2:
                coords.append([x_, y_])

    if len(coords) > 0:
        coords = np.array(coords)
        mean = np.round(coords.mean(axis=0)).astype(np.int32)
        #bbox = line_upright_bbox(line, img.shape[1], img.shape[0])

        scale = 1
        line_orig = stick.line()
        new_top = coords[np.abs(coords - mean).sum(axis=1).argmin()] - np.array([x, y_off], np.int32)
        glob_x = np.round(scale * (new_top[0] * math.cos(angle) - new_top[1] * math.sin(angle)) + line_orig[0, 0])
        glob_y = np.round(scale * (new_top[0] * math.sin(angle) + new_top[1] * math.cos(angle)) + line_orig[0, 1])
        result = np.array([glob_x, glob_y], np.int32)
    else:
        result = None

    print(f'end took {time() - st} secs')
    return result


def get_top_endpoint(stick: Stick, img: np.ndarray, half: np.ndarray) -> Optional[np.ndarray]:
    st = time()
    line = stick.line()
    amount = 50
    line_e = extend_line(line, amount=amount, endpoints='top')
    width = stick.width
    line_sample = skimage.measure.profile_line(img, line_e[0, ::-1], line_e[1, ::-1],
                                                linewidth=5 * width, reduce_func=None,
                                                mode='constant').astype(np.uint8)
    y_off = amount
    scale = 1
    if width > 7:
        #hmts = 255 * (apply_multi_hmt(cv.pyrDown(line_sample)) > 9).astype(np.uint8)
        dbd, bdb = apply_multi_hmt(cv.pyrDown(line_sample), height=3)
        dbd, bdb = 255 * (dbd > 9).astype(np.uint8), 255 * (bdb > 9).astype(np.uint8)
        #draw = cv.pyrDown(draw)
        width = int(round(0.5 * width))
        if width % 2 == 0:
            width += 1
        y_off = y_off // 2
        scale = 2
    else:
        #hmts = 255 * (apply_multi_hmt(line_sample) > 9).astype(np.uint8)
        dbd, bdb = apply_multi_hmt(line_sample, height=3)
        dbd, bdb = 255 * (dbd > 9).astype(np.uint8), 255 * (bdb > 9).astype(np.uint8)
    hmts = np.bitwise_or(dbd, bdb)
    hmts = asf(hmts, 5, w=1, mode='co')

    x = int(round(2.5 * width))
   # is_endpoint = np.count_nonzero(hmts[y_off:y_off+3, x-width//2:x+width//2 + 1]) >= width
   # is_endpoint = is_endpoint and np.count_nonzero(hmts[y_off-3:y_off, x-width//2:x+width//2 + 1]) < width
   # 
   # stick_results[stick] = is_endpoint

    angle = math.atan2(*(line[1]-line[0])[::-1]) - 0.5 * np.pi

    draw = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    coords: List[List[int]] = []
    zeros = np.zeros((2, width), np.uint8)
    kern_bot = np.ones((1, width), np.uint8)
    kern_bot = np.tile(kern_bot, (3, 1))
    kern_bot = np.vstack((zeros, kern_bot))

    kern_top = kern_bot[::-1,:]

    fil_bot = cv.filter2D(hmts, cv.CV_8U, kern_bot, borderType=cv.BORDER_CONSTANT)
    fil_top = cv.filter2D(hmts, cv.CV_8U, kern_top, borderType=cv.BORDER_CONSTANT)


    for y_ in range(y_off-5, y_off+6):
        for x_ in range(x - 2 * width, x + 2 * width+1):
            is_endpoint = np.count_nonzero(hmts[y_:y_ + 3, x_ - width // 2:x_ + width // 2 + 1]) >= 1.5 * width
            is_endpoint = is_endpoint and np.count_nonzero(hmts[y_ - 3:y_, x_ - width // 2:x_ + width // 2 + 1]) < width
            is_e2 = fil_bot[y_, x_] >= 1.5 * width
            is_e2 = is_e2 and fil_top[y_, x_] < width
            if is_endpoint:
                #draw[y_, x_, 1] = (draw[y_, x_, 1] * 0.3 + 0.7 * 255).astype(np.uint8)
                coords.append([x_, y_])
                color = np.array([0, 255, 0])
            else:
                color = np.array([0, 0, 255])
            x__ = x_ - x
            y__ = y_ - y_off
            glob_x_ = int(round(x__ * math.cos(angle) - y__ * math.sin(angle) + line[0, 0]))
            glob_y_ = int(round(x__ * math.sin(angle) + y__ * math.cos(angle) + line[0, 1]))
            draw[glob_y_, glob_x_] = (0.3 * draw[glob_y_, glob_x_] + 0.7 * color).astype(np.uint8)

    if len(coords) > 0:
        coords = np.array(coords)
        mean = np.round(coords.mean(axis=0)).astype(np.int32)

        bbox = line_upright_bbox(line, img.shape[1], img.shape[0])
        roi = draw[bbox[0,1]:bbox[1,1], bbox[0,0]:bbox[1,0]].copy()
        #cv.imshow('roi', cv.resize(roi, (0, 0), fx=4, fy=4))

        scale = 1
        new_top = coords[np.abs(coords - mean).sum(axis=1).argmin()] - np.array([x, y_off], np.int32)
        glob_x = np.round(scale * (new_top[0] * math.cos(angle) - new_top[1] * math.sin(angle)) + line[0, 0])
        glob_y = np.round(scale * (new_top[0] * math.sin(angle) + new_top[1] * math.cos(angle)) + line[0, 1])
        #glob_x *= scale
        #glob_y *= scale
        draw[int(glob_y), int(glob_x)] = (0.3 * draw[int(glob_y), int(glob_x)] + 0.7 * np.array([255, 0, 0])).astype(np.uint8)
        result = np.array([glob_x, glob_y], np.int32)
    else:
        result = None

    print(f'end took {time() - st} secs')
    #cv.imshow('en', draw)
    #cv.waitKey(0)
    return result

