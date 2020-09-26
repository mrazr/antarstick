#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
from os import scandir
from pathlib import Path
from queue import Queue, PriorityQueue
from time import time, sleep
from typing import Dict, List, Optional, Tuple, Set
import sys
import multiprocessing as mp
from random import randint
import pickle

import cv2 as cv
import numpy as np
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.segmentation
from skimage.color import label2rgb
from PyQt5.QtCore import QThreadPool
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
import skimage.measure
from pandas import DataFrame
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
from math import comb
import statistics
from random import randrange
import matplotlib.pyplot as plt
import pandas as pd

from networkx import Graph
import networkx.algorithms.bipartite.matching as matching

from my_thread_worker import MyThreadWorker
from stick import Stick

Area = float
Height = float
Ecc = float
Label = int
Centroid = Tuple[int, int]

Rectangle = List[int]

STICK_WINDOW = (64, 128) #(32, 64) #(64, 128)
STICK_WINDOW = (32, 64)
ENDPOINT_WINDOW = (48, 48)

#hog_desc = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9, 1, -1, cv.HOGDescriptor_L2Hys, 0.2, False,
#                            cv.HOGDESCRIPTOR_DEFAULT_NLEVELS, True)
hog_desc = cv.HOGDescriptor(STICK_WINDOW, (16, 16), (8, 8), (8, 8), 18, 1, -1, cv.HOGDescriptor_L2Hys, 0.2, True,
                            cv.HOGDESCRIPTOR_DEFAULT_NLEVELS, True)
#STICK_HOG_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/svc2222')
STICK_HOG_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/stick_hog_32x64_gamma_18b')
SNOW_SVC_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/snow_svc.joblib')
SNOW_SCALER_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/snow_scaler.joblib')
STICK_PIPELINE_FILE = Path(sys.argv[0]).parent / 'camera_processing/stick_verification_pipeline4.joblib'

try:
    success = hog_desc.load(STICK_HOG_FILE)
except:
    print(f'Could not load file {STICK_HOG_FILE}')
    exit(-1)

try:
    with open(STICK_PIPELINE_FILE, 'rb') as f:
        stick_pipeline: Pipeline = joblib.load(f)
except FileNotFoundError:
    print(f'Could not load file {STICK_PIPELINE_FILE}') #TODO show error message dialog
    exit(-1)

snow_svc: SVC = joblib.load(SNOW_SVC_FILE)
snow_scaler: StandardScaler = joblib.load(SNOW_SCALER_FILE)

clahe = cv.createCLAHE()
clahe.setTilesGridSize((8, 8))

def denoise(img: np.ndarray) -> np.ndarray:
    """Denoises image by perfoming cv.pyrDown twice and then cv.pyrUp twice.

    Parameters
    ----------
    img : np.ndarray
        image

    Returns
    -------
    np.ndarray
    """

    down = cv.pyrDown(img)
    # down = cv.pyrDown(down)
    # down = cv.pyrUp(down)
    return cv.pyrUp(down)

def height_of_region(region_prop) -> int:
    return region_prop.bbox[2] - region_prop.bbox[0]

def draw_lines_on_img(img, lines):
    for line in lines:
        cv.line(img, tuple(line[:2]), tuple(line[2:]), [255, 255, 0], 2)


def bbox_contains(bbox: Tuple[int, int, int, int], point: Tuple[int, int]) -> bool:
    return bbox[1] <= point[0] < bbox[3] and bbox[0] <= point[1] < bbox[2]


def show_imgs_(images: List[np.ndarray], names: List[str]) -> int:
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

def get_angle_image(img: np.ndarray) -> np.ndarray:
    dx = cv.Sobel(img, cv.CV_32F, 1, 0)
    dy = cv.Sobel(img, cv.CV_32F, 0, 1)
    return cv.phase(dx, dy, angleInDegrees=True)

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

def uhmt(img: np.ndarray, se: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e_se = (1 * (se == 1)).astype(np.uint8)
    d_se = (1 * (se == 0)).astype(np.uint8)
    e = cv.erode(img, e_se, borderType=cv.BORDER_REPLICATE)
    d = cv.dilate(img, d_se, borderType=cv.BORDER_REPLICATE)
    mask = e > d
    diff = (e.astype(np.int16) - d.astype(np.int16))
    diff[diff < 0] = 0
    return cv.convertScaleAbs(diff), (1 * mask).astype(np.uint8)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = denoise(img)
    #return img_as_ubyte(skimage.exposure.equalize_adapthist(
    #img, [int(img.shape[0] / 20.0), int(img.shape[1] / 20.0)]))
    return clahe.apply(img)


def get_non_snow_images_(path: Path, count: int = 1) -> Optional[List[np.ndarray]]:
    image_list: List[np.ndarray] = []

    for file in scandir(path):
        if file.name[-3:].lower() != "jpg":  # TODO handle JPEG
            continue
        img = cv.imread(str(file.path))[:-50, :, :]
        img = cv.pyrDown(img)
        if is_night(img):
            continue
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        if is_non_snow(hsv):
            image_list.append(img)
            if len(image_list) == count:
                return image_list

    return None

def compute_line_theta(line: List[List[int]]) -> float:
    p = np.array(line[0])
    q = np.array(line[1])
    u = (p - q) * (1.0 / np.linalg.norm(p - q))
    v = u * u.dot(-p)
    r_v = -(-p - v)
    r = np.linalg.norm(r_v)
    r_v = r_v * (1.0 / np.linalg.norm(r_v))
    theta = 90.0 + (np.math.acos(np.array([1.0, 0.0]).dot(r_v)) / np.pi) * 180.0

    return theta

def is_night(img: np.ndarray) -> bool:
    g = img[:, :, 1]
    return np.abs(np.mean(img[:, :, 2] - g)) < 10.0 and np.abs(np.mean(img[:, :, 0] - g)) < 10.0

def update_stick_heat_map(stick_img: np.ndarray, heat_map: np.ndarray, stick_lengths: np.ndarray) -> List[np.ndarray]:
    """
    Updates stick heat map `heat_map` with the according to line structures in `stick_img` and also updates
    the image `stick_lengths`.
    For every connected component (ideally line) in `stick_img` the following is performed:
        1. compute its bounding box
        2. increment values in `heat_map` located by the bounding box
        3. perform point-wise maximum of the bounding box height and the values in `stick_lengths` located by the
           bounding box
    :param stick_img: np.ndarray: preprocessed image retrieved from `preprocess_phase` method
    :param heat_map: np.ndarray: heat map specifying the likelihood of a stick at a certain position. Same shape as stick_img.
    :param stick_lengths: np.ndarray: maximum length of sticks located at certain position. Same shape as stick_img.
    :return: List[np.ndarray]: `heat_map`, `stick_lengths`, stats of connected components of `stick_img` retrieved by
            cv2.connectedComponentsWithStats
    """
    n, _, stick_img_stats, _ = cv.connectedComponentsWithStats(stick_img)
    for i in range(1, n):
        x = stick_img_stats[i][cv.CC_STAT_LEFT] - 1
        y = stick_img_stats[i][cv.CC_STAT_TOP] - 1
        w = stick_img_stats[i][cv.CC_STAT_WIDTH] + 1
        h = stick_img_stats[i][cv.CC_STAT_HEIGHT] + 1
        heat_map[y:y + h, x:x + w] += 1
        stick_lengths[y:y + h, x:x + w] = np.maximum(stick_lengths[y:y + h, x:x + w], h)

    return [heat_map, stick_lengths, stick_img_stats]


def get_non_snow_images(path: Path, queue: Queue, count: int = 9) -> None:
    nights = 0
    images_loaded = 0
    for file in scandir(path):
        if file.name[-3:].lower() != "jpg":  # TODO handle JPEG
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

def get_sticks_in_folder(folder: Path) -> Optional[Tuple[List[np.ndarray], Path, float]]:
    queue = Queue(maxsize=10)
    # thread = threading.Thread(target=get_non_snow_images, args=(folder, queue, 9))
    worker = MyThreadWorker(task=get_non_snow_images, args=(folder, queue, 9), kwargs={})

    heat_map = None
    stick_lengths = None
    imgs: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    QThreadPool.globalInstance().start(worker)

    loading_time = 0
    loading_start = time()
    path_img: Tuple[Path, np.ndarray] = queue.get()
    angles = []
    while path_img is not None:
        loading_time += (time() - loading_start)
        img = path_img[1]
        start = time()
        gray = cv.cvtColor(cv.pyrDown(img), cv.COLOR_BGR2GRAY)[:-50]
        #gray = cv.pyrUp(cv.pyrDown(gray))[:-50]
        gray_ = cv.GaussianBlur(gray, (3, 3), 1.2)
        gray = cv.copyMakeBorder(gray_, 15, 15, 15, 15, cv.BORDER_REPLICATE)
        start2 = time()
        prep, _ = preprocess_phase(gray)
        print(f'prepping took {time() - start2} secs')
        prep = (255 * prep).astype(np.uint8)
        if heat_map is None:
            heat_map = np.zeros(prep.shape, dtype=np.uint8)
            stick_lengths = np.zeros(prep.shape, dtype=np.uint16)
        start2 = time()
        heat_map, stick_lengths, img_stats = update_stick_heat_map(prep, heat_map, stick_lengths)
        print(f'heatmapping took {time() - start2} secs')
        print(f'took {time() - start} secs')
        imgs.append((Path(path_img[0]), img, prep, img_stats))
        #dx, dy = cv.Sobel(gray_, cv.CV_32F, 1, 0), cv.Sobel(gray_, cv.CV_32F, 0, 1)
        #angle = cv.phase(dx, dy, None, True)
        #angle = cv.inRange(angle, 155, 205)
        #cv.imshow('angle', gray)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        #angles.append(angle)
        loading_start = time()
        path_img = queue.get()

    if len(imgs) > 0:
        loading_time = loading_time / len(imgs)

    if stick_lengths is None or np.max(stick_lengths) < 0.01:
        return None

    stick_lengths = stick_lengths / np.max(stick_lengths)
    heat_map = heat_map * (1 + stick_lengths)
    _, stick_regions = cv.threshold(heat_map.astype(np.uint8), len(imgs) - 3, 255, cv.THRESH_BINARY)

    stick_scores = []
    sticks_list: List[List[np.ndarray]] = []
    for j in range(len(imgs)):
        sticks_ = filter_sticks(imgs[j][2], imgs[j][3], stick_regions)
        sticks_list.append(sticks_)
        stick_score = sum((map(lambda s: np.linalg.norm(s[0] - s[1]), sticks_)))
        stick_scores.append([stick_score, len(sticks_), j])

    # Sort according to number of sticks and total stick lengths
    stick_scores = sorted(stick_scores, key=lambda c: (c[1], c[0]), reverse=True)

    # Most likely stick configuration is identified by the index which sits in stick_scores[0][2]
    idx = stick_scores[0][2]

    #adjusted_sticks = []
    #for stick in sticks_list[idx]:
    #    stick_ = adjust_endpoints(imgs[idx][1], stick, 9)
    #    adjusted_sticks.append(stick_)


    return sticks_list[idx], imgs[idx][0], 20.0, stick_regions
    #return adjusted_sticks, imgs[idx][0], loading_time


def process_batch(batch: List[str], folder: Path, sticks: List[Stick]) -> DataFrame:
    data = {
        'image_name': batch.copy(),
    }

    for stick in sticks:
        data[stick.label + '_top'] = [[0, 0] for _ in range(len(batch))]
        data[stick.label + '_bottom'] = [[0, 0] for _ in range(len(batch))]
        data[stick.label + '_height_px'] = [0 for _ in range(len(batch))]
        data[stick.label + '_snow_height'] = [0 for _ in range(len(batch))]

    #img_queue = mp.Queue(maxsize=len(batch) + 1)
    #process = mp.Process(target=load_batch, args=(batch, folder, img_queue))
    #process.start()

    #queue_item = img_queue.get()
    #i = 0
    #start = time()
    #while queue_item is not None:
    for i, img_name in enumerate(batch):
        #start = time()
        img = cv.resize(cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE), (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
        #loading_time += (time() - start)
        #start = time()
        heights = measure_snow(img, sticks)
        #measuring_time += (time() - start)

        for stick, height in heights:
            data[stick.label + '_top'][i] = list(stick.top)
            data[stick.label + '_bottom'][i] = list(stick.bottom)
            data[stick.label + '_height_px'][i] = stick.length_px
            data[stick.label + '_snow_height'][i] = height
    return DataFrame(data=data)


def load_batch(image_names: List[str], folder: Path, queue: mp.Queue):
    print('hello')
    for img_name in image_names:
        print(f'loading {img_name}')
        img = cv.resize(cv.imread(str(folder / img_name), cv.IMREAD_GRAYSCALE), (0,0), fx=0.25, fy=0.25, interpolation=cv.INTER_LINEAR)
        print(f'shape = {img.shape}')
        queue.put_nowait((img_name, img))
    queue.put_nowait(None)


def get_values_at(img: np.ndarray, x: int, y: int, s: int) -> List[np.ndarray]:
    """Extracts mean values for ROI-s in `img`.
       The ROI-s are windows of size `s` by `s` pixels and they are
       arranged in a cross shape centered on the pixel (x,y).
    """
    s_ = s // 2

    c = np.mean(img[y - s_:y + s_, x - s_:x + s_])  # mean for central ROI
    t = np.mean(img[y - 3 * s_:y - s_, x - s_:x + s_])  # mean for top ROI (above central ROI)
    b = np.mean(img[y + s_:y + 3 * s_, x - s_:x + s_])  # mean for bottom ROI (below central ROI)
    l = np.mean(img[y - s_:y + s_, x - 3 * s_:x - s_])  # mean for left ROI (left of central ROI)
    r = np.mean(img[y - s_:y + s_, x + s_:x + 3 * s_])  # mean for right ROI (right of central ROI)

    return [c, t, r, b, l]


def get_sticks_in_folder_non_mp(folder: Path) -> Tuple[List[np.ndarray], Path]:
    #queue = Queue(maxsize=10)
    ## thread = threading.Thread(target=get_non_snow_images, args=(folder, queue, 9))
    #worker = MyThreadWorker(task=get_non_snow_images, args=(folder, queue, 9), kwargs={})

    heat_map = None
    stick_lengths = None
    imgs: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    #QThreadPool.globalInstance().start(worker)
    queue = Queue(maxsize=10)

    get_non_snow_images(folder, queue, 9)

    path_img: Tuple[Path, np.ndarray] = queue.get()
    angles = []
    while path_img is not None:
        img = path_img[1]
        gray = cv.cvtColor(cv.pyrDown(img), cv.COLOR_BGR2GRAY)
        gray = cv.pyrUp(cv.pyrDown(gray))[:-50]
        gray_ = cv.copyMakeBorder(gray, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=0)
        prep, _ = preprocess_phase(gray_)
        prep = (255 * prep).astype(np.uint8)
        if heat_map is None:
            heat_map = np.zeros(prep.shape, dtype=np.uint8)
            stick_lengths = np.zeros(prep.shape, dtype=np.uint16)
        heat_map, stick_lengths, img_stats = update_stick_heat_map(prep, heat_map, stick_lengths)
        imgs.append((path_img[0], img, prep, img_stats))
        dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
        angle = cv.phase(dx, dy, None, True)
        angle = cv.inRange(angle, 155, 205)
        cv.imshow('angle', angle)
        cv.waitKey(0)
        cv.destroyAllWindows()
        angles.append(angle)
        imgs.append((path_img[0], img, prep, img_stats))
        path_img = queue.get()

    if stick_lengths is None or np.max(stick_lengths) < 0.01:
        return None
    stick_lengths = stick_lengths / np.max(stick_lengths)
    heat_map = heat_map * (1 + stick_lengths)
    _, stick_regions = cv.threshold(heat_map.astype(np.uint8), len(imgs) - 3, 255, cv.THRESH_BINARY)

    stick_scores = []
    sticks_list: List[List[np.ndarray]] = []
    for j in range(len(imgs)):
        sticks_ = filter_sticks(imgs[j][2], imgs[j][3], stick_regions)
        sticks_list.append(sticks_)
        stick_score = sum((map(lambda s: np.linalg.norm(s[0] - s[1]), sticks_)))
        stick_scores.append([stick_score, len(sticks_), j])

    # Sort according to number of sticks and total stick lengths
    stick_scores = sorted(stick_scores, key=lambda c: (c[1], c[0]), reverse=True)

    # Most likely stick configuration is identified by the index which sits in stick_scores[0][2]
    idx = stick_scores[0][2]

    #adjusted_sticks = []
    #for stick in sticks_list[idx]:
    #    stick_ = adjust_endpoints(imgs[idx][1], stick, 9)
    #    adjusted_sticks.append(stick_)


    return sticks_list[idx], imgs[idx][0]
    #return adjusted_sticks, imgs[idx][0]

def iou(box1: List[int], box2: List[int]) -> float:
    """Computes the Intersection-Over-Union of two rectangles.

    Parameters
    ---------
    box1, box2: List[int]
        rectangle in the form (left, top, width, height)
    """
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


def merge_detection_boxes(boxes: List[List[int]], scores: List[float]) -> List[List[int]]:
    """
    Merges overlapping detection boxes. Uses the idea of Non-max suppresion in that it also utilizes
    the Intersection-Over-Union criterion for overlapping rectangles, but rather than throwing away overlapping
    rectangles with non-max score, we group them and finally merge them into a single rectangle.
    This gives a comfortable detection box for sticks.

    :param boxes: list of rectangles of the form (left, top, width, height)
    :param scores: list of scores for rectangles from `boxes`
    :return: list of merged rectangles
    """
    # Priority queue prioritized by each rectangle's score
    queue = PriorityQueue(maxsize=len(scores))

    for box_idx, priority in enumerate(scores):
        queue.put((-1.0 * priority, box_idx))

    # Each rectangle, identified by its index in `boxes` will have in `groups` a set of rectangles which are
    # suitable to merge with it
    groups: Dict[int, Set[int]] = {}
    unprocessed_box_ids: Set[int] = {i for i in range(len(boxes))}

    while not queue.empty():
        score, box_idx = queue.get()
        if box_idx not in unprocessed_box_ids:
            continue

        box = boxes[box_idx]
        groups[box_idx] = set({})

        idx_to_remove: List[int] = []
        for other_box_idx in unprocessed_box_ids:
            if other_box_idx == box_idx:
                continue

            other_box = boxes[other_box_idx]
            boxes_iou = iou(box, other_box)

            # This is where we differ in NMS, in NMS `boxes[other_box_idx]` would get thrown away
            # In our case it is added to `boxes[box_idx]`'s group of mergeable rectangles
            if boxes_iou > 0.55:
                groups[box_idx].add(other_box_idx)
        for idx in idx_to_remove:
            unprocessed_box_ids.remove(idx)

    final_rects: List[Tuple[Rectangle, List[Rectangle]]] = []

    merged_idx: Set[int] = set({})

    for box_idx, group in groups.items():

        if len(group) < 4 or box_idx in merged_idx:
            continue

        merged_idx.add(box_idx)
        grouped_boxes: Set[int] = set({box_idx})
        to_process = [idx for idx in group]

        while len(to_process) > 0:
            other_box_idx = to_process.pop()

            if other_box_idx in grouped_boxes:
                continue

            merged_idx.add(other_box_idx)
            grouped_boxes.add(other_box_idx)
            to_process.extend([idx for idx in groups[other_box_idx]])

        left = 9000
        right = -9000
        top = 9000
        bottom = -9000

        for box_id in grouped_boxes:
            box = boxes[box_id]
            if box[0] < left:
                left = box[0]
            if box[1] < top:
                top = box[1]
            if box[0] + box[2] > right:
                right = box[0] + box[2]
            if box[1] + box[3] > bottom:
                bottom = box[1] + box[3]
        #final_rects.append([left, top, right - left, bottom - top])
        final_rects.append(([left, top, right - left, bottom - top], list(grouped_boxes)))

    return final_rects

def detect_sticks_hog(img: np.ndarray, threshold: float) -> List[Tuple[Rectangle, List[Rectangle]]]:
    found, weights = hog_desc.detect(img, padding=(STICK_WINDOW[0], STICK_WINDOW[1]//2))
    valid_indices = np.nonzero(weights > threshold)[0]
    found = found[valid_indices, :]
    weights = weights[valid_indices]

    if len(weights) == 0 or len(weights) > 10000:
        return []

    rects = np.hstack((found,
                       STICK_WINDOW[0] * np.ones((found.shape[0], 1), dtype=np.int32),
                       STICK_WINDOW[1] * np.ones((found.shape[0], 1), dtype=np.int32)))

    rects_ = list(map(lambda f: [int(f[0]), int(f[1]), int(STICK_WINDOW[0]), int(STICK_WINDOW[1])], found))

    return merge_detection_boxes(rects_, weights)

def match_sticks_to_detections(sticks: List[Stick], boxes: List[Rectangle]) -> List[Tuple[Stick, Rectangle]]:
    g = Graph()
    box_vertex_id = 0

    for i in range(len(boxes)):
        rect1 = boxes[i]
        for j in range(len(boxes)):
            if i == j:
                continue
            rect2 = boxes[j]
            g.add_node(box_vertex_id, pair=(i, j), vec=np.array([rect2[0] - rect1[0], rect2[1] - rect1[1]]))
            box_vertex_id += 1

    stick_vertex_id = box_vertex_id

    for i in range(len(sticks)):
        for j in range(len(sticks)):
            if i == j:
                continue
            vec = 2 * sticks[j].top - 2 * sticks[i].top
            g.add_node(stick_vertex_id, pair=(i, j), vec=vec)

            for box_id in range(box_vertex_id):
                box_vector = g.nodes[box_id]['vec']
                g.add_edge(stick_vertex_id, box_id, weight=np.linalg.norm(box_vector - vec))

            stick_vertex_id += 1

    stick_pair_box_pair_matching = matching.minimum_weight_full_matching(g, top_nodes=list(range(box_vertex_id, stick_vertex_id)))
    matching_size = min(box_vertex_id, stick_vertex_id - box_vertex_id)

    h = Graph()

    h.add_nodes_from(range(len(sticks)))
    h.add_nodes_from(range(len(sticks), len(sticks) + len(boxes)))

    h.add_edges_from([(i, j, {'weight': 0}) for i in range(len(sticks)) for j in range(len(sticks), len(sticks) + len(boxes))])

    for i, nodes in enumerate(stick_pair_box_pair_matching.items()):

        if i == matching_size:
            break

        stick_vertex_idx = nodes[0]
        box_vertex_idx = nodes[1]

        stick_vertex = g.nodes[stick_vertex_idx]
        box_vertex = g.nodes[box_vertex_idx]

        stick1 = stick_vertex['pair'][0]
        stick2 = stick_vertex['pair'][1]

        box1 = box_vertex['pair'][0] + len(sticks)
        box2 = box_vertex['pair'][1] + len(sticks)

        h[stick1][box1]['weight'] -= 1
        h[stick2][box2]['weight'] -= 1

    stick_box_matching = matching.minimum_weight_full_matching(h, top_nodes=range(len(sticks)))
    matching_size = min(len(sticks), len(boxes))

    return list(map(lambda s_b: (sticks[s_b[0]], boxes[s_b[1] - len(sticks)]), list(stick_box_matching.items())[:matching_size]))
    #return list(map(lambda s_b: (stick[s_b[0]], s_b[1] - len(sticks)), list(stick_box_matching.items())[:matching_size]))

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

def is_non_snow2(gray: np.ndarray) -> bool:
    roi_x = int(0.2 * gray.shape[1])
    roi_y = int(0.4 * gray.shape[0])
    roi_w = int(0.6 * gray.shape[1])
    roi_h = int(0.4 * gray.shape[0])

    return np.mean(gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]) < 100

def is_non_snow3(img: np.ndarray) -> bool:
    roi = img[int(0.4 * img.shape[0]):int(0.95 * img.shape[0]), int(0.2 * img.shape[1]):int(0.8 * img.shape[1])]

    r_b_diff = np.abs(roi[:,:,2] - roi[:,:,0])
    return (np.count_nonzero(r_b_diff < 10) / (roi.shape[0] * roi.shape[1])) < 0.3

def ground_part(img: np.ndarray) -> np.ndarray:
    return img[int(0.4 * img.shape[0]):int(0.95 * img.shape[0]), int(0.2 * img.shape[1]):int(0.8 * img.shape[1])]

def is_snow(gray: np.ndarray, img: np.ndarray, sigma: float = 0.5, threshold: int = 5) -> bool:
    ground = ground_part(gray)
    bgr = ground_part(img)
    bl = cv.GaussianBlur(ground, (5, 5), sigmaX=sigma)
    diff = ground - bl
    diff_mean = np.array([[np.count_nonzero(diff > threshold) / (ground.shape[0] * ground.shape[1]), np.mean(bgr[:,:,0])]])
    return snow_svc.predict(snow_scaler.transform(diff_mean))[0] > 0.0

snow_rect = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))

def is_snow_(gray: np.ndarray, bgr: np.ndarray, th: int = 15) -> bool:
    ground = ground_part(gray)
    ground_bgr = ground_part(bgr)
    bth = cv.morphologyEx(ground, cv.MORPH_BLACKHAT, snow_rect)
    X = snow_scaler.transform([[np.count_nonzero(bth > th) / (ground.shape[0] * ground.shape[1]), np.mean(ground_bgr[:,:,0])]])
    return snow_svc.predict(X)[0] > 0.0

def segment_sticks(img: np.ndarray, debug: bool):
    #orig = gray.copy()
    start = time()
    #gray = cv.cvtColor(draw, cv.COLOR_BGR2GRAY)
    if debug:
        #draw = cv.pyrDown(gray)
        #draw = gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.pyrDown(gray)
        orig = cv.pyrDown(img)
    else:
        gray = img.copy()
    #gray = cv.equalizeHist(gray)
    #gray = img_as_ubyte(skimage.exposure.equalize_adapthist(gray, clip_limit=0.01))
    clahe.setClipLimit(5.0)
    gray_eq = clahe.apply(gray)

    hmt_dbd1, _ = uhmt(gray_eq, hmt_selems[0][0])
    hmt_dbd3, _ = uhmt(gray_eq, hmt_selems[0][1])
    hmt_dbd5, _ = uhmt(gray_eq, hmt_selems[0][2])
    hmt_dbd7, _ = uhmt(gray_eq, hmt_selems[0][3])
    hmt_bdb1, _ = uhmt(gray_eq, hmt_selems[1][0])
    hmt_bdb3, _ = uhmt(gray_eq, hmt_selems[1][1])
    hmt_bdb5, _ = uhmt(gray_eq, hmt_selems[1][2])
    hmt_bdb7, _ = uhmt(gray_eq, hmt_selems[1][3])

    _, hmt_dbd1_ = cv.threshold(hmt_dbd1, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd3_ = cv.threshold(hmt_dbd3, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd5_ = cv.threshold(hmt_dbd5, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd7_ = cv.threshold(hmt_dbd7, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb1_ = cv.threshold(hmt_bdb1, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb3_ = cv.threshold(hmt_bdb3, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb5_ = cv.threshold(hmt_bdb5, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb7_ = cv.threshold(hmt_bdb7, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)

    #start_hyst = time()
    #hmt_dbd1_ = skimage.filters.apply_hysteresis_threshold(hmt_dbd1, 3, 10).astype(np.uint8)
    #hmt_dbd3_ = skimage.filters.apply_hysteresis_threshold(hmt_dbd3, 3, 10).astype(np.uint8)
    #hmt_bdb1_ = skimage.filters.apply_hysteresis_threshold(hmt_bdb1, 3, 10).astype(np.uint8)
    #hmt_bdb3_ = skimage.filters.apply_hysteresis_threshold(hmt_bdb3, 3, 10).astype(np.uint8)
    #print(f'hyst took {time() - start_hyst} secs')

    th1 =  asf(hmt_dbd1_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th2 =  asf(hmt_dbd3_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th3 =  asf(hmt_bdb1_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th4 =  asf(hmt_bdb3_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th5 =  asf(hmt_bdb5_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th6 =  asf(hmt_dbd5_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th7 =  asf(hmt_dbd7_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th8 =  asf(hmt_bdb7_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))

    th = np.bitwise_or(th1, th2)
    th = np.bitwise_or(th, th3)
    th = np.bitwise_or(th, th4)
    th = np.bitwise_or(th, th5)
    th = np.bitwise_or(th, th6)
    th = np.bitwise_or(th, th7)
    th = np.bitwise_or(th, th8)

    th_ = cv.erode(th, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), anchor=(0,0))
    th_ = cv.dilate(th_, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)), anchor=(1,0))

    if not debug:
        return th
    start_hough = time()
    lines = cv.HoughLinesP(th_, 1.0, np.pi / 180.0, params['hough_th'], None, params['line_length'], params['line_gap'])
    print(f'hough took {time() - start_hough} secs')
    print(f'it took {time() - start} secs')
    f = 1.0
    #draw = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    draw = orig.copy()
    draw2 = draw.copy()
    f = 1
    if lines is not None:
        lines = np.reshape(lines, (-1, 2, 2))
        lines_ = cluster_lines(lines, draw, debug)
        #print(lines_)
        #extract_features_from_lines(orig, gray, lines_)
        #return
        temp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        for line_ in lines_:
            p = line_[0]
            q = line_[1]
            _, f_vec = extract_features_from_line(temp, 2 * line_)
            f_vec = np.reshape(list(f_vec.values()), (1, -1))
            is_stick = stick_pipeline.predict(f_vec)
            if is_stick:
                color = [0, 255, 0]
            else:
                color = [0, 0, 255]
            cv.line(draw, (int(f * p[0]), int(f * p[1])), (int(f * q[0]), int(f * q[1])), color, 1)
        for line_ in lines:
            p = line_[0]
            q = line_[1]
            cv.line(draw2, (int(f * p[0]), int(f * p[1])), (int(f * q[0]), int(f * q[1])), [0, 255, 0], 1)
        visualize_line_signals(lines_, orig, gray, img)
    #cv.imshow('uhmt_th', 255 * th)
    cv.imshow('gray', gray)
    #cv.imshow('gray_eq', gray_eq)
    #cv.imshow('lines', draw2)
    #cv.imshow('lines', cv.resize(draw2, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('lines_clustered', cv.resize(draw, (0, 0), fx=0.5, fy=0.5))
    cv.imshow('lines_clustered', draw)

    cv.waitKey(0)
    cv.destroyAllWindows()

def get_sticks_in_folder2(folder: Path) -> Optional[Tuple[List[np.ndarray], Path, float]]:
    queue = Queue(maxsize=10)
    # thread = threading.Thread(target=get_non_snow_images, args=(folder, queue, 9))
    worker = MyThreadWorker(task=get_non_snow_images, args=(folder, queue, 5), kwargs={})

    heat_map = None
    stick_lengths = None
    imgs: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    QThreadPool.globalInstance().start(worker)

    loading_time = 0
    loading_start = time()
    path_img: Tuple[Path, np.ndarray] = queue.get()
    angles = []
    gray = None
    while path_img is not None:
        loading_time += (time() - loading_start)
        img = path_img[1]
        start = time()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[:-50]
        gray = cv.pyrDown(gray)
        clahe.setClipLimit(5.0)
        gray = clahe.apply(gray)
        if heat_map is None:
            heat_map = np.zeros(gray.shape, dtype=np.uint8)
        stick_regions = segment_sticks(gray, False)
        stick_regions = cv.dilate(stick_regions, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))
        heat_map += stick_regions
        print(f'took {time() - start} secs')
        imgs.append((Path(path_img[0]), img, stick_regions, None))
        loading_start = time()
        path_img = queue.get()

    _, heat_th = cv.threshold(heat_map, 3.0, 255.0, cv.THRESH_BINARY)
    lines = cv.HoughLinesP(heat_th, 1.0, np.pi / 180.0, params['hough_th'], None, params['line_length'], params['line_gap'])

    draw = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    draw2 = draw.copy()
    f = 1
    if lines is not None:
        lines = np.reshape(lines, (-1, 2, 2))
        lines_ = cluster_lines(lines)
        for line_ in lines_:
            p = line_[0]
            q = line_[1]
            cv.line(draw, (int(f * p[0]), int(f * p[1])), (int(f * q[0]), int(f * q[1])), [0, 255, 0], 1)
        for line_ in lines:
            p = line_[0]
            q = line_[1]
            cv.line(draw2, (int(f * p[0]), int(f * p[1])), (int(f * q[0]), int(f * q[1])), [0, 255, 0], 1)

    return draw, draw2
    #if len(imgs) > 0:
    #    loading_time = loading_time / len(imgs)

    #if stick_lengths is None or np.max(stick_lengths) < 0.01:
    #    return None

    #stick_lengths = stick_lengths / np.max(stick_lengths)
    #heat_map = heat_map * (1 + stick_lengths)
    #_, stick_regions = cv.threshold(heat_map.astype(np.uint8), len(imgs) - 3, 255, cv.THRESH_BINARY)

    #stick_scores = []
    #sticks_list: List[List[np.ndarray]] = []
    #for j in range(len(imgs)):
    #    sticks_ = filter_sticks(imgs[j][2], imgs[j][3], stick_regions)
    #    sticks_list.append(sticks_)
    #    stick_score = sum((map(lambda s: np.linalg.norm(s[0] - s[1]), sticks_)))
    #    stick_scores.append([stick_score, len(sticks_), j])

    ## Sort according to number of sticks and total stick lengths
    #stick_scores = sorted(stick_scores, key=lambda c: (c[1], c[0]), reverse=True)

    ## Most likely stick configuration is identified by the index which sits in stick_scores[0][2]
    #idx = stick_scores[0][2]

    ##adjusted_sticks = []
    ##for stick in sticks_list[idx]:
    ##    stick_ = adjust_endpoints(imgs[idx][1], stick, 9)
    ##    adjusted_sticks.append(stick_)


    #return sticks_list[idx], imgs[idx][0], 20.0, stick_regions

def get_rotated_bbox(line: np.ndarray) -> np.ndarray:
    if line[0, 0] > line[1, 0]:
        v = line[0] - line[1]
    else:
        v = line[1] - line[0]
    v = v / np.linalg.norm(v)
    n = v[::-1].copy()

    if np.abs(n[0]) < 0.001:
        n[1] *= -1.0
    else:
        n[0] *= -1.0

    points = np.vstack((line - 2 * n, line + 2 * n)).astype(np.int32)
    rect = cv.minAreaRect(points)
    #return cv.boxPoints(rect).astype(np.int32)
    return rect

def cluster_lines(lines: np.ndarray) -> np.ndarray:
    box_points = np.array(list(map(lambda idx_line: (get_rotated_bbox(idx_line[1]), idx_line[0]), enumerate(lines))))

    for i, line in enumerate(lines):
        bbox_groupid_i = box_points[i]
        bbox_i = bbox_groupid_i[0]
        group_i = bbox_groupid_i[1]

        line_i_len = np.linalg.norm(line[0] - line[1])

        for j in range(i+1, len(lines)):
            line_j = lines[j]
            line_j_len = np.linalg.norm(line_j[0] - line_j[1])
            bbox_groupid_j = box_points[j]
            bbox_j = bbox_groupid_j[0]

            intersection = cv.rotatedRectangleIntersection(bbox_i, bbox_j)
            if intersection[0] > 0:
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

    return result_lines

def line_upright_bbox(line: np.ndarray, max_width: int, max_height: int, width = 25) -> np.ndarray:
    w = max(1, int((width - 1) / 2))
    top_left = np.maximum(np.min(line, axis=0) - w, 0)
    bottom_right = np.max(line, axis=0) + w
    bottom_right[0] = np.minimum(bottom_right[0], max_width-1)
    bottom_right[1] = np.minimum(bottom_right[1], max_height-1)
    return np.array([top_left, bottom_right])

def visualize_line_signals(lines: np.ndarray, img: np.ndarray, gray: np.ndarray, orig: np.ndarray):
    line_map = -1 * np.ones((img.shape[0], img.shape[1]), np.int8)

    #draw = cv.pyrDown(img)
    draw = img.copy()
    for i, line in enumerate(lines):
        bbox = line_upright_bbox(line, max_width=img.shape[1], max_height=img.shape[0], width=25)
        line_map[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]] = i
        mag = cv.magnitude(cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1))
        mag_lines = skimage.measure.profile_line(mag, line[0,::-1], line[1,::-1], linewidth=15, reduce_func=None,
                                                 mode='constant')
        line_sums = np.sum(mag_lines, axis=0)
        left_edge_idx = np.argmax(line_sums[::-1][8:]) + 1
        right_edge_idx = np.argmax(line_sums[8:]) + 1
        cv.rectangle(draw, (int(bbox[0,0]), int(bbox[0,1])), (int(bbox[1,0]), int(bbox[1,1])), [0, 255, 0])
        cv.line(draw, (int(line[0,0]), int(line[0,1])), (int(line[1,0]), int(line[1,1])), [255, 0, 0])
        cv.line(draw, (int(line[0,0]) - left_edge_idx, int(line[0,1])),
                (int(line[1,0]) - left_edge_idx, int(line[1,1])), [0, 0, 255])
        cv.line(draw, (int(line[0,0]) + right_edge_idx, int(line[0,1])),
                (int(line[1,0]) + right_edge_idx, int(line[1,1])), [0, 0, 255])

    #gray_up = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_up = cv.pyrUp(gray)
    hsv = cv.cvtColor(cv.pyrUp(img), cv.COLOR_BGR2HSV)
    cv.namedWindow('lines')
    cv.setMouseCallback('lines', line_click_handler, (line_map, lines, cv.cvtColor(orig, cv.COLOR_BGR2GRAY), hsv, orig))
    cv.imshow('lines', draw)

    cv.waitKey(0)
    cv.destroyAllWindows()


def line_click_handler(e, x, y, flags, params):
    if e != cv.EVENT_LBUTTONUP:
        return
    line_map = params[0]
    lines = params[1]
    img = params[2]
    hsv = params[3]
    orig = params[4]
    line_id = line_map[y,x]
    if line_id < 0:
        return
    line = 2 * lines[line_id]
    bbox = line_upright_bbox(line, max_width=img.shape[1], max_height=img.shape[0], width=75)
    off_y = 35
    line_roi = img[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]
    orig_line_roi = orig[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]

    hsv_line_roi = hsv[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]

    #line_roi = cv.equalizeHist(line_roi)
    line = line - bbox[0]
    v = line[1] - line[0]
    v = v / np.linalg.norm(v)
    line[0] -= (off_y * v).astype(np.int32)
    line[1] += (off_y * v).astype(np.int32)
    hsv_prof_line = np.array(skimage.measure.profile_line(hsv_line_roi, line[0,::-1], line[1,::-1], linewidth=25,
                                                          reduce_func=None, mode='constant'), np.uint8)

    prof_shape = (hsv_prof_line.shape[0], hsv_prof_line.shape[1])
    hsv_signal = np.dstack(((360.0 * (hsv_prof_line[:,:,0] / 255.0)).astype(np.float32), np.ones(prof_shape, np.float32), np.ones(prof_shape, np.float32)))
    sat_signal = np.dstack(((360.0 * (hsv_prof_line[:,:,1] / 255.0)).astype(np.float32), np.ones(prof_shape, np.float32), np.ones(prof_shape, np.float32)))
    val_signal = np.dstack(((360.0 * (hsv_prof_line[:,:,2] / 255.0)).astype(np.float32), np.ones(prof_shape, np.float32), np.ones(prof_shape, np.float32)))

    hsv_signal = cv.convertScaleAbs(cv.cvtColor(hsv_signal, cv.COLOR_HSV2BGR), alpha=255)
    sat_signal = cv.convertScaleAbs(cv.cvtColor(sat_signal, cv.COLOR_HSV2BGR), alpha=255)
    val_signal = cv.convertScaleAbs(cv.cvtColor(val_signal, cv.COLOR_HSV2BGR), alpha=255)

    #cv.imshow('hsv_s', hsv_signal)
    #cv.imshow('sat_s', sat_signal)
    #cv.imshow('val_s', val_signal)

    dx, dy = cv.Sobel(line_roi, cv.CV_32F, 1, 0), cv.Sobel(line_roi, cv.CV_32F, 0, 1)
    mag = cv.magnitude(dx, dy)
    mag /= np.max(mag)

    angles = cv.phase(dx, np.abs(dy), None, True)
    angles = np.mod(angles - 90, 180)
    #angles = np.where(angles > 180, angles - 180, angles)
    #if angle > 90:
    #    angles = np.where(angles < 90, angles + 180, angles)
    #else:
    #    angles = np.where(angles > 90, angles - 180, angles)

    angle = line_angle(line, True)

    edge_pos = line_edge_offsets(line, mag)
    #is_line_on_stick(line, edge_pos, angles)

    left_mag = skimage.measure.profile_line(mag, line[0,::-1] - [0, edge_pos[0]], line[1,::-1] - [0, edge_pos[0]],
                                            linewidth=3, reduce_func=None)
    right_mag = skimage.measure.profile_line(mag, line[0,::-1] + [0, edge_pos[1]], line[1,::-1] + [0, edge_pos[1]],
                                             linewidth=3, reduce_func=None, mode='constant')

    low, high = (angle - 10) % 360, (angle + 10) % 360
    if low > high:
        good_angles = np.bitwise_or(cv.inRange(angles, 0, high), cv.inRange(angles, low, 360))
    else:
        good_angles = cv.inRange(angles, low, high)
    low, high = (angle + 170) % 360, (angle + 190) % 360
    if low > high:
        good_angles_ = np.bitwise_or(cv.inRange(angles, low, 360), cv.inRange(angles, 0, high))
        good_angles = np.bitwise_or(good_angles, good_angles_)
    else:
        good_angles = np.bitwise_or(good_angles, cv.inRange(angles, low, high))
    good_angles = cv.bitwise_and(good_angles, cv.inRange(mag, 0.1, 1.0))
    #hsv = np.dstack((angles, np.ones(angles.shape, np.float32), mag)) #np.ones(angles.shape, np.float32)))
    #bgr = cv.convertScaleAbs(cv.cvtColor(hsv, cv.COLOR_HSV2BGR), None, 255)
    left_angles = skimage.measure.profile_line(angles, line[0,::-1] - [0, edge_pos[0]], line[1,::-1] - [0, edge_pos[0]],
                                               linewidth=5, reduce_func=None, mode='constant')
    left_angles = np.where(left_angles > 180, left_angles - 180, left_angles)
    left_diff = np.where(np.abs(left_angles - angle) < 11, 1, -1)
    left_diff = np.reshape(np.max(left_diff, axis=1), (-1, 1))
    right_angles = skimage.measure.profile_line(angles, line[0,::-1] + [0, edge_pos[1]], line[1,::-1] + [0, edge_pos[1]]
                                                , linewidth=5, reduce_func=None, mode='constant')
    right_angles = np.where(right_angles > 180, right_angles - 180, right_angles)
    right_diff = np.where(np.abs(right_angles - angle) < 11, 1, -1)
    right_diff = np.reshape(np.max(right_diff, axis=1), (-1, 1))
    stick_edge_indicator = np.hstack((left_diff, right_diff))
    top_end, bottom_end = find_stick_end(stick_edge_indicator)

    cv.circle(orig_line_roi, (int(line[0,0]), int(line[0,1]) + top_end), 3, [0, 255, 0])
    cv.circle(orig_line_roi, (int(line[1,0]), int(line[0,1]) + bottom_end), 3, [255, 0, 0])

    #prof_line = np.array(skimage.measure.profile_line(bgr, line[0,::-1] - np.array([12, 0]), line[1,::-1] + np.array([12, 0]), linewidth=25, reduce_func=None, mode='constant'), np.uint8)
    #viz = visualize_signal(prof_line)
    #line_viz_enlarged = cv.resize(bgr, (0, 0), fx=3, fy=3, interpolation=cv.INTER_NEAREST)

    #cv.imwrite(f'line{line_id}.jpg', orig_line_roi)
    #cv.imwrite(f'line{line_id}_angles.jpg', bgr)
    #with open(f'line{line_id}_coords.txt', 'wb') as f:
    #    pickle.dump(line, f)

    #cv.imshow('line_signal', line_viz_enlarged)
    #cv.setMouseCallback('line_signal', pick_angle, (angles, angle, mag))
    cv.imshow('line_roi', line_roi)
    #cv.imshow('good', good_angles)
    #cv.imshow('mag', cv.convertScaleAbs(mag, alpha=255))
    #cv.imshow('left_mags', cv.convertScaleAbs(left_mag, alpha=255))
    #cv.imshow('right_mags', cv.convertScaleAbs(right_mag, alpha=255))
    #cv.imshow('left_ang', left_angles)
    #cv.imshow('right_ang', right_angles)
    cv.imshow('stick_ends', orig_line_roi)

def visualize_signal(signal: np.ndarray) -> np.ndarray:
    if signal.ndim > 1:
        return signal.astype(np.uint8)
    viz = np.zeros((255, signal.shape[0]), np.uint8)
    for t, v in enumerate(signal):
        viz[255 - v:255,t] = 255
    return viz

def line_angle(line: np.ndarray, normal: bool = False) -> float:
    v = line[1] - line[0]
    return (math.degrees(math.atan2(v[1], v[0]))) % 180
    v = v / np.linalg.norm(v)
    angle = math.degrees(math.acos(np.dot(v, np.array([0, 1]))))
    if normal:
        return (angle + 180)
    return angle

def pick_angle(e, x, y, flags, params):
    if e != cv.EVENT_LBUTTONUP:
        return
    angles = params[0]
    angle = params[1]
    mag = params[2]
    print(f'angle is {angles[y // 3,x // 3]} and norma line angle is {params[1]}')
    print(f'range to check is {((angle - 5) % 360, (angle + 5) % 360)}, {((angle + 175) % 360, (angle + 185) % 360)}')
    print(f'mag is {mag[y // 3, x // 3]}')

def line_edge_offsets(line: np.ndarray, mag: np.ndarray) -> List[int]:
    #bbox = line_upright_bbox(line, 17)
    mag_lines = skimage.measure.profile_line(mag, line[0, ::-1], line[1, ::-1], linewidth=15, reduce_func=None,
                                             mode='constant')
    line_sums = np.sum(mag_lines, axis=0)
    left_edge_idx = np.argmax(line_sums[::-1][8:]) + 1
    right_edge_idx = np.argmax(line_sums[8:]) + 1

    return [left_edge_idx, right_edge_idx]

def is_line_on_stick(line: np.ndarray, edge_pos: List[int], angles: np.ndarray) -> bool:
    left_angles = skimage.measure.profile_line(angles,
                                               line[0,::-1] - [0, edge_pos[0]],
                                               line[1,::-1] - [0, edge_pos[0]], mode='reflect')
    left_angles = np.where(left_angles > 180, left_angles - 180, left_angles)
    right_angles = skimage.measure.profile_line(angles,
                                               line[0,::-1] + [0, edge_pos[1]],
                                               line[1,::-1] + [0, edge_pos[1]], mode='reflect')

    right_angles = np.where(right_angles > 180, right_angles - 180, right_angles)

    diff = np.abs(left_angles - right_angles)

    plt.figure(figsize=(8, 8))
    plt.plot(left_angles, linewidth=1, color='green')
    plt.plot(right_angles, linewidth=1, color='blue')
    plt.plot(diff, linewidth=1, color='red', linestyle='dashed')
    plt.title(f'diff_m={np.mean(diff):<.1f} diff_std={np.std(diff):<.1f} left: m={np.mean(left_angles):<.1f} s={np.std(left_angles)},\n right: m={np.mean(right_angles):<.1f}, s={np.std(right_angles)}')
    plt.savefig('fig.png')

    fig = cv.imread('fig.png')
    cv.imshow('fig', fig)

features = None
labels = None
FEATURES = ['DIFF_MEAN', 'DIFF_STD', 'LEFT_STD', 'RIGHT_STD', 'LR_MEAN_DIFF', 'I_STD', 'DIFF_FROM_ANGLE', 'LEFT_PER', 'RIGHT_PER', 'LEFT_RIGHT_DIFF']
lines_copied = None
def extract_features_from_lines(img: np.ndarray, gray: np.ndarray, lines: np.ndarray):
    global lines_copied, features, labels, FEATURES
    if features is None:
        features = {}
        labels = []
        for feature in FEATURES:
            features[feature] = []
    local_features = {}
    local_labels = []
    for feature in FEATURES:
        local_features[feature] = []
    line_map = -1 * np.ones(gray.shape, np.int8)
    line_img = img.copy()
    line_count = len(lines)
    lines_copied = lines.copy()
    for i, line in enumerate(lines):
        #line = lines[line_idx]
        bbox = line_upright_bbox(line, width=25, max_width=line_img.shape[1], max_height=line_img.shape[0])
        line_map[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]] = i
        edge_pos, f_vec = extract_features_from_line(gray, line)
        draw_line_with_edge(line_img, line, edge_pos)
        for feature_name, value in f_vec.items():
            local_features[feature_name].append(value)
        local_labels.append(1)
        cv.rectangle(line_img, (int(bbox[0,0]), int(bbox[0,1])),
                     (int(bbox[1,0]), int(bbox[1,1])), [0, 255, 0])
    cv.namedWindow('lines')
    cv.setMouseCallback('lines', feature_line_img_callback, param=(local_features, local_labels, line_map, gray,
                                                                   lines_copied, line_img))

    cv.imshow('lines', line_img)
    key = cv.waitKey(0)
    if key == ord('a') or key == ord('q'):
        for feature in FEATURES:
            features[feature].extend(local_features[feature])
        labels.extend(local_labels)
        if key == ord('q'):
            to_save = features.copy()
            to_save['LABEL'] = labels
            df = pd.DataFrame(data=to_save).dropna(axis=0)
            df.to_csv('line_features.csv', mode='a', header=True)
            features = None
            labels = None
    cv.destroyWindow('lines')

def feature_line_img_callback(e, x, y, flags, params):
    global lines_copied
    line_map = params[2]
    local_features = params[0]
    local_labels = params[1]
    lines = params[4]
    gray = params[3]
    line_img = params[5]
    if e not in [cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP, cv.EVENT_MBUTTONUP]:
        return
    line_id = line_map[y,x]
    if line_id < 0:
        new_id = len(local_labels)
        y_off = randint(20, 150)
        x_off = randint(-20, 20)
        line = np.array([[x+x_off, y-y_off], [x, y]], np.int32)
        bbox = line_upright_bbox(line, max_width=line_map.shape[1], max_height=line_map.shape[0], width=25)
        line_map[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]] = new_id
        edge_pos, f_vec = extract_features_from_line(gray, line)
        for feature_name, value in f_vec.items():
            local_features[feature_name].append(value)
        local_labels.append(0)
        cv.rectangle(line_img, (int(bbox[0,0]), int(bbox[0,1])),
                     (int(bbox[1,0]), int(bbox[1,1])), [0, 0, 255])
        draw_line_with_edge(line_img, line, edge_pos)
        #lines_copied = np.append(lines_copied, line)
        lines_copied.append(line)
    else:
        if e == cv.EVENT_LBUTTONUP:
            local_labels[line_id] = 1
            color = [0, 255, 0]
        elif e == cv.EVENT_RBUTTONUP:
            local_labels[line_id] = 0
            color = [0, 0, 255]
        elif e == cv.EVENT_MBUTTONUP:
            local_labels[line_id] = math.nan
            color = [255, 0, 255]
        line = lines[line_id]
        bbox = line_upright_bbox(line, max_width=line_map.shape[1], max_height=line_map.shape[0], width=25)
        cv.rectangle(line_img, (int(bbox[0,0]), int(bbox[0,1])),
                     (int(bbox[1,0]), int(bbox[1,1])), color)
    cv.imshow('lines', line_img)

def draw_line_with_edge(line_img: np.ndarray, line: np.ndarray, edge_pos: List[int]):
    cv.line(line_img, (int(line[0, 0]), int(line[0, 1])),
            (int(line[1, 0]), int(line[1, 1])), [0, 255, 0], lineType=cv.LINE_8)
    cv.line(line_img, (int(line[0, 0] - edge_pos[0]), int(line[0, 1])),
            (int(line[1, 0] - edge_pos[0]), int(line[1, 1])), [255, 0, 0], lineType=cv.LINE_8)
    cv.line(line_img, (int(line[0, 0] + edge_pos[1]), int(line[0, 1])),
            (int(line[1, 0] + edge_pos[1]), int(line[1, 1])), [255, 0, 0], lineType=cv.LINE_8)


def extract_features_from_line(gray: np.ndarray, line: np.ndarray, plain_fvec: bool = True) -> np.ndarray:
    bbox = line_upright_bbox(line, max_width=gray.shape[1], max_height=gray.shape[0], width=25)
    line_roi = gray[bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]]
    intensity = skimage.measure.profile_line(gray, line[0, ::-1], line[1, ::-1], mode='constant')
    dx, dy = cv.Sobel(line_roi, cv.CV_32F, 1, 0), cv.Sobel(line_roi, cv.CV_32F, 0, 1)
    mag, angles = cv.magnitude(dx, dy), cv.phase(dx, dy, angleInDegrees=True)

    angles = np.mod(angles - 90, 180)

    line_l = line - bbox[0]
    edge_pos = line_edge_offsets(line_l, mag)

    left_angles = skimage.measure.profile_line(angles,
                                               line_l[0,::-1] - [0, edge_pos[0]],
                                               line_l[1,::-1] - [0, edge_pos[0]], mode='constant', linewidth=3,
                                               reduce_func=None)
    #left_angles = np.where(left_angles > 180, left_angles - 180, np.mod(left_angles + 180, 180))
    #left_angles = np.where(left_angles >= 270, left_angles - 180, left_angles)
    left_angles = np.where(left_angles > 180, left_angles - 180, left_angles)
    #left_angles = np.where(left_angles < 100, left_angles + 180, left_angles)
    right_angles = skimage.measure.profile_line(angles,
                                                line_l[0,::-1] + [0, edge_pos[1]],
                                                line_l[1,::-1] + [0, edge_pos[1]], mode='constant', linewidth=3,
                                                reduce_func=None)
    #right_angles = np.where(right_angles >= 270, right_angles - 180, right_angles)
    right_angles = np.where(right_angles > 180, right_angles - 180, right_angles)

    angle = line_angle(line)

    right_angle_diff = np.abs(right_angles - angle)
    right_angle_diff = np.count_nonzero(right_angle_diff < 11) / 3.0
    left_angle_diff = np.abs(left_angles - angle)
    left_angle_diff = np.count_nonzero(left_angle_diff < 11) / 3.0

    diff = np.abs(left_angles[:,1] - right_angles[:,1])
    features = {
        FEATURES[0] : np.mean(diff),
        FEATURES[1] : np.std(diff),
        #FEATURES[2] : np.mean(left_angles),
        FEATURES[2] : np.std(left_angles),
        #FEATURES[4] : np.mean(right_angles),
        FEATURES[3] : np.std(right_angles),
        FEATURES[4] : np.abs(np.mean(left_angles) - np.mean(right_angles)),
        FEATURES[5] : np.std(intensity),
        FEATURES[6] : np.mean(np.abs(left_angles - angle)),
        FEATURES[7] : left_angle_diff / float(left_angles.shape[0]),
        FEATURES[8] : right_angle_diff / float(right_angles.shape[0]),
        #FEATURES[9] : np.abs(left_angles - right_angles),
    }
    if plain_fvec:
        return np.reshape(list(features.values()), (1, -1))
    return edge_pos, features

def find_stick_end(stick_edge_indicator: np.ndarray) -> List[int]:
    top_sums = np.zeros((stick_edge_indicator.shape[0],), np.int32)
    bottom_sums = np.zeros((stick_edge_indicator.shape[0],), np.int32)

    for i in range(stick_edge_indicator.shape[0]):
        top_sums[i] = np.sum(stick_edge_indicator[i:])
        bottom_sums[i] = np.sum(stick_edge_indicator[:i])

    top_end = np.argmax(top_sums)
    bottom_end = np.argmax(bottom_sums)

    return [int(top_end), int(bottom_end)]

def detect_sticks(gray: np.ndarray):
    clahe.setClipLimit(5.0)
    gray_eq = clahe.apply(gray)

    hmt_dbd1, _ = uhmt(gray_eq, hmt_selems[0][0])
    hmt_dbd3, _ = uhmt(gray_eq, hmt_selems[0][1])
    hmt_dbd5, _ = uhmt(gray_eq, hmt_selems[0][2])
    hmt_dbd7, _ = uhmt(gray_eq, hmt_selems[0][3])
    hmt_bdb1, _ = uhmt(gray_eq, hmt_selems[1][0])
    hmt_bdb3, _ = uhmt(gray_eq, hmt_selems[1][1])
    hmt_bdb5, _ = uhmt(gray_eq, hmt_selems[1][2])
    hmt_bdb7, _ = uhmt(gray_eq, hmt_selems[1][3])

    _, hmt_dbd1_ = cv.threshold(hmt_dbd1, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd3_ = cv.threshold(hmt_dbd3, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd5_ = cv.threshold(hmt_dbd5, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_dbd7_ = cv.threshold(hmt_dbd7, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb1_ = cv.threshold(hmt_bdb1, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb3_ = cv.threshold(hmt_bdb3, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb5_ = cv.threshold(hmt_bdb5, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)
    _, hmt_bdb7_ = cv.threshold(hmt_bdb7, 7.0, 1.0, cv.THRESH_BINARY)# | cv.THRESH_OTSU)

    th1 =  asf(hmt_dbd1_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th2 =  asf(hmt_dbd3_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th3 =  asf(hmt_bdb1_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th4 =  asf(hmt_bdb3_, 9, 1, 'co') #cv.morphologyEx(hmt_dbd7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th5 =  asf(hmt_bdb5_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb1_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th6 =  asf(hmt_dbd5_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb3_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th7 =  asf(hmt_dbd7_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb5_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    th8 =  asf(hmt_bdb7_, 9, 1, 'co') #cv.morphologyEx(hmt_bdb7_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))

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
    return cluster_lines(lines)
