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
import math
from math import comb
import statistics
from random import randrange
import matplotlib.pyplot as plt

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
ENDPOINT_SVC_FILE = Path(sys.argv[0]).parent / 'camera_processing/endpoint_detector_svc_sklearn.joblib'
BOTTOM_ENDPOINT_HOG_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/bottom_endpoint_hogdesc2')
TOP_ENDPOINT_HOG_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/top_endpoint_hogdesc4')
SNOW_SVC_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/snow_svc.joblib')
SNOW_SCALER_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/snow_scaler.joblib')

try:
    success = hog_desc.load(STICK_HOG_FILE)
except:
    print(f'Could not load file {STICK_HOG_FILE}')
    exit(-1)

endpoint_svc: SVC  = joblib.load(ENDPOINT_SVC_FILE)
endpoint_hog = cv.HOGDescriptor(ENDPOINT_WINDOW, (16, 16), (8, 8), (8, 8), 9, 1, -1, cv.HOGDescriptor_L2Hys, 0.2, False, cv.HOGDESCRIPTOR_DEFAULT_NLEVELS, True)

bottom_endpoint_hog = cv.HOGDescriptor(ENDPOINT_WINDOW, (16, 16), (8, 8), (8, 8), 9, 1, -1, cv.HOGDescriptor_L2Hys, 0.2, False, cv.HOGDESCRIPTOR_DEFAULT_NLEVELS, True)
bottom_endpoint_hog.load(BOTTOM_ENDPOINT_HOG_FILE)

top_endpoint_hog = cv.HOGDescriptor(ENDPOINT_WINDOW, (16, 16), (8, 8), (8, 8), 18, 1, -1, cv.HOGDescriptor_L2Hys, 0.2, False, cv.HOGDESCRIPTOR_DEFAULT_NLEVELS, True)
top_endpoint_hog.load(TOP_ENDPOINT_HOG_FILE)

snow_svc: SVC = joblib.load(SNOW_SVC_FILE)
snow_scaler: StandardScaler = joblib.load(SNOW_SCALER_FILE)

#print(endpoint_svc)

clahe = cv.createCLAHE()
clahe.setTilesGridSize((8, 8))


def rect_se(width: int, height: int) -> np.ndarray:
    """Constructs a rectangular structuring element of a given width and height.

    Parameters
    ----------
    width : int
    height : int

    Returns
    -------
    np.ndarray
        a rectangular structuring element
    """

    return cv.getStructuringElement(cv.MORPH_RECT, (width, height))


def stick_segmentation_preprocess(img: np.ndarray) -> np.ndarray:
    """Segments line-like structures out of a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        a grayscale image

    Returns
    -------
    np.ndarray
        a binary image of segmented line-like structures
    """

    thresh_method = cv.ADAPTIVE_THRESH_MEAN_C

    denoised = denoise(img)

    # Closing off the possible duct-tapes on sticks which would disconnect components in segmented image
    closed: np.ndarray = cv.morphologyEx(denoised, cv.MORPH_CLOSE, rect_se(19, 19))

    wth: np.ndarray = cv.morphologyEx(closed, cv.MORPH_TOPHAT, rect_se(19, 19))

    thresh: np.ndarray = cv.adaptiveThreshold(wth, 255.0, thresh_method, cv.THRESH_BINARY, 25, -2)

    # close holes in our thresholded image
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rect_se(5, 13))
    open = cv.morphologyEx(thresh, cv.MORPH_OPEN, rect_se(1, 5))

    # This extracts and finally returns line-like structures
    return cv.morphologyEx(thresh, cv.MORPH_OPEN, rect_se(1, 13))
    # return cv.morphologyEx(open, cv.MORPH_CLOSE, rect_se(3, 15))


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


def detect_sticks(img: np.ndarray, scale_lines_by: float = 1.0, merge_lines: bool = True) -> List[List[int]]:
    """Detects sticks in the given image.

    Parameters
    ----------
    img : np.ndarray
        the image to detect sticks in
    scale_lines_by : float, optional
        all detected lines will be scaled by this factor (default is 1.0)
    merge_lines : bool, optional
        debug parameter, whether to merge multiple lines belonging to the same stick (default is True)

    Returns
    -------
    List[List[int]]
        list of detected lines, each line is a list of the following format: [x0, y0, x1, y1] where y0 < y1
    """
    img_ = img
    if len(img.shape) == 3:
        img_ = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
    # First preprocess the input image
    preprocessed = stick_segmentation_preprocess(img_)

    n_labels, label_img = cv.connectedComponents(preprocessed, connectivity=4, ltype=cv.CV_16U)
    region_props = regionprops(label_img)

    likely_labels_stats: List[Tuple[Label, Height, Area, Ecc, Centroid]] = get_likely_labels(preprocessed, region_props)
    likely_labels = {l[0] for l in likely_labels_stats}

    # Filter out labels which are not interesting
    for region_prop in region_props:
        if region_prop.label in likely_labels:
            continue
        bbox = region_prop.bbox
        preprocessed[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0

    # for r in range(preprocessed.shape[0]):
    #    for c in range(preprocessed.shape[1]):
    #        if label_img[r, c] not in likely_labels:
    #            preprocessed[r, c] = 0
    height = likely_labels_stats[5][1]

    lines = cv.HoughLinesP(preprocessed, 1, np.pi / 180, int(0.8 * height), height, 20)

    # Transform the lines so they all have their first endpoint is the higher one
    lines = list(
        map(lambda line: [line[0][2], line[0][3], line[0][0], line[0][1]] if line[0][1] > line[0][3] else list(line[0]),
            lines))

    if not merge_lines:
        return (np.array(lines) * scale_lines_by).astype(int)

    # Now we're onto merging multiple detected lines per label
    close_lines: Dict[int, List[List[int]]] = {l[0]: [] for l in likely_labels_stats}

    # Assign each line the label whose bounding box contains that line, we get bins of lines that belong to the same bbox
    for line in lines:
        line_mid_point = (int(0.5 * (line[0] + line[2])), int(0.5 * (line[1] + line[3])))
        for l in likely_labels:
            if bbox_contains(region_props[l - 1].bbox, line_mid_point):
                close_lines[l].append(line)

    merged_lines: List[List[int]] = []
    # This is the actual "merging", from each line bin select the 2 lines that have the maximum and minumum y coordinate
    # and create a new line combining the endpoints of the two lines
    for lines in close_lines.values():
        if len(lines) == 0:
            continue
        # Retrieve the highest endpoint
        max_y_endpoint = max(lines, key=lambda line: line[3])[2:]
        # Retrieve the lowest endpoint
        min_y_endpoint = min(lines, key=lambda line: line[1])[:2]

        max_y_endpoint[0] = int(max_y_endpoint[0] * scale_lines_by)
        max_y_endpoint[1] = int(max_y_endpoint[1] * scale_lines_by)

        min_y_endpoint[0] = int(min_y_endpoint[0] * scale_lines_by)
        min_y_endpoint[1] = int(min_y_endpoint[1] * scale_lines_by)

        merged_lines.append(min_y_endpoint + max_y_endpoint)

    return merged_lines


def get_likely_labels(label_img: np.ndarray, label_stats) -> List[Tuple[Label, Height, Area, Ecc, Centroid]]:
    """Returns labels that are most likely to represent regions of sticks.

    Parameters
    ----------
    label_img : np.ndarray
        image of labeled regions
    label_stats : List[RegionProps]
        list of scikit structure RegionProps containing various properties of labels of label_img

    Returns
    -------
    List[Tuple[int, int, int, float, Tuple[int, int]]]
        a list of tuples of the most likely labels along with some properties: Label, Height, Area, Eccentricity, Centroid
    """

    # Retain labels that are elongated
    likely_labels = list(
        sorted(filter(lambda l: l.eccentricity > 0.87, label_stats), key=height_of_region, reverse=True))

    max_height = height_of_region(likely_labels[0])

    likely_labels = list(filter(lambda l: height_of_region(l) >= 0.2 * max_height, likely_labels))

    return list(map(lambda l: (l.label, height_of_region(l), l.area, l.eccentricity, l.centroid), likely_labels))


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

def verify_stick(stick: Stick, angle_img: np.ndarray, sigma: float) -> bool:
    e1 = np.array([stick.top[1], stick.top[0]])
    e2 = np.array([stick.bottom[1], stick.bottom[0]])
    profile_line = skimage.measure.profile_line(angle_img, e1, e2)
    return np.std(profile_line) < sigma


def get_angle_image(img: np.ndarray) -> np.ndarray:
    dx = cv.Sobel(img, cv.CV_32F, 1, 0)
    dy = cv.Sobel(img, cv.CV_32F, 0, 1)
    return cv.phase(dx, dy, angleInDegrees=True)


def filter_non_valid_angles(angle_img: np.ndarray) -> np.ndarray:
    mask1 = angle_img <= 30.0
    mask2 = np.bitwise_and(angle_img >= 150, angle_img <= 210)
    mask3 = angle_img >= 330
    mask = np.bitwise_or(np.bitwise_or(mask1, mask2), mask3)
    return angle_img * mask

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


def detect_sticks_hmt(img: np.ndarray, height_perc: float) -> List[List[int]]:
    prep = preprocess_image(img)
    hmt, mask = uhmt(prep, dbd_se)
    height = int(height_perc * img.shape[0])
    rankd = skimage.filters.rank.percentile(mask, skimage.morphology.rectangle(7, 3), p0=0.8)
    closed = cv.morphologyEx(rankd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    thin = img_as_ubyte(skimage.morphology.thin(closed))
    # show_imgs_([mask, rankd, closed, thin], ["mask", "randk", "closed", "thin"])
    # cv.destroyAllWindows()
    # lines = skimage.transform.probabilistic_hough_line(closed,
    #                                              threshold=int(0.5 * height),
    #                                              line_length=int(0.4 * height),
    #                                              line_gap=int(0.07 * height))
    lines = skimage.transform.probabilistic_hough_line(thin,
                                                       threshold=int(0.1 * height),
                                                       line_length=int(1.0 * height),
                                                       line_gap=int(0.2 * height))

    labels, num_labels = skimage.measure.label(thin, connectivity=2, return_num=True)
    print(num_labels)

    show_imgs_([img_as_ubyte(labels), thin], ['labels', 'thin'])
    cv.destroyAllWindows()

    return lines
    # return merge_lines(lines)


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


def detect_sticks_hmt2(img: np.ndarray, height_percentage: float) -> List[List[int]]:
    print(height_percentage)
    height = int(0.8 * height_percentage * img.shape[0])
    print(height)
    prep = preprocess_image(img)
    hmt, mask = uhmt(prep)

    rankd = skimage.filters.rank.percentile(mask, skimage.morphology.rectangle(7, 3), p0=0.8)
    closed = cv.morphologyEx(rankd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    thin = skimage.morphology.thin(closed)

    valid = skimage.morphology.remove_small_objects(thin, height, connectivity=2)

    show_imgs_([img_as_ubyte(thin), img_as_ubyte(valid)], ['thin', 'valid'])
    cv.waitKey(0)
    cv.destroyAllWindows()

    labels, num_labels = skimage.measure.label(thin, connectivity=2, return_num=True)

    region_props = regionprops(labels)

    return []


def merge_lines(lines: List[Tuple[Tuple[int]]]) -> List[List[int]]:
    if len(lines) == 0:
        return []
    dist_angles = []

    thetas = []
    rhos = []

    lines_rhos_thetas = []
    for line in lines:
        p = np.array(line[0])
        q = np.array(line[1])
        u = (q - p) * (1.0 / np.linalg.norm(q - p))
        v = u * u.dot(-p)
        r_v = -(-p - v)
        r = np.linalg.norm(r_v)
        r_v = r_v * (1.0 / np.linalg.norm(r_v))
        theta = 90.0 + (np.math.acos(np.array([1.0, 0.0]).dot(r_v)) / np.pi) * 180.0

        dist_angles.append((r, theta))

        rhos.append(r)
        thetas.append(theta)
        line = [list(line[0]), list(line[1])]

        if line[0][1] > line[1][1]:
            line[0], line[1] = line[1], line[0]

        lines_rhos_thetas.append((line, r, theta))

    lines_rhos_thetas = sorted(lines_rhos_thetas, key=lambda lrt: lrt[1])

    lines = []
    current_line_r_t = lines_rhos_thetas[0]

    for line_r_t in lines_rhos_thetas[1:]:
        if np.abs(line_r_t[2] - current_line_r_t[2]) > 5:
            # if np.abs(line_r_t[1] - current_line_r_t[1]) > 20:
            lines.append(current_line_r_t[0])
            current_line_r_t = line_r_t
            continue
        line = line_r_t[0]
        current_line = current_line_r_t[0]
        if line[0][1] < current_line[0][1]:
            new_line = [line[0], current_line[1]]
            if np.abs(compute_line_theta(new_line) - current_line_r_t[2]) < 3:
                current_line_r_t = (new_line, current_line_r_t[1], current_line_r_t[2])
        if line[1][1] > current_line[1][1]:
            new_line = [current_line[0], line[1]]
            if np.abs(compute_line_theta(new_line) - current_line_r_t[2]) < 3:
                current_line_r_t = (new_line, current_line_r_t[1], current_line_r_t[2])

    lines.append(current_line_r_t[0])

    # print(lines)

    # plot = sb.scatterplot(x=rhos, y=thetas).figure
    # plot.savefig("fig.png")
    # fig = cv.imread("fig.png")

    # cv.imshow("fig", fig)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # plt.clf()

    return lines


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


strel = cv.getStructuringElement(cv.MORPH_RECT, (3, 7))

def preprocess_phase(img: np.ndarray) -> np.ndarray:
    prep = preprocess_image(img)
    hit_miss, mask = uhmt(prep, dbd_se3)
    # _, mask2 = uhmt(prep, bdb_se5)
    _, mask = cv.threshold(hit_miss, 10.0, 1.0, cv.THRESH_BINARY)
    #rankd = skimage.filters.rank.percentile(mask, skimage.morphology.rectangle(7, 3), p0=0.3)
    rankd = cv.filter2D(mask, -1, cv.getStructuringElement(cv.MORPH_RECT, (3, 7)))
    _, rankd = cv.threshold(rankd, int(0.7 * 21), 255.0, cv.THRESH_BINARY)
    rankd = cv.dilate(rankd, cv.getStructuringElement(cv.MORPH_RECT, (3, 7)))
    rankd = np.minimum(mask, rankd)
    #rankd = cv.filter2D(mask, cv.CV_8U, strel)
    #_, rankd = cv.threshold(rankd, 6, 255, cv.THRESH_BINARY)
    # rankd2 = skimage.filters.rank.percentile(mask2, skimage.morphology.rectangle(7, 3), p0=0.8)
    #closed = cv.morphologyEx(rankd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))
    # closed2 = cv.morphologyEx(rankd2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    thin = skimage.morphology.thin(rankd)
    #thin = skimage.morphology.thin(closed)
    # thin2 = skimage.morphology.thin(closed2)

    area_opened = skimage.morphology.remove_small_objects(thin, min_size=8, connectivity=2)
    # area_opened2 = skimage.morphology.remove_small_objects(thin2, min_size=8, connectivity=2)
    # area_opened = np.bitwise_or(area_opened, area_opened2)

    # up = cv.resize(prep, (0, 0), fx=2, fy=2)
    # overlaid = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    return area_opened, area_opened


def detect_sticks_from_preprocessed(img: np.ndarray, height_percentage: float) -> List[Tuple[Tuple[int]]]:
    height = int(height_percentage * img.shape[0])

    lines = skimage.transform.probabilistic_hough_line(img, threshold=int(0.4 * height),
                                                       line_length=int(height * 0.9),
                                                       line_gap=15)

    return lines


def get_lines_from_preprocessed(img: np.ndarray, _img: np.ndarray) -> List[List[np.ndarray]]:
    labels, num_labels = skimage.measure.label(img, connectivity=2, return_num=True)
    reg_props = skimage.measure.regionprops(labels)

    lines = []
    start = time()
    for reg_prop in reg_props:
        coords = reg_prop.coords
        coords[:, 0] = coords[:, 0] + 64
        coords[:, 1] = coords[:, 1] + 32
        coords[:, [0, 1]] = coords[:, [1, 0]]
        # if not verify_lines(list(coords), cv.copyMakeBorder(_img, 64, 64, 64, 64, cv.BORDER_REFLECT)):
        #    continue
        min_y_coord = min(reg_prop.coords, key=lambda c: c[0])
        max_y_coord = max(reg_prop.coords, key=lambda c: c[0])

        lines.append([np.array([min_y_coord[1], min_y_coord[0]]), np.array([max_y_coord[1], max_y_coord[0]])])

    print(f"total verification time is {time() - start} secs")
    return lines


def verify_line(coords: List[Tuple[int, int]], img: np.ndarray) -> bool:
    res = (64, 128)
    f_vec_length = 4212
    features = np.zeros((len(coords), f_vec_length), dtype=np.float)

    # individual_predicts = 0
    building_features = time()
    for i, (y, x) in enumerate(coords):
        y += 60
        x += 24
        patch = img[y - res[0] // 2:y + res[0] // 2, x - res[1] // 2: x + res[1] // 2]
        # f_ = skimage.feature.hog(patch)
        # print(f"featu.shape = {features.shape} f_.shape = {f_.shape}")
        features[i] = skimage.feature.hog(patch)
        # start = time()
        # pre = lin_svc.predict([features[i]])
        # individual_predicts += (time() - start)
    building_features = time() - building_features
    start = time()
    predicts = lin_svc.predict(features)
    print(f"batch predictions took {time() - start} secs")
    print(f"building features took {building_features} secs")
    # print(f"individual predictions took {individual_predicts} secs")

    return True


def verify_lines(coords: List[Tuple[int, int]], img: np.ndarray) -> bool:
    _, weights = hog_detect.detect(img, searchLocations=coords)
    if len(weights) == 0:
        return False
    return True
    # valid_points = np.count_nonzero(weights.ravel() > 0.1)
    # return (valid_points / weights.shape[0]) >= 0.75


def is_night(img: np.ndarray) -> bool:
    g = img[:, :, 1]
    return np.abs(np.mean(img[:, :, 2] - g)) < 10.0 and np.abs(np.mean(img[:, :, 0] - g)) < 10.0


def find_sticks(imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
    heat_map = None
    box_heights = None
    imgs_stats = []
    for i, img in enumerate(imgs):
        img = cv.cvtColor(cv.pyrDown(img), cv.COLOR_BGR2GRAY)
        d_img = cv.pyrUp(cv.pyrDown(img))
        d_img = cv.copyMakeBorder(d_img, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=0)
        if heat_map is None:
            heat_map = np.zeros(d_img.shape, dtype=np.uint8)
            box_heights = np.zeros(d_img.shape, dtype=np.uint16)
        prep, _ = img_as_ubyte(preprocess_phase(d_img))
        n, labels, stats, centroids = cv.connectedComponentsWithStats(prep)
        imgs_stats.append((n, labels, stats, centroids))
        for i in range(1, n):
            stat = stats[i]
            x = stat[cv.CC_STAT_LEFT] - 1
            y = stat[cv.CC_STAT_TOP] - 1
            w = stat[cv.CC_STAT_WIDTH] + 1
            h = stat[cv.CC_STAT_HEIGHT] + 1
            heat_map[y:y + h, x:x + w] += 1
            box_heights[y:y + h, x:x + w] = np.maximum(box_heights[y:y + h, x:x + w], h)

    box_heights = box_heights / np.max(box_heights)
    heat_map = heat_map * (1 + box_heights)
    _, stick_regions = cv.threshold(heat_map, len(imgs) - 1, 255.0, cv.THRESH_BINARY)

    match_lines_to_stick_regions(stick_regions, imgs_stats)


def match_lines_to_stick_regions(stick_regions: np.ndarray, imgs_stats: Tuple[int, np.ndarray, np.ndarray, np.ndarray]):
    # Find connected components in stick_regions
    n, label_img, stats, centroids = cv.connectedComponentsWithStats(stick_regions)
    labels = range(1, n)
    # Sort labels by the height in descending order, assuming that the longest regions represent stick regions
    labels = list(sorted(labels, key=lambda l: stats[l][cv.CC_STAT_HEIGHT], reverse=True))

    target_centroid = centroids[labels[0]]
    target_height = stats[labels[0]][cv.CC_STAT_HEIGHT]

    centroids_heights = np.array(list(map(lambda l: [centroids[l][0], centroids[l][1], stats[l][cv.CC_STAT_HEIGHT]],
                                          labels)))

    img = imgs_stats[1]
    img_labels = range(1, imgs_stats[0])
    img_stats = imgs_stats[2]
    img_centroids = imgs_stats[3]
    img_labels = list(sorted(img_labels, key=lambda l: img_stats[l][cv.CC_STAT_HEIGHT], reverse=True))

    img_centroids_heights = np.array(
        list(map(lambda l: [img_centroids[l][0], img_centroids[l][1], img_stats[l][cv.CC_STAT_HEIGHT]],
                 img_labels)))

    shifts: List[np.ndarray] = []
    errors: List[int] = []


def find_sticks(imgs: List[np.ndarray]) -> List[List[np.ndarray]]:
    heat_map = None
    box_heights = None
    imgs_stats = []
    for i, img in enumerate(imgs):
        img = cv.cvtColor(cv.pyrDown(img), cv.COLOR_BGR2GRAY)
        d_img = cv.pyrUp(cv.pyrDown(img))[:-50]
        d_img = cv.copyMakeBorder(d_img, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=0)
        if heat_map is None:
            heat_map = np.zeros(d_img.shape, dtype=np.uint8)
            box_heights = np.zeros(d_img.shape, dtype=np.uint16)
        prep, _ = img_as_ubyte(preprocess_phase(d_img))
        n, labels, stats, centroids = cv.connectedComponentsWithStats(prep)
        # imgs_stats.append((n, prep, stats, centroids))
        imgs_stats.append((1, prep))
        for i in range(1, n):
            stat = stats[i]
            x = stat[cv.CC_STAT_LEFT] - 1
            y = stat[cv.CC_STAT_TOP] - 1
            w = stat[cv.CC_STAT_WIDTH] + 1
            h = stat[cv.CC_STAT_HEIGHT] + 1
            heat_map[y:y + h, x:x + w] += 1
            box_heights[y:y + h, x:x + w] = np.maximum(box_heights[y:y + h, x:x + w], h)

    box_heights = box_heights / np.max(box_heights)
    heat_map = heat_map * (1 + box_heights)
    _, stick_regions = cv.threshold(heat_map.astype(np.uint8), len(imgs) - 2, 255, cv.THRESH_BINARY)

    # for i in range(len(imgs_stats)):
    #    match_lines_to_stick_regions(stick_regions, imgs_stats[i])

    return heat_map, stick_regions, imgs_stats


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


def filter_sticks(stick_img: np.ndarray, stick_img_stats: np.ndarray, stick_regions: np.ndarray) -> List[np.ndarray]:
    stick_regions_d = cv.dilate(stick_regions, cv.getStructuringElement(cv.MORPH_RECT, (5, 1)))
    n, labels, stats, _ = cv.connectedComponentsWithStats(stick_regions_d)

    labels_sticks: Dict[int, np.ndarray] = dict({})

    for l in range(1, stick_img_stats.shape[0]):
        stat = stick_img_stats[l]
        x = stat[cv.CC_STAT_LEFT]
        y = stat[cv.CC_STAT_TOP]
        w = stat[cv.CC_STAT_WIDTH]
        h = stat[cv.CC_STAT_HEIGHT]
        stick_box = stick_img[y:y + h, x:x + w]
        labels_box = labels[y:y + h, x:x + w]
        if not np.any(labels_box):
            continue
        non_zero_coords = np.argwhere(stick_box > 0)
        top_idx = np.argmin(non_zero_coords[:, 0])
        bottom_idx = np.argmax(non_zero_coords[:, 0])
        top = non_zero_coords[top_idx] + np.array(
            [y - 15, x - 15])  # TODO 15 is a border expansion, handle it maybe in the caller?
        bottom = non_zero_coords[bottom_idx] + np.array([y - 15, x - 15])
        target_label = np.max(labels_box)

        if target_label not in labels_sticks:
            labels_sticks[target_label] = np.array([[10000, 10000], [-10000, -10000]])
        label_entry = labels_sticks[target_label]

        if label_entry[0][0] > top[0]:
            label_entry[0] = top
        if label_entry[1][0] < bottom[0]:
            label_entry[1] = bottom
        labels_sticks[target_label] = label_entry

    return list(labels_sticks.values())


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


def is_endpoint_at(img: np.ndarray, x: int, y: int, s: int, top: bool) -> Tuple[bool, float]:
    """Analyzes whether there is an endpoint of a stick located at pixel (x,y) in `img`

    Parameters
    ----------
    img: np.ndarray
        image to look for endpoint in
    x, y: int
        coordinates where endpoint is being looked for
    s: int
        size of investigation window
    top: bool
        True if looking for top endpoint, False if looking for bottom endpoint
    """
    top_val = get_values_at(img, x, y, s)

    c = top_val[0]
    t = top_val[1]
    r = top_val[2]
    b = top_val[3]
    l = top_val[4]

    c_b_dist = np.linalg.norm(c - b)
    c_t_dist = np.linalg.norm(c - t)

    c_r_dist = np.linalg.norm(c - r)
    c_l_dist = np.linalg.norm(c - l)

    mean_dist = np.mean([c_b_dist, c_t_dist, c_r_dist, c_l_dist])
    if mean_dist < 7.0:
        return False, -1.0
    if top:
        return c_b_dist < mean_dist and c_t_dist > mean_dist and (c_r_dist > mean_dist or c_l_dist > mean_dist), mean_dist
    else:
        return c_t_dist < mean_dist and c_b_dist > mean_dist and (c_r_dist > mean_dist or c_l_dist > mean_dist), mean_dist


def adjust_endpoints(img: np.ndarray, stick: np.ndarray, w: int) -> np.ndarray:

    stick_length = np.linalg.norm(stick[0] - stick[1])
    stick_vec = (stick[0] - stick[1]) / stick_length

    top_endpoints_scores: List[Tuple[np.ndarray, float]] = []

    print(f'analyzing stick {stick}')

    endpoint_heat_map = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    endpoint_heat_map2 = np.zeros(endpoint_heat_map.shape, np.uint8)

    show = True
    k = 12
    for y in range(2 * stick[0][0] - w//1, 2 * stick[0][0] + w//1):
        for x in range(2 * stick[0][1] - w//1, 2 * stick[0][1] + w//1):
            to_show = img.copy()
            cv.circle(to_show, (2 * stick[0][1], 2 * stick[0][0]), 2, [255, 0, 0], 2)
            res, dist = is_endpoint_at(img, x, y, w, top=True)
            col = [0, 255, 0] if res else [0, 0, 255]
            cv.circle(to_show, (x, y), 2, col, 2)
            #if show:
            #    #cv.imshow('control', cv.resize(to_show, (0,0), fx=0.5, fy=0.5))
            #    #k = cv.waitKey(0)
            if k == ord('q'):
                show = False
                cv.destroyAllWindows()
            print(res)
            if res:
                candidate = np.array([y, x])
                vec = (candidate - 2 * stick[0])
                dist = np.linalg.norm(vec) #np.abs(np.linalg.norm(vec) - 2 * stick_length)
                if dist < w:
                    top_endpoints_scores.append((np.array([y, x]), dist))
                    endpoint_heat_map[y - w//2:y+w//2, x-w//2:x+w//2] += 1
                elif np.abs(np.dot(vec / np.linalg.norm(vec), stick_vec)) > 0.999:
                    top_endpoints_scores.append((np.array([y, x]), dist))
                    endpoint_heat_map[y - w//2:y+w//2, x-w//2:x+w//2] += 1

    bottom_endpoints_scores: List[Tuple[np.ndarray, float]] = []

    show = True
    for y in range(2 * stick[1][0] - w//1, 2 * stick[1][0] + w//1):
        for x in range(2 * stick[1][1] - w//1, 2 * stick[1][1] + w//1):
            to_show = img.copy()
            cv.circle(to_show, (2 * stick[1][1], 2 * stick[1][0]), 2, [255, 0, 0], 2)
            res, dist = is_endpoint_at(img, x, y, w, top=False)
            col = [0, 255, 0] if res else [0, 0, 255]
            cv.circle(to_show, (x, y), 2, col, 2)
            #if show:
            #    #cv.imshow('control', cv.resize(to_show, (0,0), fx=0.5, fy=0.5))
            #    #k = cv.waitKey(0)
            if k == ord('q'):
                show = False
                cv.destroyAllWindows()
            if res:
                candidate = np.array([y, x])
                vec = (candidate - 2 * stick[1])
                dist = np.linalg.norm(vec) #np.abs(np.linalg.norm(vec) - 2 * stick_length)
                if dist < w:
                    bottom_endpoints_scores.append((np.array([y, x]), dist))
                    endpoint_heat_map2[y - w // 2:y + w // 2, x - w // 2:x + w // 2] += 1
                elif np.abs(np.dot(vec / np.linalg.norm(vec), stick_vec)) > 0.999:
                    bottom_endpoints_scores.append((np.array([y, x]), dist))
                    endpoint_heat_map2[y - w//2:y+w//2, x-w//2:x+w//2] += 1

    _, _, _, te = cv.minMaxLoc(endpoint_heat_map)
    _, _, _, be = cv.minMaxLoc(endpoint_heat_map2)

    #new_top = 0.5 * max(top_endpoints_scores, key=lambda t: t[1])[0] if len(top_endpoints_scores) > 0 else stick[0]
    #new_bottom = 0.5 * max(bottom_endpoints_scores, key=lambda b: b[1])[0] if len(bottom_endpoints_scores) > 0 else stick[1]

    if te[0] == 0 and te[1] == 0:
        new_top = stick[0]
    else:
        new_top = np.round(0.5 * np.array([te[1] - w//2, te[0]])).astype(np.int32)
    if be[0] == 0 and be[1] == 0:
        new_bottom = stick[1]
    else:
        new_bottom = np.round(0.5 * np.array([be[1] + w//2, be[0]])).astype(np.int32)

    return np.array([np.round(new_top).astype(np.int32), np.round(new_bottom).astype(np.int32)])


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


def func1(img: np.ndarray, stick: np.ndarray) -> Tuple[np.ndarray, int]:
    mid = (0.5 * (stick[0] + stick[1])).astype(np.int32)

    size = 7
    for sz in range(7, 2, -2):
        non_zero = np.count_nonzero(img[mid[0]-sz//2:mid[0]+sz//2,mid[1]-sz//2:mid[1]+sz//2])
        if non_zero / (sz * sz) > 0.8:
            size = sz
            break
    return stick, size


def check_endpoints_(img: np.ndarray, sticks: List[Stick]) -> List[Tuple[bool, np.ndarray]]:
    founds = []

    padded = img[:,:,2] #cv.copyMakeBorder(img, 24, 24, 24, 24, cv.BORDER_REPLICATE)[:,:,2]
    print(padded.shape)
    locations = []
    start = time()
    for stick in sticks:
        left = min(stick.top[0] + 24, stick.bottom[0] + 24)
        right = max(stick.top[0] + 24, stick.bottom[0] + 24)

        #stick_roi = padded[stick.top[1] - 24 + 24: stick.bottom[1] + 24 + 24, left - 24: right + 24]
        #offset = (left - 24, stick.top[1] + 24 - 24)
        #print(stick.top)
        #print(stick.bottom)
        for y in range(stick.top[1] - 5, stick.top[1] + 5):
            for x in range(stick.top[0] - 5, stick.top[0] + 5):
                locations.append([x - 24, y - 24])

    descriptors = np.reshape(endpoint_hog.compute(padded, (8, 8), (48,48), locations=locations), (-1, endpoint_hog.getDescriptorSize()))
    predictions = endpoint_svc.predict(descriptors)
    #print(predictions)



def check_endpoints(img: np.ndarray, sticks: List[Stick]):

    locations = []
    offsets = []
    start = time()

    for stick in sticks:
        left = int(max(0, min(stick.top[0] - 24, stick.bottom[0] - 24)))
        right = int(min(img.shape[1]-1, max(stick.top[0] + 24, stick.bottom[0] + 24)))
        top = int(stick.top[1]) - 24
        bottom = int(stick.bottom[1]) + 24
        locations.clear()
        offsets.clear()
        roi = cv.copyMakeBorder(img[top:bottom, left:right, 2].copy(), 24, 24, 24, 24, cv.BORDER_REPLICATE)
        for y in range(-7, 8):
            for x in range(-7, 8):
                locations.append([int(stick.top[0]) + x - left, int(stick.top[1]) + y - top])
                offsets.append([x, y])
        start = time()
        descriptors = np.reshape(endpoint_hog.compute(roi, (8, 8), (0, 0), locations), (-1, endpoint_hog.getDescriptorSize()))
        predictions = np.reshape(endpoint_svc.predict(descriptors), (-1, 15))
        offsets_np = np.reshape(offsets, (-1, 15, 2))
        number_of_top_detections_per_row = np.sum(predictions == 0, axis=1)

        #if number_of_top_detections_per_row[5] >= 4:
        #    continue

        most_detections_row = np.argmax(number_of_top_detections_per_row)
        top_detections_x = np.argwhere(predictions[most_detections_row] == 0)
        top_offsets = np.reshape(offsets_np[most_detections_row, np.argwhere(predictions[most_detections_row] == 0)], (-1, 2))
        print(predictions)
        if top_offsets.shape[0] == 0:
            continue
        #top_offsets = np.array([offsets[i] for i in top_locations])
        top_location_offset = np.mean(top_offsets, axis=0).astype(np.int32)
        top_location = stick.top + top_location_offset
        print(top_location)
        draw = img.copy()
        draw2 = img.copy()
        cv.rectangle(draw, (int(top_location[0] - 3), int(top_location[1] - 3)), (int(top_location[0] + 3), int(top_location[1] + 3)), [255, 0, 0], 1)
        cv.rectangle(draw2, (int(stick.top[0] - 3), int(stick.top[1] - 3)), (int(stick.top[0] + 3), int(stick.top[1] + 3)), [0, 0, 255], 1)
        cv.imshow('img', draw)
        cv.imshow('img2', draw2)
        cv.waitKey(0)
    cv.destroyAllWindows()


def estimate_sticks_width(img: np.ndarray, sticks: List[Stick]) -> int:
    to_estimate = len(sticks)
    width = 3
    widths = [-1 for _ in sticks]

    if len(img.shape) > 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    while to_estimate > 0:
        erosion = cv.erode(gray, cv.getStructuringElement(cv.MORPH_RECT, (width, width)))

        for i, stick in enumerate(sticks):
            if widths[i] > 0:
                continue
            top_val = erosion[stick.top[1], stick.top[0]-3:stick.top[0]+3]
            bottom_val = erosion[stick.bottom[1], stick.bottom[0]-3:stick.bottom[0]+3]

            top_min, top_max = np.min(top_val), np.max(top_val)
            bottom_min, bottom_max = np.min(bottom_val), np.max(bottom_val)

            top_diff = top_max - top_min
            bottom_diff = bottom_max - bottom_min

            if top_diff < 30 and bottom_diff < 30:
                widths[i] = width
                to_estimate -= 1
        width += 2

    draw = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    for i, stick in enumerate(sticks):
        cv.line(draw, (int(stick.top[0]), int(stick.top[1])), (int(stick.bottom[0]), int(stick.bottom[1])),
                [255, 0, 0], 2)
        width = widths[i]
        cv.putText(draw, str(width), (int(stick.top[0]), int(stick.top[1])), cv.FONT_HERSHEY_PLAIN, 2.0, [0, 255, 0])

    cv.imshow('draw', draw)
    cv.waitKey(0)
    cv.destroyAllWindows()


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

gray_ = None
opeeen = None
tbb = None
sx = 5
sy = 5
sigma = 1.48
theta = 0.0
lambd = np.deg2rad(183)
gamma = 0.41
psi = np.deg2rad(36)
op__ = None
dx = None
dy = None
mag = None
va = None
va_ = None
def look_for_endpoints(img: np.ndarray, sticks: List[Stick]):
    #if len(img.shape) > 0:
    #    gray = img[:,:,1]
    #else:
    #    gray = img
    global gray_, opeeen, tbb, sx, sy, sigma, theta, lambd, gamma, psi, op__, dx, dy, mag, va, va_
    gray = img[:-100,:,2].copy() #cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.GaussianBlur(gray, (3, 3), sigmaX=1.5)
    gray_d = gray.copy() # cv.resize(gray, (0, 0), fx=0.5, fy=0.5)
    #gray = cv.resize(gray, (0, 0), fx=0.5, fy=0.5)
    print(gray.shape)
    #gray = cv.GaussianBlur(gray, (3, 3), sigmaX=0.5)
    #gray_ = cv.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST) #cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9))) #gray
    #gray_ = cv.resize(gray_, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    #gray = img_as_ubyte(skimage.filters.unsharp_mask(gray))
    #gray = clahe.apply(gray) #cv.equalizeHist(gray)
    #tbb = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, -6)
    #opeeen = cv.morphologyEx(tbb, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #opeeen = cv.morphologyEx(opeeen, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)))

    #_, tbb = cv.threshold(gray, 230, 255.0, cv.THRESH_BINARY)
    gray_ = gray #cv.erode(gray, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #gray = clahe.apply(gray)
    #bth = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_RECT, (19, 3)))
    #wth = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (19, 3)))
    #comb = cv.bitwise_or(bth, wth)
    brisk = cv.BRISK_create()
    start = time()

    detections, weights = top_endpoint_hog.detect(gray, padding=(24, 24))
    title = f'took {time() - start} secs'
    # print(np.hstack((detections, weights)))
    weights = np.reshape(weights, (-1,))
    print(weights.shape)
    if weights.shape[0] == 0:
        return
    likely = np.argmax(weights)
    print(likely)
    print(detections.shape)
    detections += 23

    ind = np.argwhere(weights > 3.0)
    print(ind)
    det = np.reshape(detections[ind], (-1, 2))
    waights = np.reshape(weights[ind], (-1,))

    #cv.namedWindow('magnitude')
    #cv.createTrackbar('Block_size', 'magnitude', 13, 100, on_trackbar_change)
    #cv.createTrackbar('C', 'magnitude', 44, 100, on_trackbar_change)

    #bl = cv.GaussianBlur(gray, (5, 5), sigmaX=1.5)
    #gray = clahe.apply(gray)

    #op = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (19, 1)))
    ##op = cv.morphologyEx(op, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    ##_, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #th = cv.adaptiveThreshold(op, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, -6)
    #th = cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)))

    #wth = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (19, 1)))
    #bth = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_RECT, (23, 1)))
    #op_ = cv.morphologyEx(wth, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
    #cl_ = cv.morphologyEx(op_, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 21)))
    #cl_ = cv.morphologyEx(cl_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 25)))

    #op_b = cv.morphologyEx(bth, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
    #cl_b = cv.morphologyEx(op_b, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 21)))
    #cl_b = cv.morphologyEx(cl_b, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 25)))

    #closed_ = cv.morphologyEx(gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
    #med = cv.medianBlur(gray, 3)
    #gray_clahe = clahe.apply(gray)
    start = time()
    #wth = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (17, 1)))
    #opp = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #gray_inv = 255 - gray
    #start_ = time()
    #th = cv.adaptiveThreshold(gray, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, -25)
    #th_asf = asf(th, 9, 1, mode='oco')
    #th_asf = cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)))
    #th_asf_wth = cv.morphologyEx(th_asf, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (9, 1)))
    #th_asf_wth = asf(th_asf_wth, 9, 1, 'oco')
    #print(f'asfing took {time() - start_} secs')
    #op__ = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #cv.namedWindow('gab_des')
    #cv.createTrackbar('sx', 'gab_des', sx, 200, on_trackbar_change)
    #cv.createTrackbar('sy', 'gab_des', sy, 200, on_trackbar_change)
    #cv.createTrackbar('sigma', 'gab_des', int(sigma * 100), 900, on_trackbar_change)
    #cv.createTrackbar('theta', 'gab_des', int(np.rad2deg(theta)), 360, on_trackbar_change)
    #cv.createTrackbar('lambda', 'gab_des', int(np.rad2deg(lambd)), 1000, on_trackbar_change)
    #cv.createTrackbar('gamma', 'gab_des', int(gamma * 100), 400, on_trackbar_change)
    #cv.createTrackbar('psi', 'gab_des', int(np.rad2deg(psi)), 360, on_trackbar_change)
    #gab = cv.getGaborKernel((sx, sx), sigma, theta, lambd, gamma, psi)
    ##gab = cv.resize(gab, (300, 300), interpolation=cv.INTER_CUBIC)
    #filo = cv.filter2D(op__, -1, gab)
    #cv.imshow('gab_des', gab)
    #cv.imshow('filo', filo)
    #cv.imshow('gray', gray) #cv.resize(gray, (0, 0), fx=0.5, fy=0.5))
    #cv.waitKey(0)

    #gab = cv.getGaborKernel((5, 15), 0.8, np.pi * 0, np.deg2rad(60), 0.0, psi=0.0)
    #gabi = img_as_ubyte(gab)
    #filo = cv.filter2D(gray, -1, gab)
    ##th = cv.adaptiveThreshold(filo, 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -6)
    ##_, th = cv.threshold(filo, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.imshow('gab', gab)
    #cv.imshow('gabi', gabi)
    #cv.imshow('filo', filo)
    #cv.imshow('th_filo', th)
    #enh = skimage.filters.rank.enhance_contrast(gray, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    #gr = cv.morphologyEx(gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #rec_start = time()
    #gr = skimage.morphology.reconstruction(cv.dilate(gray, cv.getStructuringElement(cv.MORPH_RECT, (1, 15))), gray, method='erosion').astype(np.uint8)
    #er = cv.erode(gray, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #op_rec = skimage.morphology.reconstruction(er, gray).astype(np.uint8)
    #print(f'rec took {time() - rec_start} secs')
    #cv.imshow('op_rec', op_rec)
    #cl_rec = cv.morphologyEx(gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    #h_max = skimage.morphology.reconstruction(np.maximum(0, cl_rec.astype(np.int16) - 50).astype(np.uint8), cl_rec).astype(np.uint8)

    #cv.imshow('cl_rec', cl_rec)
    #cv.imshow('h_max', h_max)
    #thrr = cv.adaptiveThreshold(op_rec, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, -20)
    #op_thrr = cv.morphologyEx(thrr, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))
    s_start = time()
    dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
    #dx = cv.resize(dx, (0, 0), fx=0.5, fy=0.5)
    #dy = cv.resize(dy, (0, 0), fx=0.5, fy=0.5)
    angle = cv.phase(dx, dy, None, True)
    mag = cv.magnitude(dx, dy)

    #cv.imshow('sobel_mag', cv.convertScaleAbs(mag))
    #op_ = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 15)))
    #cl_ = cv.morphologyEx(op_, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
    #op_ = cv.morphologyEx(cl_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 25)))
    #cl_ = cv.dilate(cl_, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))
    #cl_b = cv.dilate(cl_b, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))
    #th = cv.adaptiveThreshold(cl_, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 55, -10)
    #th_b = cv.bitwise_and(th, cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 1))))
    #th_bth = cv.adaptiveThreshold(cl_b, 255.0, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 55, -10)
    #th_bth_b = cv.bitwise_and(th_bth, cv.morphologyEx(th_bth, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 1))))
    #_, th2 = cv.threshold(cl_, 0, 255.0, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    #th = cv.morphologyEx(th, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (15, 1)))
    vert_angle = cv.inRange(angle, 150, 210)
    #vert_angle = skimage.filters.rank.percentile(vert_angle, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)), p0=0.3).astype(np.uint8)
    magi = cv.inRange(mag, 10, 2000)
    vert_angle_ = cv.bitwise_and(vert_angle, magi)
    vert_angle_0 = cv.inRange(angle, 0, 30)
    vert_angle_0 = cv.bitwise_or(vert_angle_0, cv.inRange(angle, 330, 360))

    va = vert_angle
    va_ = vert_angle_0

    print(f'sob took {time() - s_start} secs')


    #cv.imshow('gr', gr)
    #cv.imshow('op_rec', op_rec)
    #cv.imshow('thrr', thrr)
    #cv.imshow('op_thrr', op_thrr)

    r_start = time()
    r_h = 13
    r_w = 2
    p0 = 0.5
    p0 = 0.6
    rank_0 = cv.filter2D(vert_angle_0, cv.CV_16S, cv.getStructuringElement(cv.MORPH_RECT, (r_w, r_h)))
    rank_0 = cv.inRange(rank_0, r_w * p0 * r_h * 255, r_w * r_h * 255)
    rank = cv.filter2D(vert_angle, cv.CV_16S, cv.getStructuringElement(cv.MORPH_RECT, (r_w, r_h)))
    rank = cv.inRange(rank, r_w * p0 * r_h * 255, r_w * r_h * 255)
    rank_0 = cv.morphologyEx(rank_0, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))
    rank = cv.dilate(rank, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

    #cv.imshow('rank1', rank)
    #cv.imshow('rank2', rank_0)

    print(f'rank took {time() - r_start} secs')
    #cv.imshow('rank_0', rank_0) #cv.bitwise_and(rank, magi))
    #cv.imshow('rank', rank)


    #vert_angle_0 = skimage.filters.rank.percentile(vert_angle_0, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)), p0=0.3).astype(np.uint8)
    vert_angle_0_ = cv.bitwise_and(vert_angle_0, magi)

    #vert_angle_co = asf(vert_angle_, 9, 1)
    #vert_angle_oc = asf(vert_angle_, 5, 1, 'oco')
    #vert_angle_0_oc = asf(vert_angle_0_, 5, 1, 'oco')

    #cv.imshow('vert1', vert_angle_)
    #cv.imshow('vert2', vert_angle_0_)

    #vert_angle_oc = cv.morphologyEx(vert_angle_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)))
    m_start = time()
    hh = 7
    vert_angle_oc = cv.morphologyEx(rank, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, hh)))
    #vert_angle_0_oc = cv.morphologyEx(vert_angle_0_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)))
    vert_angle_0_oc = cv.morphologyEx(rank_0, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, hh)))

    w = 6
    dila1 = cv.dilate(vert_angle_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, 3)), anchor=(0, 1))
    dila2 = cv.dilate(vert_angle_0_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, 3)), anchor=(w-1, 1))
    dila_and = cv.bitwise_and(dila1, dila2)

    dila_1 = cv.dilate(vert_angle_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, 3)), anchor=(w-1, 1))
    dila_2 = cv.dilate(vert_angle_0_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, 3)), anchor=(0, 1))
    dila_and = cv.bitwise_or(dila_and, cv.bitwise_and(dila_1, dila_2))
    #marker = np.minimum(dila_and, cl_rec).astype(np.uint8)
    #rec_gray = img_as_ubyte(skimage.morphology.reconstruction(marker, cl_rec).astype(np.uint8))
    #cv.imshow('rec_gray', rec_gray)
    dila_and = cv.morphologyEx(dila_and, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    #label_img = skimage.measure.label(dila_and, connectivity=2)
    #reg_props = regionprops(label_img, intensity_image=angle)

    #cv.imshow('regs', dila_and)
    #cv.setMouseCallback('regs', on_click, param=(label_img, dila_and, reg_props))

    print(f'm took {time() - m_start} secs')

    print(f'preprocess for Hough took {time() - start} secs')
    #rank_or_rank = skimage.filters.rank.percentile(rank_or, cv.getStructuringElement(cv.MORPH_RECT, (7, 1)), p0=0.5)
    start = time()
    lines = cv.HoughLinesP(dila_and, 1.0, np.pi/180.0, 70, None, 30, 8)
    print(f'hough took {time() - start} secs')
    #cv.namedWindow('DI')
    #cv.createTrackbar('w', 'DI', 2, 30, on_trackbar_change2)
    #cv.createTrackbar('h', 'DI', 19, 30, on_trackbar_change2)
    #cv.createTrackbar('c', 'DI', 19, 50, on_trackbar_change2)
    #cv.imshow('DI', dila_and)
    line_img = img.copy() #cv.pyrDown(img) #img.copy()
    if lines is not None:
        for line_ in lines:
            line = line_[0]
            cv.line(line_img, (line[0], line[1]), (line[2], line[3]), [0, 255, 0], 2)

    draw = img.copy() #cv.pyrDown(img) #img.copy()
    #for p in det:
    #    cv.circle(draw, (int(p[0]), int(p[1])), 3, [0, 255, 0], 2)

    #kp_img = img.copy()
    ##op = cv.morphologyEx(cv.convertScaleAbs(gray, 0.5), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 21)))
    #op = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (25, 1)))
    #op = cv.morphologyEx(op, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))
    #kps = brisk.detect(op)
    #kp_img = cv.drawKeypoints(kp_img, kps, None)


    detection_boxes = detect_sticks_hog(gray)

    for i, (rect, rects) in enumerate(detection_boxes):
        cv.rectangle(draw, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), [0, 255, 0], 2)
        #angle_roi = angle[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        ##hist = np.histogram(angle_roi, bins=361)
        #fig = plt.figure()
        #fig.add_subplot(111)
        #plt.hist(angle_roi, bins=361)
        #fig.canvas.draw()
        #img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        #cv.imshow(f'box{i}', img)


    boxes = list(map(lambda rect_rects: rect_rects[0], detection_boxes))

    #hist = np.reshape(cv.calcHist([mag], [0], None, [np.max(mag) - np.min(mag)], [0, 256]), (-1,))
    #hist, hist_edges = np.histogram(mag, 200)
    #x_axis = [(b1 + b2) * 0.5 for b1, b2 in zip(hist_edges, hist_edges[1:])]
    #fig = plt.figure()
    #fig.add_subplot(111)
    #plt.bar(x_axis, hist)
    ###plt.hist(gray, bins=10)
    #fig.canvas.draw()
    #plt.show()

    #cv.imshow('kps', cv.resize(kp_img, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('op', cv.resize(op, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_asf', cv.resize(vert_asf, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank', cv.resize(rank, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_asf', cv.resize(rank_asf, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_asf_cl', cv.resize(rank_cl, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_asf_cl', cv.resize(vert_asf_cl, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('area', cv.resize(area_open, (0, 0), fx=0.5, fy=0.5))
    ##cv.imshow('closed_', cv.resize(closed_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('cl_', cv.resize(cl_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('op_', cv.resize(op_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('th', cv.resize(th, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('th_b', cv.resize(th_b, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('th2', cv.resize(th2, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('th_bth', cv.resize(th_bth, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('th_bth_b', cv.resize(th_bth_b, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('th_comb', cv.resize(th_comb, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_', cv.resize(vert_angle_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_asf_co', cv.resize(vert_angle_co, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_asf_oc', cv.resize(vert_angle_oc, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank', rank) #cv.resize(rank, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_cl', rank_cl) #cv.resize(rank_cl, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_', rank_) #cv.resize(rank_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_cl_', rank_cl_) #cv.resize(rank_cl_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_or', rank_or) #cv.resize(rank_or, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rank_or_rank', rank_or_rank) #cv.resize(rank_or_rank, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('reco', cv.resize(reconstruction, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_0_', cv.resize(vert_angle_0_, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('med', med)
    #cv.imshow('rank_dil', rank_dil2)
    #cv.imshow('rank_dil_', rank_dil_2)
    #cv.imshow('rank_and', rank_and)
    #cv.imshow('rec1', rec1)
    #cv.imshow('rec2', rec2)
    #cv.imshow('rec_or', rec_and)
    #cv.imshow('vert', vert_angle_) # cv.resize(vert_angle, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('vert_0', vert_angle_0_) # cv.resize(vert_angle_0, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('magi', magi)
    cv.imshow(title, cv.resize(draw, (0, 0), fx=.5, fy=.5)) #cv.resize(draw, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('rec_and_u', rec_and_u)
    #cv.imshow('vert_asf_cl', vert_asf_cl)
    #cv.imshow('vert_asf', vert_angle_oc)
    #cv.imshow('vert_asf_oco_0', vert_angle_0_oc)
    cv.imshow('vert_angle_', vert_angle_oc)
    cv.imshow('vert_angle_0_', vert_angle_0_oc)
    #cv.imshow('th', th)
    #cv.imshow('th_asf', th_asf)
    #cv.imshow('th_asf_wth', th_asf_wth)
    cv.imshow('dila_and', cv.resize(dila_and, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('gray_', gray_)
    #cv.imshow('tbb', tbb)
    #cv.imshow('magnitude', tbb)
    #cv.imshow('opeeen', opeeen)
    #cv.imshow('op', op)
    #cv.imshow('th', th)
    cv.imshow('lines', cv.resize(line_img, (0, 0), fx=0.5, fy=0.5)) #cv.resize(line_img, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('mag_th', cv.resize(mag_th, (0, 0), fx=0.5, fy=0.5))


    cv.waitKey(0)

    if len(boxes) > 1:
        matches = match_sticks_to_detections(sticks, boxes) #match_detections_with_sticks(sticks, boxes)
        draw2 = img.copy()
        for match in matches:
            stick1 = match[0]
            box1 = match[1]
            #stick2 = match[2]
            #box2 = match[3]
            color = [randrange(0, 256), randrange(0, 256), randrange(0, 256)]
            cv.line(draw2, (2 * int(stick1.top[0]), 2 * int(stick1.top[1])),
                    (2 * int(stick1.bottom[0]), 2 * int(stick1.bottom[1])), color, 2)
            #cv.line(draw2, (2 * int(stick2.top[0]), 2 * int(stick2.top[1])),
            #        (2 * int(stick2.bottom[0]), 2 * int(stick2.bottom[1])), [255, 0, 0], 2)
            cv.rectangle(draw2, (box1[0], box1[1]), (box1[0]+box1[2], box1[1]+box1[3]), color, 2)
            #cv.rectangle(draw2, (box2[0], box2[1]), (box2[0]+box2[2], box2[1]+box2[3]), [255, 0, 0], 2)
        cv.imshow('matches', cv.resize(draw2, (0, 0), fx=0.5, fy=0.5))
        cv.waitKey()



    cv.destroyAllWindows()
    return

    cv.imshow('img', img)
    for stick in sticks:
        left = int(max(0, min(2 * stick.top[0] - 24, 2 * stick.bottom[0] - 24)))
        right = int(min(img.shape[1] - 1, max(2 * stick.top[0] + 24, 2 * stick.bottom[0] + 24)))
        top = int(2 * stick.top[1]) - 24
        bottom = int(2 * stick.bottom[1]) + 24

        w = right - left
        h = bottom - top

        w_padding = int((8 * (np.ceil(w / 8))) - w)
        h_padding = int((8 * (np.ceil(h / 8))) - h)

        w += w_padding
        h += h_padding

        left_padding = w_padding // 2
        right_padding = int(np.ceil(w_padding / 2))
        top_padding = h_padding // 2
        bottom_padding = int(np.ceil(h_padding / 2))

        #if left - left_padding < 0:
        #    right_padding += left_padding
        #else:
        #    left -= left_padding
        #
        #right += right_padding
        #



        #locations.clear()
        #offsets.clear()
        roi = cv.copyMakeBorder(img[top:bottom,left:right].copy(),
                                top_padding,
                                bottom_padding,
                                left_padding,
                                right_padding, cv.BORDER_REPLICATE)


        #roi = cv.erode(roi, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))
        #roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 11)))
        #roi = cv.morphologyEx(roi, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 11)))

        edges = skimage.filters.sobel(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
        low = 0.1
        high = 0.25

        #tr_th = skimage.filters.threshold_triangle(roi)

        r_g = roi[:,:,2] #cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        h_max = (skimage.morphology.reconstruction(np.maximum(r_g.astype(np.int16) - 35, 0), r_g)).astype(np.uint8)
        h_convex = (3 * (r_g - h_max)).astype(np.uint8)
        h_convex = cv.dilate(h_convex, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))

        r_g = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        mag = cv.magnitude(cv.Sobel(r_g, cv.CV_32F, 1, 0), cv.Sobel(r_g, cv.CV_32F, 0, 1))
        mag = cv.convertScaleAbs(mag)

        #cv.imshow('mag', mag)


        tr_th = skimage.filters.threshold_local(h_convex, 15, offset=-3)
        #_, th_roi = cv.threshold(cv.cvtColor(r_g, tr_th, 255.0, cv.THRESH_BINARY)
        th_roi = cv.convertScaleAbs(255 * (h_convex > tr_th))

        h_min = (skimage.morphology.reconstruction(r_g.astype(np.uint16) + 35, r_g, method='erosion')).astype(np.uint8)
        h_concave = (3 * (h_min - r_g)).astype(np.uint8)
        h_concave = cv.dilate(h_concave, cv.getStructuringElement(cv.MORPH_RECT, (3, 1)))

        #detections, weights = top_endpoint_hog.detect(roi, padding=(0, 0))
        detections, weights = top_endpoint_hog.detect(gray)
        #print(np.hstack((detections, weights)))
        weights = np.reshape(weights, (-1,))
        print(weights.shape)
        if weights.shape[0] == 0:
            continue
        likely = np.argmax(weights)
        print(likely)
        print(detections.shape)
        detections += 23

        ind = np.argwhere(weights > 3.0)
        print(ind)
        det = np.reshape(detections[ind], (-1, 2))
        waights = np.reshape(weights[ind], (-1,))

        draw = img.copy()
        for p in det:
            cv.circle(draw, (int(p[0]), int(p[1])), 3, [0, 255, 0], 2)
        cv.imshow('draw', draw)
        cv.waitKey(0)
        continue

        print(f'd1 {detections.shape} det {det.shape}')

        heat_map = 0 * np.ones((roi.shape[0], roi.shape[1]), dtype=np.float32)
        heat_map[det[:,1], det[:,0]] = waights
        heat_map[int(0.5 * heat_map.shape[0]):,:] = 0.0

        #dx = cv.Sobel(roi, cv.CV_32F, 1, 0)
        #mag = cv.magnitude(cv.Sobel(roi, cv.CV_32F, 1, 0), np.zeros(roi.shape, dtype=np.float32))
        #
        #heat_map *= mag

        added = cv.filter2D(heat_map, cv.CV_32F, cv.getStructuringElement(cv.MORPH_RECT, (17, 17)), delta=-0.005, borderType=cv.BORDER_CONSTANT)
        #added += mag

        _, th = cv.threshold(heat_map, 0, 255.0, cv.THRESH_BINARY)
        cv.imshow('th', (th.astype(np.uint8)))

        _, _, _, pos = cv.minMaxLoc(added)

        #pos = detections[likely]
        print(f'likely pos is {pos}')


        #lowt = (edges > low).astype(int)
        #hight = (edges > high).astype(int)

        hyst = 255 * skimage.filters.apply_hysteresis_threshold(edges, low, high).astype(np.uint8)

        cv.imshow('h_convex', h_convex.astype(np.uint8))
        cv.imshow('h_max', h_max)
        #cv.imshow('h_min', h_min)
        cv.imshow('h_concave', h_concave)
        cv.imshow('dil', roi)
        #cv.imshow('tr_th', th_roi)

        prep = preprocess_phase(h_convex)

        roi = cv.copyMakeBorder(img[top:bottom,left:right].copy(),
                                top_padding,
                                bottom_padding,
                                left_padding,
                                right_padding, cv.BORDER_REPLICATE)

        cv.circle(roi, (int(pos[0]), int(pos[1])), 3, [0, 255, 0])

        #rows = (h) // 8
        #windows_per_row = (w) // 8

        #top_responses = np.argwhere(predictions == 0)
        #bottom_responses = np.argwhere(predictions == 1)

        #roi = cv.copyMakeBorder(img[top:bottom,left:right].copy(),
        #                        top_padding,
        #                        bottom_padding,
        #                        left_padding,
        #                        right_padding, cv.BORDER_REPLICATE)


        cv.imshow('roi', roi)
        #cv.imshow('hmt', img_as_ubyte(prep[0]))
        cv.waitKey(0)
        #for y in range(-7, 8):
        #    for x in range(-7, 8):
        #        locations.append([int(stick.top[0]) + x - left, int(stick.top[1]) + y - top])
        #        offsets.append([x, y])
    cv.destroyAllWindows()


def detect_sticks_hog(img: np.ndarray, threshold: float) -> List[Tuple[Rectangle, List[Rectangle]]]:

    #bth = cv.morphologyEx(img, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_RECT, (15, 3)))
    #wth = cv.morphologyEx(img, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (15, 3)))
    #comb = cv.bitwise_or(bth, wth)

    found, weights = hog_desc.detect(img, padding=(STICK_WINDOW[0], STICK_WINDOW[1]//2))
    #valid_indices = np.nonzero(weights > 0.5)[0]
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

def match_detections_with_sticks(sticks: List[Stick], detections: List[Rectangle]) -> List[Tuple[Stick, Rectangle]]: #List[Tuple[Stick, Rectangle, Stick, Rectangle]]:
    differences = []
    detection_pairs = []

    for i in range(len(detections)):
        rect1 = detections[i]
        for j in range(len(detections)):
            if i == j:
                continue
            rect2 = detections[j]
            differences.append([rect2[0] - rect1[0], rect2[1] - rect1[1]])
            detection_pairs.append((i, j))

    differences = np.array(differences)

    stick_differences = []
    stick_pairs = []

    for i in range(len(sticks)):
        for j in range(len(sticks)):
            if i == j:
                continue
            stick_pairs.append((i, j))
            vec = 2 * sticks[j].top - 2 * sticks[i].top
            stick_differences.append(np.linalg.norm(differences - vec, axis=1))

    stick_differences = np.array(stick_differences)

    argmins = np.argmin(stick_differences, axis=1)

    matches: Dict[int, List[int]] = dict({})
    for stick_pair, argmin in zip(stick_pairs, argmins):
        stick1 = sticks[stick_pair[0]]
        stick2 = sticks[stick_pair[1]]
        box_pair = detection_pairs[argmin]
        #matches.append((stick1, detections[box_pair[0]], stick2, detections[box_pair[1]]))
        if stick_pair[0] not in matches:
            matches[stick_pair[0]] = []
        if stick_pair[1] not in matches:
            matches[stick_pair[1]] = []

        matches[stick_pair[0]].append(box_pair[0])
        matches[stick_pair[1]].append(box_pair[1])

    return list(map(lambda sid_bid: (sticks[sid_bid[0]], detections[statistics.mode(sid_bid[1])]), matches.items()))

    #return matches

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
def find_sticks(img: np.ndarray, hog_th: float, w: int, h, p0: float):

    #gray = img[:-100,:,2].copy() #cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)[:-100,:]
    #gray = cv.resize(gray, (0, 0), fx=0.5, fy=0.5)

    #start = time()
    #detection_boxes = detect_sticks_hog(gray, hog_th)
    #wth = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (55, 55)))
    #print(f'detection took {time() - start} secs')
    draw = img.copy()
    #hyst = skimage.filters.apply_hysteresis_threshold(op, 130, 180)
    #hyst_gray = np.where(hyst, gray, 0).astype(np.uint8)
    C = params['C']
    blockSize = params['blockSize']
    if blockSize % 2 == 0:
        blockSize += 1
    op1_h = params['op1_h']
    op2_h = params['op2_h']
    cl1_h = params['cl1_h']
    cl2_h = params['cl2_h']
    asf_sz = params['asf_sz']
    asf_mode = params['asf_mode']
    method = cv.ADAPTIVE_THRESH_GAUSSIAN_C if params['method'] == 'gauss' else cv.ADAPTIVE_THRESH_MEAN_C

    start = time()
    #wth = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_RECT, (21, 1)))
    wth = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 25)))
    wth_ = cv.morphologyEx(gray, cv.MORPH_TOPHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE, (55, 55)))
    wth_ = cv.morphologyEx(wth_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
    #_, th_ = cv.threshold(wth_, 0, 255.0, cv.THRESH_BINARY | cv.THRESH_OTSU)
    th_ = cv.adaptiveThreshold(wth_, 255.0, method, cv.THRESH_BINARY, blockSize, C)
    bth = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_RECT, (21, 1)))
    th2 = cv.adaptiveThreshold(wth, 255, method, cv.THRESH_BINARY, blockSize, C)
    th = cv.adaptiveThreshold(bth, 255, method, cv.THRESH_BINARY, blockSize, C)
    #op = cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
    #op = cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 25)))
    op = cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, op1_h)))
    op = cv.morphologyEx(op, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, cl1_h)))
    #op = asf(op, 15, 1, 'oco')
    #op = skimage.morphology.opening(op, selem=cv.getStructuringElement(cv.MORPH_RECT, (1, op2_h)))
    #op = asf(op, asf_sz, 1, asf_mode)
    #op = skimage.morphology.closing(op, selem=cv.getStructuringElement(cv.MORPH_RECT, (5, cl2_h)))

    op2 = cv.morphologyEx(th2, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, op1_h)))
    op2 = cv.morphologyEx(op2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, cl1_h)))
    #o2p = asf(op, 15, 1, 'oco')
    #op2 = skimage.morphology.opening(op2, selem=cv.getStructuringElement(cv.MORPH_RECT, (1, op2_h)))
    #op2 = asf(op2, asf_sz, 1, asf_mode)
    #op2 = skimage.morphology.closing(op2, selem=cv.getStructuringElement(cv.MORPH_RECT, (5, cl2_h)))

    op_or = np.bitwise_or(op, op2)
    #op_or = cv.morphologyEx(op_or, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 1), anchor=(0,0)))
    print(f'it took {time() - start} secs')
    lab = skimage.measure.label(op_or)
    #print(f'min of label is {np.min(lab)}, max is {np.max(lab)}')
    overlaid = img_as_ubyte(label2rgb(lab, image=img[:-100,:], bg_label=0))

    f = 0.5
    cv.imshow('gray', cv.resize(gray, (0, 0), fx=f, fy=f))
    cv.imshow('th', cv.resize(th, (0, 0), fx=f, fy=f))
    cv.imshow('th2', cv.resize(th2, (0, 0), fx=f, fy=f))
    cv.imshow('op2', cv.resize(op2, (0, 0), fx=f, fy=f))
    cv.imshow('op', cv.resize(op, (0, 0), fx=f, fy=f))
    cv.imshow('overlaid', cv.resize(overlaid, (0, 0), fx=f, fy=f))
    cv.imshow('open', cv.resize(wth_, (0, 0), fx=f, fy=f))
    cv.imshow('th_ot', cv.resize(th_, (0, 0), fx=f, fy=f))

    cv.waitKey(0)
    cv.destroyAllWindows()
    return

    for i, (rect, rects) in enumerate(detection_boxes):
        #width = estimate_stick_width(gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
        if rect[0]+rect[2] >= gray.shape[1] or rect[0] < 0 or rect[1]+rect[3] >= gray.shape[0] or rect[1] < 0:
            continue
        roi = gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cv.rectangle(draw, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), [0, 0, 255], 2)
        if np.max(roi) - np.min(roi) > 50:
            cv.rectangle(draw, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), [0, 255, 0], 2)
        #cv.putText(draw, f's: {width}', (rect[0], rect[1]), cv.FONT_HERSHEY_DUPLEX, 1.5, [255, 0, 0])
        #cv.line(draw, (rect[0], rect[1]), (rect[0] + width, rect[1]), [0, 0, 255], 5)

    boxes = list(map(lambda rect_rects: rect_rects[0], detection_boxes))

    dx, dy = cv.Sobel(op, cv.CV_32F, 1, 0), cv.Sobel(op, cv.CV_32F, 0, 1)
    angle = cv.phase(dx, dy, None, True)
    mag = cv.magnitude(dx, dy)

    #mag_hmin = skimage.morphology.reconstruction(mag + 50, mag, method='erosion').astype(np.uint8)
    #ws = skimage.segmentation.watershed(mag_hmin)
    #segs = cv.cvtColor(img_as_ubyte(label2rgb(ws, gray)), cv.COLOR_RGB2BGR)


    vert_angle = cv.inRange(angle, 150, 210)
    magi = cv.inRange(mag, 10, 2000)
    vert_angle_ = cv.bitwise_and(vert_angle, magi)
    vert_angle_0 = cv.inRange(angle, 0, 30)
    vert_angle_0 = cv.bitwise_or(vert_angle_0, cv.inRange(angle, 330, 360))

    va = vert_angle
    va_ = vert_angle_0

    r_h = 9
    r_w = 1
    #p0 = 0.5
    #p0 = 0.7

    rank_0 = cv.filter2D(vert_angle_0, cv.CV_16S, cv.getStructuringElement(cv.MORPH_RECT, (r_w, r_h)))
    rank_0 = cv.inRange(rank_0, r_w * p0 * r_h * 255, r_w * r_h * 255)
    rank = cv.filter2D(vert_angle, cv.CV_16S, cv.getStructuringElement(cv.MORPH_RECT, (r_w, r_h)))
    rank = cv.inRange(rank, r_w * p0 * r_h * 255, r_w * r_h * 255)
    rank_0 = np.minimum(cv.morphologyEx(rank_0, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (r_w, r_h))),
                        rank_0)
    rank = np.minimum(cv.dilate(rank, cv.getStructuringElement(cv.MORPH_RECT, (r_w, r_h))),
                      rank)

    vert_angle_0_ = cv.bitwise_and(vert_angle_0, magi)

    hh = h
    vert_angle_oc = cv.morphologyEx(rank, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, hh)))
    #vert_angle_0_oc = cv.morphologyEx(vert_angle_0_, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 7)))
    vert_angle_0_oc = cv.morphologyEx(rank_0, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, hh)))

    #w = 3
    dila1 = cv.dilate(vert_angle_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, h)), anchor=(0, (h-1) // 2))
    dila2 = cv.dilate(vert_angle_0_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, h)), anchor=(w-1, (h-1) // 2))
    dila_and = cv.bitwise_and(dila1, dila2)

    dila_1 = cv.dilate(vert_angle_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, h)), anchor=(w-1, (h-1) // 2))
    dila_2 = cv.dilate(vert_angle_0_oc, cv.getStructuringElement(cv.MORPH_RECT, (w, h)), anchor=(0, (h-1) // 2))
    dila_and = cv.bitwise_or(dila_and, cv.bitwise_and(dila_1, dila_2))

    dila_and = cv.morphologyEx(dila_and, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    lines = cv.HoughLinesP(dila_and, 1.0, np.pi/180.0, 70, None, 30, 8)

    line_img = img.copy() #cv.pyrDown(img) #img.copy()
    if lines is not None:
        for line_ in lines:
            line = line_[0]
            cv.line(line_img, (line[0], line[1]), (line[2], line[3]), [0, 255, 0], 2)

    f = 0.5
    cv.imshow('gray', cv.resize(gray, (0, 0), fx=f, fy=f))
    cv.imshow('lines', cv.resize(line_img, (0, 0), fx=f, fy=f))
    cv.imshow('detections', cv.resize(draw, (0, 0), fx=f, fy=f))
    cv.imshow('line_regions', cv.resize(dila_and, (0, 0), fx=f, fy=f))
    cv.imshow('th', cv.resize(th, (0, 0), fx=f, fy=f))
    cv.imshow('op', cv.resize(op, (0, 0), fx=f, fy=f))
    cv.imshow('overlaid', cv.resize(overlaid, (0, 0), fx=f, fy=f))
    #cv.imshow('segs', cv.resize(segs, (0, 0), fx=f, fy=f))
    #cv.waitKey(0)
    #cv.destroyAllWindows()

def estimate_stick_width(stick_roi: np.ndarray) -> int:
    max_loss = 0
    max_loss_size = 0

    current_size = 3
    current_img = stick_roi.copy()

    while np.max(current_img) - np.min(current_img) > 80:
        er = cv.erode(current_img, cv.getStructuringElement(cv.MORPH_ELLIPSE, (current_size, current_size)))
        #current_loss = np.max(er) - np.min(er)
        #if current_loss > max_loss:
        #    max_loss = current_loss
        #    max_loss_size = current_size
        current_img = er
        current_size += 2

    return current_size

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
    #th = cv.morphologyEx(th, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 1)))
    #print(f'theyre equal? = {np.all(th_ == th)}')
    #star_c = time()
    #_ = cv.connectedComponentsWithStats(th)
    #print(f'cc took {time() - star_c} secs')


    #return lines
    if not debug:
        return th
    start_hough = time()
    lines = cv.HoughLinesP(th_, 1.0, np.pi / 180.0, params['hough_th'], None, params['line_length'], params['line_gap'])
    print(f'hough took {time() - start_hough} secs')
    print(f'it took {time() - start} secs')
    f = 1.0
    draw = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    draw2 = draw.copy()
    f = 1
    if lines is not None:
        lines = np.reshape(lines, (-1, 2, 2))
        lines_ = cluster_liness(lines, draw, debug)
        for line_ in lines_:
            p = line_[0]
            q = line_[1]
            cv.line(draw, (int(f * p[0]), int(f * p[1])), (int(f * q[0]), int(f * q[1])), [0, 255, 0], 1)
        for line_ in lines:
            p = line_[0]
            q = line_[1]
            cv.line(draw2, (int(f * p[0]), int(f * p[1])), (int(f * q[0]), int(f * q[1])), [0, 255, 0], 1)
        visualize_line_signals(lines_, orig, gray)

    cv.imshow('uhmt_th', 255 * th)
    cv.imshow('gray', gray)
    cv.imshow('gray_eq', gray_eq)
    cv.imshow('lines', draw2)
    #cv.imshow('lines', cv.resize(draw2, (0, 0), fx=0.5, fy=0.5))
    #cv.imshow('lines_clustered', cv.resize(draw, (0, 0), fx=0.5, fy=0.5))
    cv.imshow('lines_clustered', draw)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #th_idx = 0
    #th_to_show = th1
    #cv.imshow('th_i', 1 * th_to_show)
    #print(f'type is {th1.dtype}')
    #key = cv.waitKey(0)
    #while key != ord('q'):
    #    if key == ord('n'):
    #        th_idx = min(th_idx + 1, 3)
    #    elif key == ord('p'):
    #        th_idx = max(th_idx - 1, 0)
    #    if th_idx == 0:
    #        th_to_show = th1
    #    elif th_idx == 1:
    #        th_to_show = th2
    #    elif th_idx == 2:
    #        th_to_show = th3
    #    elif th_idx == 3:
    #        th_to_show = th4
    #    cv.imshow('th_i', 1 * th_to_show)
    #    key = cv.waitKey(0)
    #cv.destroyAllWindows()

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
        lines_ = cluster_liness(lines, draw, False)
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


def get_polar_coordinates(line: np.ndarray, img: np.ndarray) -> List[np.ndarray]:
    # v = line[0] - line[1]
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
    box_points = cv.boxPoints(rect).astype(np.int32)

    draw2 = img.copy()

    for i in range(4):
        p = box_points[i]
        if i == 3:
            q = box_points[0]
        else:
            q = box_points[i+1]
        cv.line(draw2, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), [255, 0, 0], 1)
        cv.line(draw2, (int(line[0,0]), int(line[0,1])), (int(line[1,0]), int(line[1,1])), [0, 255, 0], 1)

    theta = math.atan2(n[1], n[0])

    rho = line[0, 0] * math.cos(theta) + line[0, 1] * math.sin(theta)

    top = np.array([0, 0])
    bottom = np.array([img.shape[1] - 1, img.shape[0] - 1])

    if np.abs(theta) < 0.001:
        top[0] = rho
        bottom[0] = rho
    elif np.abs(np.abs(theta) - 0.5 * np.pi) < 0.001:
        top[1] = rho
        bottom[1] = rho
    else:
        top[0] = rho / math.cos(theta)
        if top[0] < 0:
            top[0] = 0
            top[1] = rho / math.sin(theta)
        elif top[0] >= img.shape[1]:
            top[0] = img.shape[1] - 1
            top[1] = (rho - top[0] * math.cos(theta)) / math.sin(theta)
        bottom[0] = (rho - bottom[1] * math.sin(theta)) / math.cos(theta)
        if bottom[0] >= img.shape[1]:
            bottom[0] = img.shape[1] - 1
            bottom[1] = (rho - bottom[0] * math.cos(theta)) / math.sin(theta)

    draw = None

    draw = img.copy()
    cv.line(draw, (int(line[0, 0]), int(line[0, 1])),
            (int(line[1, 0]), int(line[1, 1])), [255, 0, 0], 2)
    cv.line(draw, (int(top[0]), int(top[1])), (int(bottom[0]), int(bottom[1])), [0, 255, 0], 1)

    return np.array([theta, rho]), draw2

def cluster_lines(lines: np.ndarray, img, debug: bool) -> List[np.ndarray]:
    polar_space: List[Tuple[np.ndarray, bool]] = []

    #polar_space = list(map(lambda line: (get_polar_coordinates(line, img), False), lines))
    for line in lines:
        polar, draw = get_polar_coordinates(line, img)
        polar_space.append((polar, False))
        if draw is not None and debug:
            cv.imshow('jf', draw)
            cv.waitKey(0)
            cv.destroyWindow('jf')

    line_groups: List[List[int]] = []

    for i, p_line_grouped in enumerate(polar_space):
        if p_line_grouped[1]:
            continue
        line_i = p_line_grouped[0]
        idx = len(line_groups)
        line_groups.append([i])
        polar_space[i] = (polar_space[i][0], True)

        for j in range(i+1, len(polar_space)):
            line_j = polar_space[j][0]

            if np.linalg.norm(line_i - line_j) < 5.0:
                line_groups[idx].append(j)
                polar_space[j] = (polar_space[j][0], True)

    result_lines: List[np.ndarray] = []
    for group in line_groups:
        x = np.array([9000, -1])
        y = np.array([9000, -1])

        for line_id in group:
            line = lines[line_id]
            if line[0][0] < x[0]:
                x[0] = line[0][0]
            if line[0][0] > x[1]:
                x[1] = line[0][0]
            if line[1][0] < x[0]:
                x[0] = line[0][0]
            if line[1][0] > x[1]:
                x[1] = line[1][0]

            if line[0][1] < y[0]:
                y[0] = line[0][1]
            if line[0][1] > y[1]:
                y[1] = line[0][1]
            if line[1][1] < y[0]:
                y[0] = line[1][1]
            if line[1][1] > y[1]:
                y[1] = line[1][1]

        result_lines.append(np.array([[x[0], y[0]], [x[1], y[1]]]))

    return result_lines

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

def cluster_liness(lines: np.ndarray, img: np.ndarray, debug: bool) -> np.ndarray:
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

def line_upright_bbox(line: np.ndarray, width = 25) -> np.ndarray:
    w = max(1, int((width - 1) / 2))
    return np.array([np.min(line, axis=0) - w, np.max(line, axis=0) + w])

def visualize_line_signals(lines: np.ndarray, img: np.ndarray, gray: np.ndarray):
    print('hello')
    line_map = -1 * np.ones((img.shape[0], img.shape[1]), np.int8)

    #draw = cv.pyrDown(img)
    draw = img.copy()
    for i, line in enumerate(lines):
        bbox = line_upright_bbox(line, 25)
        line_map[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]] = i
        mag = cv.magnitude(cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1))
        mag_lines = skimage.measure.profile_line(mag, line[0,::-1], line[1,::-1], linewidth=15, reduce_func=None)
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
    cv.setMouseCallback('lines', line_click_handler, (line_map, lines, gray_up, hsv))
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
    line_id = line_map[y,x]
    if line_id < 0:
        return
    line = 2 * lines[line_id]
    bbox = line_upright_bbox(line, width=50)
    line_roi = img[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]


    hsv_line_roi = hsv[bbox[0,1]:bbox[1,1],bbox[0,0]:bbox[1,0]]

    #line_roi = cv.equalizeHist(line_roi)
    line = line - bbox[0]
    hsv_prof_line = np.array(skimage.measure.profile_line(hsv_line_roi, line[0,::-1] - np.array([25, 0]), line[1,::-1] + np.array([25, 0]), linewidth=25, reduce_func=None), np.uint8)

    prof_shape = (hsv_prof_line.shape[0], hsv_prof_line.shape[1])
    hsv_signal = np.dstack(((360.0 * (hsv_prof_line[:,:,0] / 255.0)).astype(np.float32), np.ones(prof_shape, np.float32), np.ones(prof_shape, np.float32)))
    sat_signal = np.dstack(((360.0 * (hsv_prof_line[:,:,1] / 255.0)).astype(np.float32), np.ones(prof_shape, np.float32), np.ones(prof_shape, np.float32)))
    val_signal = np.dstack(((360.0 * (hsv_prof_line[:,:,2] / 255.0)).astype(np.float32), np.ones(prof_shape, np.float32), np.ones(prof_shape, np.float32)))

    hsv_signal = cv.convertScaleAbs(cv.cvtColor(hsv_signal, cv.COLOR_HSV2BGR), alpha=255)
    sat_signal = cv.convertScaleAbs(cv.cvtColor(sat_signal, cv.COLOR_HSV2BGR), alpha=255)
    val_signal = cv.convertScaleAbs(cv.cvtColor(val_signal, cv.COLOR_HSV2BGR), alpha=255)

    cv.imshow('hsv_s', hsv_signal)
    cv.imshow('sat_s', sat_signal)
    cv.imshow('val_s', val_signal)

    dx, dy = cv.Sobel(line_roi, cv.CV_32F, 1, 0), cv.Sobel(line_roi, cv.CV_32F, 0, 1)
    mag = cv.magnitude(dx, dy)
    mag /= np.max(mag)

    angles = cv.phase(dx, dy, None, True)
    angle = line_angle(line, True)

    edge_pos = line_edge_offsets(line, mag)
    is_line_on_stick(line, edge_pos, angles)

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
    hsv = np.dstack((angles, np.ones(angles.shape, np.float32), mag)) #np.ones(angles.shape, np.float32)))
    bgr = cv.convertScaleAbs(cv.cvtColor(hsv, cv.COLOR_HSV2BGR), None, 255)
    prof_line = np.array(skimage.measure.profile_line(bgr, line[0,::-1] - np.array([12, 0]), line[1,::-1] + np.array([12, 0]), linewidth=25, reduce_func=None), np.uint8)
    viz = visualize_signal(prof_line)
    line_viz_enlarged = cv.resize(bgr, (0, 0), fx=3, fy=3, interpolation=cv.INTER_NEAREST)
    cv.imshow('line_signal', line_viz_enlarged)
    cv.setMouseCallback('line_signal', pick_angle, (angles, angle,))
    cv.imshow('line_roi', line_roi)
    cv.imshow('good', good_angles)
    cv.imshow('mag', cv.convertScaleAbs(mag, alpha=255))

def visualize_signal(signal: np.ndarray) -> np.ndarray:
    if signal.ndim > 1:
        return signal.astype(np.uint8)
    viz = np.zeros((255, signal.shape[0]), np.uint8)
    for t, v in enumerate(signal):
        viz[255 - v:255,t] = 255
    return viz

def line_angle(line: np.ndarray, normal: bool = False) -> float:
    v = line[0] - line[1]
    return math.atan2(v[1], v[0])
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
    print(f'angle is {angles[y // 3,x // 3]} and norma line angle is {params[1]}')
    print(f'range to check is {((angle - 5) % 360, (angle + 5) % 360)}, {((angle + 175) % 360, (angle + 185) % 360)}')

def line_edge_offsets(line: np.ndarray, mag: np.ndarray) -> List[int]:
    bbox = line_upright_bbox(line, 17)
    mag_lines = skimage.measure.profile_line(mag, line[0, ::-1], line[1, ::-1], linewidth=15, reduce_func=None)
    line_sums = np.sum(mag_lines, axis=0)
    left_edge_idx = np.argmax(line_sums[::-1][8:]) + 1
    right_edge_idx = np.argmax(line_sums[8:]) + 1

    return [left_edge_idx, right_edge_idx]

def is_line_on_stick(line: np.ndarray, edge_pos: List[int], angles: np.ndarray) -> bool:
    left_angles = skimage.measure.profile_line(angles,
                                               line[0,::-1] - [0, edge_pos[0]],
                                               line[1,::-1] - [0, edge_pos[0]], mode='reflect')
    left_angles = np.mod(left_angles + 180, 180)
    left_angles = np.where(left_angles < 100, left_angles + 180, left_angles)
    right_angles = skimage.measure.profile_line(angles,
                                               line[0,::-1] + [0, edge_pos[1]],
                                               line[1,::-1] + [0, edge_pos[1]], mode='reflect')

    diff = np.abs(left_angles - right_angles)

    plt.figure(figsize=(8, 8))
    plt.plot(left_angles, linewidth=1, color='green')
    plt.plot(right_angles, linewidth=1, color='blue')
    plt.plot(diff, linewidth=1, color='red', linestyle='dashed')
    plt.title(f'diff_m={np.mean(diff):<.1f} diff_std={np.std(diff):<.1f} left: m={np.mean(left_angles):<.1f} s={np.std(left_angles)},\n right: m={np.mean(right_angles):<.1f}, s={np.std(right_angles)}')
    plt.savefig('fig.png')

    fig = cv.imread('fig.png')
    cv.imshow('fig', fig)
