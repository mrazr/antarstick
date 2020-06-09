#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
from os import scandir
from pathlib import Path
from queue import Queue
from time import time
from typing import Dict, List, Optional, Tuple
import sys

import cv2 as cv
import numpy as np
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
from PyQt5.QtCore import QThreadPool
from skimage.measure import regionprops
from skimage.util import img_as_ubyte

from my_thread_worker import MyThreadWorker
from stick import Stick

Area = float
Height = float
Ecc = float
Label = int
Centroid = Tuple[int, int]

hog_desc = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9, 1, -1, cv.HOGDescriptor_L2Hys, 0.2, False,
                            cv.HOGDESCRIPTOR_DEFAULT_NLEVELS, True)
HOG_FILE = str(Path(sys.argv[0]).parent / 'camera_processing/stick_hog_svm')

try:
    success = hog_desc.load(HOG_FILE)
except:
    print(f'Could not load file {HOG_FILE}')
    exit(-1)

clahe = cv.createCLAHE()
clahe.setTilesGridSize((16, 20))

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


def measure_snow(img: np.ndarray, sticks: List[Stick]) -> Dict[int, float]:
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

    blurred = cv.GaussianBlur(img, (0, 0), 2.5)
    measurements = {}

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
        diff_norm = 1.0 / diff_norm

        diff = diff_norm * diff

        # Find the indices where the normalized difference is greater than 0.01,
        # this ideally indicates that after seeing snow, we arrived at the stick
        height: np.ndarray = np.argwhere(diff > 0.01)

        # TODO the mapping between the index from np.argwhere and the height isn't totally 1:1 probably, so probably adjust this
        if height.shape[0] == 0:
            measurements[stick.id] = stick.length_px
        else:
            measurements[stick.id] = height[0]

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


dbd_se = np.ones((3, 3), dtype=np.int8)
dbd_se = np.pad(dbd_se, ((0, 0), (1, 1)), 'constant', constant_values=-1)
dbd_se = np.pad(dbd_se, ((0, 0), (2, 2)), 'constant', constant_values=0)

top_end_se = np.pad(dbd_se, ((1, 0), (0, 0)), 'constant', constant_values=-1)
top_end_se = np.pad(top_end_se, ((3, 0), (0, 0)), 'constant', constant_values=0)

bdb_se = np.zeros((3, 3), dtype=np.int8)
bdb_se = np.pad(bdb_se, ((0, 0), (1, 1)), 'constant', constant_values=-1)
bdb_se = np.pad(bdb_se, ((0, 0), (2, 2)), 'constant', constant_values=1)


def uhmt(img: np.ndarray, se: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e_se = (1 * (se == 1)).astype(np.uint8)
    d_se = (1 * (se == 0)).astype(np.uint8)
    e = cv.erode(img, e_se)
    d = cv.dilate(img, d_se)
    mask = e > d
    diff = (e.astype(np.int16) - d.astype(np.int16))
    diff[diff < 0] = 0
    return cv.convertScaleAbs(diff), (255 * mask).astype(np.uint8)


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


def preprocess_phase(img: np.ndarray) -> np.ndarray:
    prep = preprocess_image(img)
    _, mask = uhmt(prep, dbd_se)
    # _, mask2 = uhmt(prep, bdb_se)

    rankd = skimage.filters.rank.percentile(mask, skimage.morphology.rectangle(7, 3), p0=0.8)
    # rankd2 = skimage.filters.rank.percentile(mask2, skimage.morphology.rectangle(7, 3), p0=0.8)
    closed = cv.morphologyEx(rankd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))
    # closed2 = cv.morphologyEx(rankd2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    thin = skimage.morphology.thin(closed)
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


def get_sticks_in_folder(folder: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    queue = Queue(maxsize=10)
    # thread = threading.Thread(target=get_non_snow_images, args=(folder, queue, 9))
    worker = MyThreadWorker(task=get_non_snow_images, args=(folder, queue, 9), kwargs={})

    heat_map = None
    stick_lengths = None
    imgs: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray]] = []
    QThreadPool.globalInstance().start(worker)

    path_img: Tuple[Path, np.ndarray] = queue.get()
    while path_img is not None:
        img = path_img[1]
        gray = cv.cvtColor(cv.pyrDown(img), cv.COLOR_BGR2GRAY)
        gray = cv.pyrUp(cv.pyrDown(gray))[:-50]
        gray = cv.copyMakeBorder(gray, 15, 15, 15, 15, cv.BORDER_CONSTANT, value=0)
        prep, _ = preprocess_phase(gray)
        prep = (255 * prep).astype(np.uint8)
        if heat_map is None:
            heat_map = np.zeros(prep.shape, dtype=np.uint8)
            stick_lengths = np.zeros(prep.shape, dtype=np.uint16)
        heat_map, stick_lengths, img_stats = update_stick_heat_map(prep, heat_map, stick_lengths)
        imgs.append((path_img[0], img, prep, img_stats))
        path_img = queue.get()

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

    return sticks_list[idx], imgs[idx][0]
