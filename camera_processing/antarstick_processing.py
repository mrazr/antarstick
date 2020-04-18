#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
from os import scandir
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
import skimage.exposure
import skimage.filters
import skimage.feature
import skimage.measure
import skimage.morphology
import skimage.transform
from skimage.measure import regionprops
from skimage.util import img_as_ubyte

from stick import Stick

import seaborn as sb
import matplotlib.pyplot as plt

Area = float
Height = float
Ecc = float
Label = int
Centroid = Tuple[int, int]


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
    #return cv.morphologyEx(open, cv.MORPH_CLOSE, rect_se(3, 15))


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
    #down = cv.pyrDown(down)
    #down = cv.pyrUp(down)
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

    #for r in range(preprocessed.shape[0]):
    #    for c in range(preprocessed.shape[1]):
    #        if label_img[r, c] not in likely_labels:
    #            preprocessed[r, c] = 0
    height = likely_labels_stats[5][1]

    lines = cv.HoughLinesP(preprocessed, 1, np.pi / 180, int(0.8 * height), height, 20)

    # Transform the lines so they all have their first endpoint is the higher one
    lines = list(map(lambda line: [line[0][2], line[0][3], line[0][0], line[0][1]] if line[0][1] > line[0][3] else list(line[0]), lines))

    if not merge_lines:
        return (np.array(lines) * scale_lines_by).astype(int)

    # Now we're onto merging multiple detected lines per label
    close_lines: Dict[int, List[List[int]]] = {l[0]: [] for l in likely_labels_stats}

    # Assign each line the label whose bounding box contains that line, we get bins of lines that belong to the same bbox
    for line in lines:
        line_mid_point = (int(0.5 * (line[0] + line[2])), int(0.5 * (line[1] + line[3])))
        for l in likely_labels:
            if bbox_contains(region_props[l-1].bbox, line_mid_point):
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
    likely_labels = list(sorted(filter(lambda l: l.eccentricity > 0.87, label_stats), key=height_of_region, reverse=True))

    max_height = height_of_region(likely_labels[0])

    likely_labels = list(filter(lambda l: height_of_region(l) >= 0.2 * max_height, likely_labels))

    return list(map(lambda l: (l.label, height_of_region(l), l.area, l.eccentricity, l.centroid), likely_labels))


def height_of_region(region_prop) -> int:
    return region_prop.bbox[2] - region_prop.bbox[0]


def draw_lines_on_img(img, lines):
    for line in lines:
        cv.line(img, tuple(line[:2]), tuple(line[2:]), [255, 255, 0], 2)


def bbox_contains(bbox: Tuple[int, int, int, int], point: Tuple[int, int]) -> bool:
    return point[0] >= bbox[1] and point[0] < bbox[3] and point[1] >= bbox[0] and point[1] < bbox[2]


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

    return np.mean(hsv_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 2]) < 100


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


hmt_se = np.ones((3, 3), dtype=np.int8)
hmt_se = np.pad(hmt_se, ((0, 0), (1, 1)), 'constant', constant_values=-1)
hmt_se = np.pad(hmt_se, ((0, 0), (2, 2)), 'constant', constant_values=0)

hmt_se2= np.zeros((5, 5), dtype=np.int8)
hmt_se2 = np.pad(hmt_se2, ((1, 1), (0, 0)), 'constant', constant_values=-1)
hmt_se2 = np.pad(hmt_se2, ((3, 3), (0, 0)), 'constant', constant_values=1)

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
    return img_as_ubyte(skimage.exposure.equalize_adapthist(
        img, [int(img.shape[0] / 20.0), int(img.shape[1] / 20.0)]))


def detect_sticks_hmt(img: np.ndarray, height_perc: float) -> List[List[int]]:
    prep = preprocess_image(img)
    hmt, mask = uhmt(prep, hmt_se)
    height = int(height_perc * img.shape[0])
    rankd = skimage.filters.rank.percentile(mask, skimage.morphology.rectangle(7, 3), p0=0.8)
    closed = cv.morphologyEx(rankd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 15)))

    thin = img_as_ubyte(skimage.morphology.thin(closed))
    #show_imgs_([mask, rankd, closed, thin], ["mask", "randk", "closed", "thin"])
    #cv.destroyAllWindows()
    #lines = skimage.transform.probabilistic_hough_line(closed,
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
    #return merge_lines(lines)

def get_non_snow_images(path: Path, count: int = 1) -> Optional[List[np.ndarray]]:
    image_list: List[np.ndarray] = []

    for file in scandir(path):
        if file.name[-3:].lower() != "jpg": # TODO handle JPEG
            continue
        img = cv.imread(str(file.path))
        img = cv.pyrDown(img)
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
        #if np.abs(line_r_t[1] - current_line_r_t[1]) > 20:
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

    #print(lines)

    #plot = sb.scatterplot(x=rhos, y=thetas).figure
    #plot.savefig("fig.png")
    #fig = cv.imread("fig.png")

    #cv.imshow("fig", fig)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    #plt.clf()

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
    _, mask = uhmt(prep, hmt_se)

    rankd = skimage.filters.rank.percentile(mask, skimage.morphology.rectangle(7, 3), p0=0.8)
    closed = cv.morphologyEx(rankd, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 13)))

    thin = skimage.morphology.thin(closed)

    area_opened = skimage.morphology.remove_small_objects(thin, min_size=8, connectivity=2)

    return area_opened

def detect_sticks_from_preprocessed(img: np.ndarray, height_percentage: float) -> List[Tuple[Tuple[int]]]:
    height = int(height_percentage * img.shape[0])

    lines = skimage.transform.probabilistic_hough_line(img, threshold=int(0.4 * height),
                                                       line_length=int(height * 0.9),
                                                       line_gap=15)

    return lines

def get_lines_from_preprocessed(img: np.ndarray) -> List[List[np.ndarray]]:
    labels, num_labels = skimage.measure.label(img, connectivity=2, return_num=True)
    reg_props = skimage.measure.regionprops(labels)

    lines = []
    for reg_prop in reg_props:
        coords = reg_prop.coords
        min_y_coord = min(coords, key=lambda c: c[0])
        max_y_coord = max(coords, key=lambda c: c[0])

        lines.append([np.array([min_y_coord[1], min_y_coord[0]]), np.array([max_y_coord[1], max_y_coord[0]])])
    
    return lines
