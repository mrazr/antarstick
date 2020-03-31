#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
import skimage as sk
from skimage.measure import regionprops
from skimage.morphology import rectangle, white_tophat

from stick import Stick

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

    # Closing off the possible duct-tapes on sticks which would disconnect compenents in segmented image
    closed: np.ndarray = cv.morphologyEx(denoised, cv.MORPH_CLOSE, rect_se(19, 19))

    wth: np.ndarray = cv.morphologyEx(closed, cv.MORPH_TOPHAT, rect_se(19, 19))

    thresh: np.ndarray = cv.adaptiveThreshold(wth, 255.0, thresh_method, cv.THRESH_BINARY, 25, -3)

    # close holes in our thresholded image
    closed: np.ndarray = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rect_se(5, 13))

    # This extracts and finally returns line-like structures
    return cv.morphologyEx(closed, cv.MORPH_OPEN, rect_se(1, 13))

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
    # First preprocess the input image
    preprocessed = stick_segmentation_preprocess(img)


    n_labels, label_img = cv.connectedComponents(preprocessed, connectivity=4, ltype=cv.CV_16U)
    region_props = regionprops(label_img)

    likely_labels_stats: List[Tuple[Label, Height, Area, Ecc, Centroid]] = get_likely_labels(preprocessed, region_props)

    likely_labels = {l[0] for l in likely_labels_stats}


    # Filter out labels which are not interesting
    for r in range(preprocessed.shape[0]):
        for c in range(preprocessed.shape[1]):
            if label_img[r, c] not in likely_labels:
                preprocessed[r, c] = 0

    height = likely_labels_stats[-1][1]

    lines = cv.HoughLinesP(preprocessed, 1, np.pi / 180, 50, height, 15)

    # Transform the lines so they all have their first endpoint is the higher one
    lines = list(map(lambda line: [line[0][2], line[0][3], line[0][0], line[0][1]] if line[0][1] > line[0][3] else list(line[0]), lines))

    if not merge_lines:
        return (np.array(lines) * scale_lines_by).astype(int)


    # Now we're onto merging multiple detected lines per label
    close_lines = {l[0] : [] for l in likely_labels_stats}

    # Assign each line the label whose bounding box contains that line, we get bins of lines that belong to the same bbox
    for line in lines:
        line_mid_point = (int(0.5 * (line[0] + line[2])), int(0.5 * (line[1] + line[3])))
        for l in likely_labels:
            bbox = region_props[l-1].bbox
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
    likely_labels = list(filter(lambda l: l.eccentricity > 0.87, label_stats))


    likely_labels = list(sorted(likely_labels, key=height_of_region, reverse=True))

    max_height = height_of_region(likely_labels[0])


    likely_labels = list(filter(lambda l: height_of_region(l) >= 0.1 * max_height, likely_labels))


    return list(map(lambda l: (l.label, height_of_region(l), l.area, l.eccentricity, l.centroid), likely_labels))

def height_of_region(region_prop) -> int:
    return region_prop.bbox[2] - region_prop.bbox[0]


def draw_lines_on_img(img, lines):
    for line in lines:
        cv.line(img, tuple(line[:2]), tuple(line[2:]), [255, 255, 0], 2)
    
def bbox_contains(bbox: Tuple[int, int, int, int], point: Tuple[int, int]) -> bool:
    return point[0] >= bbox[1] and point[0] < bbox[3] and point[1] >= bbox[0] and point[1] < bbox[2]

def show_imgs_(images: List[np.ndarray], names: List[str]):
    for image, name in zip(images, names):
        cv.imshow(name, image)
    cv.waitKey(0)

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
        line_profile = sk.measure.profile_line(img, end2, end1, mode='reflect')
        
        off_x = np.array([0, int(1.5 * thickness)])
        # Now extract intensity profiles left of and right of the stick
        left_neigh_profile = sk.measure.profile_line(img, end2 - off_x, end1 - off_x, mode='reflect')
        right_neigh_profile = sk.measure.profile_line(img, end2 + off_x, end1 + off_x, mode='reflect')

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
