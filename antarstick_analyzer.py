#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import rectangle, white_tophat

Area = float
Height = float
Ecc = float
Label = int
Centroid = Tuple[int, int]

def rect_se(width, height):
    return cv.getStructuringElement(cv.MORPH_RECT, (width, height))

def stick_segmentation_preprocess(img):
    thresh_method = cv.ADAPTIVE_THRESH_GAUSSIAN_C

    denoised = denoise(img)

    # Closing off the possible duct-tapes on sticks which would disconnect compenents in segmented image
    closed = cv.morphologyEx(denoised, cv.MORPH_CLOSE, rect_se(19, 19))

    wth = cv.morphologyEx(closed, cv.MORPH_TOPHAT, rect_se(19, 19))

    thresh = cv.adaptiveThreshold(wth, 255.0, thresh_method, cv.THRESH_BINARY, 23, -3)

    # close holes in our thresholded image
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, rect_se(5, 13))

    # This extracts and finally returns line-like structures
    return cv.morphologyEx(closed, cv.MORPH_OPEN, rect_se(1, 13))

def denoise(img):
    down = cv.pyrDown(cv.pyrDown(img))
    return cv.pyrUp(cv.pyrUp(down))
    
def detect_sticks(img: np.ndarray, image_scale: float, merge_lines: bool = True):
    # First preprocess the input image
    preprocessed = stick_segmentation_preprocess(img)


    n_labels, label_img = cv.connectedComponents(preprocessed, connectivity=4, ltype=cv.CV_16U)
    #n_labels, label_img, centroids, stats = cv.connectedComponentsWithStats(preprocessed, connectivity=4, ltype=cv.CV_32S)
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
        return (np.array(lines) * (1.0 / image_scale)).astype(int)


    # Now we're onto merging multiple detected lines per label
    close_lines = {l[0] : [] for l in likely_labels_stats}

    # Assign each line the label whose bounding box contains that line, we get bins of lines that belong to the same bbox
    for line in lines:
        line_mid_point = (int(0.5 * (line[0] + line[2])), int(0.5 * (line[1] + line[3])))
        for l in likely_labels:
            bbox = region_props[l-1].bbox
            if bbox_contains(region_props[l-1].bbox, line_mid_point):
                close_lines[l].append(line)

    merged_lines = []
    scale_factor = 1.0 / image_scale

    # This is the actual "merging", from each line bin select the 2 lines that have the maximum and minumum y coordinate
    # and create a new line combining the endpoints of the two lines
    for lines in close_lines.values():
        if len(lines) == 0:
            continue
        # Retrieve the highest endpoint
        max_y_endpoint = max(lines, key=lambda line: line[3])[2:]
        # Retrieve the lowest endpoint
        min_y_endpoint = min(lines, key=lambda line: line[1])[:2]

        max_y_endpoint[0] = int(max_y_endpoint[0] * scale_factor)
        max_y_endpoint[1] = int(max_y_endpoint[1] * scale_factor)

        min_y_endpoint[0] = int(min_y_endpoint[0] * scale_factor)
        min_y_endpoint[1] = int(min_y_endpoint[1] * scale_factor)

        merged_lines.append(min_y_endpoint + max_y_endpoint)

    return merged_lines


def get_likely_labels(label_img, label_stats) -> List[Tuple[Label, Height, Area, Ecc, Centroid]]:

    # Retain labels that are elongated
    likely_labels = list(filter(lambda l: l.eccentricity > 0.87, label_stats))


    likely_labels = list(sorted(likely_labels, key=height_of_region, reverse=True))

    max_height = height_of_region(likely_labels[0])


    likely_labels = list(filter(lambda l: height_of_region(l) >= 0.4 * max_height, likely_labels))


    return list(map(lambda l: (l.label, height_of_region(l), l.area, l.eccentricity, l.centroid), likely_labels))

def height_of_region(region_prop):
    return region_prop.bbox[2] - region_prop.bbox[0]


def draw_lines_on_img(img, lines):
    for line in lines:
        cv.line(img, tuple(line[:2]), tuple(line[2:]), [255, 255, 0], 2)
    
def bbox_contains(bbox: Tuple[int, int, int, int], point: Tuple[int, int]) -> bool:
    return point[0] >= bbox[1] and point[0] < bbox[3] and point[1] >= bbox[0] and point[1] < bbox[2]