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

# not using the skimage rectange method for rectangle SE because it interprets
# width as height and vice versa. It is a known issue that is still not resolved (https://github.com/scikit-image/scikit-image/issues/4125)
def rect_se(width, height):
    #return np.ones((height, width), dtype=np.uint8)
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
    
def detect_sticks(img: np.ndarray, image_scale: float):
    # First preprocess the input image
    preprocessed = stick_segmentation_preprocess(img)


    n_labels, label_img = cv.connectedComponents(preprocessed, connectivity=4, ltype=cv.CV_16S)
    #n_labels, label_img, centroids, stats = cv.connectedComponentsWithStats(preprocessed, connectivity=4, ltype=cv.CV_32S)
    region_props = regionprops(label_img)

    likely_labels_stats: List[Tuple[Label, Height, Area, Ecc, Centroid]] = get_likely_labels(preprocessed, region_props)

    likely_labels = {l[0] for l in likely_labels_stats}

    with np.nditer(label_img, op_flags=['readwrite']) as it:
        for label in it:
            if likely_labels[label]:
                label = 255
            else:
                label = 0

    height = likely_labels_stats[-1][1]

    lines = cv.HoughLinesP(label_img, 1, np.pi / 180, 50, height, 15)

    lines = list(map(lambda line: (line[2], line[3], line[0], line[1]) if line[1] > line[3] else line, lines))

    lines.sort(key=lambda line: line[0])

    merged_lines = np.array([[-1, preprocessed.shape[1] * 2, -1, -1]] * len(likely_labels_stats))

    for line in lines:
        dist = 2 * preprocessed.shape[1]
        line_idx = 0
        line_centroid = (0.5 * (line[0] + line[2]), 0.0)
        for idx, l in enumerate(likely_labels_stats):
            centroid_x = l.centroid[1]

            if line_centroid[0] - centroid_x < dist:
                dist = line_centroid[0] - centroid_x 
                line_idx = idx
        
        merged_line = merged_lines[line_idx]
        if line[1] < merged_line[1]:
            merged_line[:2] = line[:2]
        if line[3] > merged_line[3]:
            merged_line[2:] = line[2:]

    return merged_lines * scale_factor

def get_likely_labels(label_img, label_stats) -> List[Tuple[Label, Height, Area, Ecc, Centroid]]:
    label_stats = regionprops(label_img)

    # Retain labels that are elongated
    likely_labels = filter(lambda l: l.eccentricity > 0.87)

    likely_labels = list(sorted(likely_labels, key=lambda l: l.height))

    max_height = likely_labels[0].height

    likely_labels = filter(lambda l: l.height >= 0.4 * max_height, likely_labels)

    return list(map(lambda l: (l.label, l.height, l.area, l.eccentricity, l.centroid), likely_labels))
