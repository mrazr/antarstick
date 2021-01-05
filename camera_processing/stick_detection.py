#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:20:07 2020

@author: radoslav
"""

import math
from typing import List, Optional, Tuple

import cv2 as cv
import numpy as np
import skimage.exposure
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation
import skimage.transform

# structuring elements to match horizontal slices of sticks in hit-or-miss transform
# dbd variants, or dark-bright-dark, will match sticks that are brighter than their surroundings,
# and analogically for bdb variants
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

hmt_selems = [
    [
        dbd_se1,
        dbd_se3,
        dbd_se5,
        dbd_se7,
    ],
    [
        bdb_se1,
        bdb_se3,
        bdb_se5,
        bdb_se7,
    ]
]


def uhmt(img: np.ndarray, se: np.ndarray, anchor: Tuple[int, int] = (-1, -1)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs the unconstrained grayscale Hit-or-Miss transform with the given structuring element `se`

    Parameters
    ----------
    img : (np.uint8) np.ndarray
        input grayscale image
    se : np.ndarray
        the structuring element, pixels valued by ones will match foreground, pixels with zeros will match
        background, and pixels with values -1 will match both.
    anchor : Tuple[int, int]
        origin of `se`, default is (-1, -1) which is interpreted as the center of the structuring element.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        tuple containing the result of the uhmt, and a binary image marking the pixels where the erosion had greater
        value than the dilation at particular pixel.
    """

    e_se = (1 * (se == 1)).astype(np.uint8)
    d_se = (1 * (se == 0)).astype(np.uint8)
    e: np.ndarray = cv.erode(img, e_se, borderType=cv.BORDER_REPLICATE, anchor=anchor)
    d: np.ndarray = cv.dilate(img, d_se, borderType=cv.BORDER_REPLICATE, anchor=anchor)
    mask: np.ndarray = e > d
    diff = (e.astype(np.int16) - d.astype(np.int16))
    diff[diff < 0] = 0
    return cv.convertScaleAbs(diff), (1 * mask).astype(np.uint8)


def asf(img: np.ndarray, size: int, w: int, mode: str = 'co') -> np.ndarray:
    """
    Filters the input image with the alternating sequential filter using a vertical line structuring element.

    Parameters
    ----------
    img: np.ndarray
        a grayscale image to be filtered
    size: int
        maximum length of the line structuring element
    w: int
        width of the line structuring element
    mode: str
        type of alternating sequential filter to be used, default 'co'

    Returns
    -------
    np.ndarray
        the filtered image
    """

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
    """
    Generates an axis-aligned bounding box for `line`.

    Parameters
    ----------
    line: np.ndarray
        a (2, 2) shaped ndarray representing a line
    max_width: int
        should correspond to the width of the image the line is in
    max_height: int
        similar to `max_width`
    width: int
        relates to the width of the bounding box, though when a line is not aligned with the vertical axis, the generated
        box might be a little wider, default 25
    out_of_bounds: bool
        whether the vertices of the generated box can go beyond the image's boundary, default False
    left_range: int
        sticks that are close to each other will have limited left and right range of their boxes so they do not overlap,
        this is specified by this parameter, default -1, meaning the line is not close to other line
    right_range: int
        similar to `left_range`, default -1

    Returns
    -------
    np.ndarray
        a (2, 2) shaped ndarray containing the top-left and bottom-right vertices of the bounding box
    """

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


def angle_of_line(line: np.ndarray) -> float:
    v = line[1] - line[0]
    return (math.degrees(math.atan2(v[1], v[0]))) % 180


def line_vector(line: np.ndarray) -> np.ndarray:
    v = line[1] - line[0]
    return v / (np.linalg.norm(v) + 0.00001)


def line_edge_offsets(line: np.ndarray, mag: np.ndarray, w: int = 19) -> List[int]:
    """
    Calculates the offsets of the left and right edges of a stick from the line representing it.

    Parameters
    ----------
    line: np.ndarray
    mag: np.darray
        gradient magnitude of the image where the line is located
    w: int
        maximum stick width to consider, default 19

    Returns
    -------
    List[int]
        two-element list containing the offsets of the left and right edge, respectively
    """
    h_w = (w - 1) // 2
    # obtain an image by sampling `mag` with `w` lines with the orientation defined by `line` and centered around it
    # the image is `w` pixels wide, and it is aligned with `line`
    mag_lines = skimage.measure.profile_line(mag, line[0, ::-1], line[1, ::-1], linewidth=w, reduce_func=None,
                                             mode='constant')
    # compute the sum of the gradient magnitude along the columns of `mag_line`
    line_sums = np.sum(mag_lines, axis=0)
    # now, the probable left edge is identified by the column left having maximum magnitude sum, left of the central
    # columns
    left_edge_idx = np.argmax(line_sums[::-1][h_w + 1:]) + 1
    # similarly for the right edge
    right_edge_idx = np.argmax(line_sums[h_w + 1:]) + 1

    return [left_edge_idx, right_edge_idx]


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


def detect_lines(gray: np.ndarray, h: int = 9) -> List[np.ndarray]:
    """
    Performs line detection in the image `gray`

    Parameters
    ----------
    gray: np.ndarray
        a grayscale image where lines are to be detected
    h: int
        a parameter for filtering, corresponds to the length of vertical line structuring element, default 9

    Returns
    -------
    List[np.ndarray]
        list of lines, each line is a (2, 2) shaped ndarray, first row contains the top endpoint, the second row
        contains the bottom endpoint
    """
    gray_eq = gray
    line_count = 140
    # to avoid reporting too many lines, the procedure ends only when less than or equal to 100 lines are found
    # until <= 100 lines are found, in each while-iteration the intensity of the input image is scaled by the factor of
    # 0.5 to bring down the contrast, hopefully reducing the number of detected lines
    while line_count > 100:
        hmt_dbd1, _ = uhmt(gray_eq, hmt_selems[0][0])
        hmt_dbd3, _ = uhmt(gray_eq, hmt_selems[0][1])
        hmt_dbd5, _ = uhmt(gray_eq, hmt_selems[0][2])
        hmt_dbd7, _ = uhmt(gray_eq, hmt_selems[0][3])
        hmt_bdb1, _ = uhmt(gray_eq, hmt_selems[1][0])
        hmt_bdb3, _ = uhmt(gray_eq, hmt_selems[1][1])
        hmt_bdb5, _ = uhmt(gray_eq, hmt_selems[1][2])
        hmt_bdb7, _ = uhmt(gray_eq, hmt_selems[1][3])

        threshold = 9.0

        _, hmt_dbd1_ = cv.threshold(hmt_dbd1, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_dbd3_ = cv.threshold(hmt_dbd3, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_dbd5_ = cv.threshold(hmt_dbd5, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_dbd7_ = cv.threshold(hmt_dbd7, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_bdb1_ = cv.threshold(hmt_bdb1, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_bdb3_ = cv.threshold(hmt_bdb3, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_bdb5_ = cv.threshold(hmt_bdb5, threshold, 1.0, cv.THRESH_BINARY)
        _, hmt_bdb7_ = cv.threshold(hmt_bdb7, threshold, 1.0, cv.THRESH_BINARY)

        th1 = asf(hmt_dbd1_, h, 1, 'co')
        th2 = asf(hmt_dbd3_, h, 1, 'co')
        th3 = asf(hmt_bdb1_, h, 1, 'co')
        th4 = asf(hmt_bdb3_, h, 1, 'co')
        th5 = asf(hmt_bdb5_, h, 1, 'co')
        th6 = asf(hmt_dbd5_, h, 1, 'co')
        th7 = asf(hmt_dbd7_, h, 1, 'co')
        th8 = asf(hmt_bdb7_, h, 1, 'co')

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


#def apply_multi_hmt(img: np.ndarray, height: int = 1, width: int = -1) -> Tuple[np.ndarray, np.ndarray]:
#    dbd_hmts = []
#    bdb_hmts = []
#
#    for selem in hmt_selems[0][1:]:
#        s = selem
#        if height > 1:
#            s = np.tile(selem, (height, 1))
#        dbd_hmts.append(uhmt(img, s, anchor=(selem.shape[1] // 2, 0))[0])
#
#    for selem in hmt_selems[1][1:]:
#        s = selem
#        if height > 1:
#            s = np.tile(selem, (height, 1))
#        bdb_hmts.append(uhmt(img, s, anchor=(selem.shape[1] // 2, 0))[0])
#
#    return reduce(np.bitwise_or, dbd_hmts), reduce(np.bitwise_or, bdb_hmts)


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


def find_sticks(gray: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """
    Finds sticks in the image `gray`

    Parameters
    ----------
    gray: np.ndarray
        a grayscale image where sticks are to be detected

    Returns
    -------
    List[Tuple[np.ndarray]]
        list of tuples containing a line in the form of (2, 2) shaped ndarray and its pixel width
    """
    gray = gray[:-100, :] # crop the image from the below to get rid of the information strip in the image
    dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
    mag = cv.magnitude(dx, dy)

    lines1 = detect_lines(gray)
    lines1 = list(lines1)
    # multiple lines are reported along each stick, to obtain one line delineating the stick, merge close lines
    lines1 = merge_lines_with_same_orientation(lines1)
    lines1 = np.array(lines1)

    lines05 = detect_lines(cv.pyrDown(gray))
    lines05 = 2 * lines05
    lines05 = list(lines05)
    lines05 = merge_lines_with_same_orientation(list(lines05))
    lines05 = np.array(lines05)

    if len(lines1) > 0 and len(lines05) > 0:
        lines = np.array(merge_lines_with_same_orientation(list(np.vstack((lines1, lines05)))))
    else:
        lines = lines1 if len(lines1) > 0 else lines05
    final_lines = []

    for line in lines:
        line_vec = (line[1] - line[0]).astype(np.float32)
        if np.linalg.norm(line_vec) < 2:
            continue
        line_vec /= np.linalg.norm(line_vec)
        a = 20
        w = 25
        line_ = extend_line(line, amount=a, endpoints='both')
        line_[1, 1] = np.minimum(line_[1, 1], gray.shape[0] - 1)

        edge_offsets = line_edge_offsets(line_, mag, w)
        # the stick's width is approximated by adding the left and right offsets from the line, that is assumed to
        # be aligned with its major axis
        final_lines.append((line, int(edge_offsets[0] + edge_offsets[1])))

    return final_lines


#def extract_feature_vector(gray: np.ndarray, mag: np.ndarray, angle: np.ndarray, line_angle: float, left_edge: int,
#                           right_edge: int) -> np.ndarray:
#    off = 2
#    left_angles = angle[:, max(0, left_edge - off):left_edge + off + 1]
#    left_angles = np.where(left_angles > 180, left_angles - 180, left_angles)
#
#    right_angles = angle[:, right_edge - off:min(angle.shape[1], right_edge + off + 1)]
#    right_angles = np.where(right_angles > 180, right_angles - 180, right_angles)
#
#    right_angle_diff = np.abs(right_angles - line_angle)
#    right_angle_diff_count = np.count_nonzero(right_angle_diff < 11) / 3.0
#    left_angle_diff = np.abs(left_angles - line_angle)
#    left_angle_diff_count = np.count_nonzero(left_angle_diff < 11) / 3.0
#
#    diff = np.abs(left_angles[:, 1] - right_angles[:, 1])
#    intensity = gray[:, (gray.shape[1]) // 2]
#
#    return np.array([
#        np.mean(diff),
#        np.std(diff),
#        np.std(left_angles),
#        np.std(right_angles),
#        np.abs(np.mean(left_angles) - np.mean(right_angles)),
#        np.std(intensity),
#        np.mean(left_angle_diff),
#        left_angle_diff_count / float(left_angles.shape[0]),
#        right_angle_diff_count / float(right_angles.shape[0]),
#    ])


def length_of_line(line: np.ndarray) -> float:
    return np.linalg.norm(line[0] - line[1])


def merge_lines_with_same_orientation(lines: List[np.ndarray], max_gap: int = 25) -> List[np.ndarray]:
    """
    Merges lines that have the same orientation and are close to each other

    Parameters
    ----------
    lines: List[np.ndarray]
        list of lines to be merged
    max_gap: int
        maximum gap between two lines to merged together, default 25
    """
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


def fit_into_length(line: np.ndarray, length: float) -> np.ndarray:
    """
    Modifies `line`, preserving its top endpoint, so that it is `length` pixels long
    """
    vec = line_vector(line)
    bottom = line[0] + length * vec
    bottom = np.round(bottom).astype(np.int32)

    line_to_return = np.array([line[0], bottom])
    return np.array([line[0], bottom])


def fit_into_length_from_bottom(line: np.ndarray, length: float) -> np.ndarray:
    """
    Modifies `line`, preserving its bottom endpoint, so that it is `length` pixels long
    """
    vec = -1.0 * line_vector(line)
    top = line[1] + length * vec
    top = np.round(top).astype(np.int32)

    line_to_return = np.array([top, line[1]])
    return line_to_return


#def deb_draw_lines(img: np.ndarray, lines: Union[List[np.ndarray], np.ndarray]):
#    dr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#    for l in lines:
#        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
#        cv.line(dr, (int(l[0, 0]), int(l[0, 1])), (int(l[1, 0]), int(l[1, 1])), color)
#    cv.imshow('dr', dr)
#    cv.waitKey(0)
#    cv.destroyWindow('dr')


def boxes_intersection(box1: np.ndarray, box2: np.ndarray) -> Optional[np.ndarray]:
    x1 = max(box1[0, 0], box2[0, 0])
    y1 = max(box1[0, 1], box2[0, 1])
    x2 = min(box1[1, 0], box2[1, 0])
    y2 = min(box1[1, 1], box2[1, 1])

    if x1 >= x2 or y1 >= y2:
        return None
    return np.array([[x1, y1], [x2, y2]], np.int32)


#def box_area(box: np.ndarray) -> int:
#    return (box[1, 0] - box[0, 0]) * (box[1, 1] - box[0, 1])


def line_to_line_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    line_vec = line_vector(line1)
    v = line2[0] - line1[0]
    dot = np.dot(line_vec, v)
    alpha = math.acos(dot / (np.linalg.norm(v) + 0.0001))
    return math.sin(alpha) * np.linalg.norm(v)

