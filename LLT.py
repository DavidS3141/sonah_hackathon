"""
The Low Level Trigger (LLT) should detect all possible plates in scene
The lower part of the document compromises the High Level Trigger (HLT) which is used to reduce the numbers of plates detected by the HLT
"""

import cv2 as cv
import numpy as np
from PP import *
from Utils import *


class LLT:
    def __init__(self):
        self.thresh = {}

    def detectPossiblePlates(self, frame, realtime=False):
        height, width, channels = frame.shape

        # hsv_mask = pp_hsv_mask(frame)
        # masked_frame = cv.bitwise_and(frame, frame, mask=hsv_mask)
        rectangles = find_rectangles(frame, realtime)
        # rectangles = [konvex_rectangle(rectangle) for rectangle in rectangles]
        return rectangles


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def deselect_rects(rects, image):
    selected = [rect for rect in rects if isPromisingRect(rect, image)]
    # removing duplicates, overlaps
    region_selected = np.zeros(image.shape[:2])
    rect_areas = [cv.contourArea((rect * image.shape[1::-1]).astype(int)) for rect in selected]
    idx_sorted = np.argsort(rect_areas)[::-1]
    unique_selected = []
    for idx in idx_sorted:
        rect = selected[idx]
        xmin = int(np.min(rect[:, 0]) * image.shape[0])
        xmax = int(np.max(rect[:, 0]) * image.shape[0])
        ymin = int(np.min(rect[:, 1]) * image.shape[1])
        ymax = int(np.max(rect[:, 1]) * image.shape[1])
        cut_region_selected = region_selected[xmin:xmax+1, ymin:ymax+1]
        if np.sum(cut_region_selected) == 0:
            unique_selected.append(rect)
            region_selected[xmin:xmax+1, ymin:ymax+1] = 1
    return unique_selected


def isPromisingRect(rect, image):
    parallel_sides = has_parallel_sides(rect)
    good_ratio = has_good_side_ratios(rect)
    if parallel_sides and good_ratio:
        looks_like_text = rect_looks_like_text(rect, image)
    else:
        looks_like_text = False
    return parallel_sides and good_ratio and looks_like_text


def has_parallel_sides(rect):
    s0 = rect[0, :] - rect[1, :]
    s1 = rect[1, :] - rect[2, :]
    s2 = rect[2, :] - rect[3, :]
    s3 = rect[3, :] - rect[0, :]
    phi0 = np.arctan2(s0[1], s0[0])
    phi1 = np.arctan2(s1[1], s1[0])
    phi2 = np.arctan2(s2[1], s2[0])
    phi3 = np.arctan2(s3[1], s3[0])
    delta1 = (phi0-phi2 + 4 * np.pi) % np.pi
    if delta1 > np.pi/2:
        delta1 = np.pi - delta1
    delta2 = (phi1-phi3 + 4 * np.pi) % np.pi
    if delta2 > np.pi/2:
        delta2 = np.pi - delta2
    assert delta1 >= 0.
    assert delta2 >= 0.
    return (delta1 < np.deg2rad(45)) and (delta2 < np.deg2rad(45))


def has_good_side_ratios(rect):
    m0 = 0.5 * (rect[0, :] + rect[1, :])
    m1 = 0.5 * (rect[1, :] + rect[2, :])
    m2 = 0.5 * (rect[2, :] + rect[3, :])
    m3 = 0.5 * (rect[3, :] + rect[0, :])
    d1 = m0 - m2
    d2 = m1 - m3
    d1 = np.sqrt(np.sum(d1**2))
    d2 = np.sqrt(np.sum(d2**2))
    d = np.sqrt(d1**2 + d2**2)
    ratio = d1/d2
    if ratio < 1.:
        ratio = 1./ratio
    return (ratio > 1.5) and (ratio < 10) and (d > 0.1) and (d < 0.8)


def rect_looks_like_text(rect, image):
    global_grayscale = np.mean(cv.cvtColor(image, cv.COLOR_BGR2GRAY))/255.
    global_graystd = np.std(cv.cvtColor(image, cv.COLOR_BGR2GRAY))/255
    image = image.copy()
    small = np.min(image.shape[:-1])
    su = 2000//small + 1
    xlen = 4*520
    ylen = 4*110
    h, w = image.shape[:-1]
    image = cv.resize(image, (su * w, su * h), interpolation=cv.INTER_CUBIC)

    # # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    # bilateralFilterScale = image.shape[1] / 500
    # image = cv.bilateralFilter(image, int(11*bilateralFilterScale), 17, 17)
    # # cv.imshow("2 - Bilateral Filter", image)

    targetSize = np.float32([[0, 0], [xlen, 0], [xlen, ylen], [0, ylen]])
    scaledCoordinates = np.multiply(rect, np.array([image.shape[1], image.shape[0]])).astype(int)
    scaledCoordinates = np.float32(scaledCoordinates)
    M = cv.getPerspectiveTransform(scaledCoordinates, targetSize)
    target = cv.warpPerspective(image, M, (xlen, ylen))
    # cv.imshow('all', image)
    # cv.imshow('region', target)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    image = target

    # convert to gray scale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    local_grayscale = np.mean(gray)/255.

    # check to see if we should apply thresholding to preprocess the
    # image
    # gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    # gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    gray = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)[1]
    # gray = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #     cv.THRESH_BINARY,11*64+1,2)
    # cv.imshow('bin', gray)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # make a check to see if median blurring should be done to remove
    # noise
    # gray = cv.medianBlur(gray, 3)

    ret, labels = cv.connectedComponents(255-gray)

    listOfBB = []
    delta = 10
    for i in range(ret):
        r1, r2, c1, c2 = bbox(labels == i)
        dr = r2 - r1
        dc = c2 - c1
        if dr > ylen / 3 and dr < ylen * 0.99 and dc > xlen / 40 and dc < xlen * 0.9:
            # possibly one or more chars
            n_chars = int(round(dc/dr))
            ci = np.round(np.linspace(c1, c2, n_chars+1)).astype(int)
            for k in range(n_chars):
                listOfBB.append((r1-delta, r2+delta+1, ci[k]-delta, ci[k+1]+delta+1))

    # cv.imshow('bin%d' % (len(listOfBB) >= 2), gray)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return len(listOfBB) >= 2 and local_grayscale < global_grayscale + global_graystd
