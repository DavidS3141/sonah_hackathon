""" Image Preprocessing """
import cv2 as cv
import numpy as np
import imutils
from Utils import *

def pp_hsv_mask(frame, s=15, v_lower=170):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_white = np.array([0,0,v_lower], dtype=np.uint8)
    upper_white = np.array([255,s,255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_white, upper_white)
    cv.imshow('mask', mask)

    return mask

def enhance(img):
    kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    return cv.filter2D(img, -1, kernel)

def image_to_contours(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("1 - Grayscale Conversion", gray)

    edges = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

    #cv.waitKey(0)
    # Find contours based on Edges
    contour_img, contours, _ = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return gray, contours, edges

def find_rectangles(image):
    wh = np.array(image.shape)[1::-1]
    # Resize the image - change width to 500
    width=500
    image = imutils.resize(image, width=width)

    gray, contours, edges = image_to_contours(image)
    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    squares= []
    for c in contours:
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                squares.append(np.squeeze(np.array(approx)))

    # Drawing the selected contour on the original image
    cv.drawContours(image, squares, -1, (0,255,0), 3)
    concat_image = np.concatenate((image, cv.cvtColor(gray,cv.COLOR_GRAY2RGB), cv.cvtColor(edges,cv.COLOR_GRAY2RGB)), axis=1)

    cv.imwrite('output_poly/%i.png' % np.random.randint(0,10000), concat_image)

    return rescale_squares(squares, wh, width)
