""" Image Preprocessing """
import cv2 as cv
import numpy as np
import imutils
from Utils import *
import itertools


def pp_hsv_mask(frame, s=100, v_lower=50):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_white = np.array([0,0,v_lower], dtype=np.uint8)
    upper_white = np.array([255,s,255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_white, upper_white)

    return mask

def image_to_contours(image, thresh_size=11):
    # image = cv.bilateralFilter(image,11,30,30)
    hsv_mask = pp_hsv_mask(image)
    # cv.imshow("hsv_mask", hsv_mask)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gray = cv.medianBlur(gray, 5)
    # gray = cv.blur(gray, (5,5))
    # gray = cv.bilateralFilter(gray,11,17,17)

    # cv.imshow("1 - Grayscale Conversion", gray)

    edges = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,thresh_size,2)
    # cv.imshow("edges pre", edges)
    edges = (hsv_mask > 0) * edges
    # cv.imshow("edges post", edges)
    #cv.waitKey(0)
    # Find contours based on Edges
    contour_img, contours_inner, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour_img, contours_outer, _ = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # cv.waitKey(0)
    contours = contours_inner + contours_outer
    return gray, contours, edges

def enhance(img):
    kernel = np.array([[-1,0,1],[-2,0,2],[1,0,1]])
    return cv.filter2D(img, -1, kernel)

def image_to_contours_2(image):
    se_shape = (16,4)

    if options.get('type') == 'rect':
        se_shape = (17,4)

    elif options.get('type') == 'square':
        se_shape = (7,6)

    input_image = np.copy(image)

    gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    gray = enhance(gray)
    gray = cv.GaussianBlur(gray, (5,5), 0)
    gray = cv.Sobel(gray, -1, 1, 0)
    h,sobel = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    se = cv.getStructuringElement(cv.MORPH_RECT, se_shape)
    gray = cv.morphologyEx(sobel, cv.MORPH_CLOSE, se)
    ed_img = np.copy(gray)
    return findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

def find_rectangles(image):
    wh = np.array(image.shape)[1::-1]
    # Resize the image - change width to 500
    width=500
    image = imutils.resize(image, width=width)
    thresh_size_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    epsilon_list = [0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.05]
    squares= []
    all_contours = []
    for thresh_size, epsilon in itertools.product(thresh_size_list, epsilon_list):
        gray, contours, edges = image_to_contours(image, thresh_size)
        # loop over our contours to find the best possible approximate contour of number plate
        for c in contours:
            all_contours.append(c)
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, epsilon * peri, True)
            if len(approx) == 4:
                #contours must have at least certain size (250 px)
                if cv.contourArea(approx) > 250:
                    squares.append(np.squeeze(np.array(approx)))
        print("thresh_size: %i, epsilon_size: %.3f, num_squares = %i" % (thresh_size, epsilon, len(squares)))

    # Drawing the selected contour on the original image
    cv.drawContours(image, all_contours, -1, (0,0,255), 3)
    cv.drawContours(image, squares, -1, (0,255,0), 3)
    concat_image = np.concatenate((image, cv.cvtColor(gray,cv.COLOR_GRAY2RGB), cv.cvtColor(edges,cv.COLOR_GRAY2RGB)), axis=1)

    cv.imwrite('output_poly/%i.png' % np.random.randint(0,10000), concat_image)

    return rescale_squares(squares, wh, width)
