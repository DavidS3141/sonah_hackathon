""" Image Preprocessing """
import cv2 as cv
import numpy as np

def pp_hsv_mask(frame, s=15, v_lower=180):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_white = np.array([0,0,v_lower], dtype=np.uint8)
    upper_white = np.array([255,s,255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_white, upper_white)
    cv.imshow('mask', mask)

    return mask

def find_rectangles(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

    frame2, contours, h = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    num_squares = 0
    squares = []
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, cv.arcLength(cnt,True), True)

        if len(approx)==4:
            # print("square")
            if cv.contourArea(cnt) > 50*50 and cv.contourArea(cnt) < 50*50:
                num_squares += 1
                squares.append(cnt)
                cv.drawContours(thresh,[approx],0,(0,0,255),-1)

    cv.imshow('frame',thresh)
    cv.imwrite('output_poly/%i.png' % np.random.randint(0,10000), thresh)
    print(num_squares)
    # cv.waitKey(0)
    cv.destroyAllWindows()

    print("Num Squares = %i" % num_squares)
    return squares
