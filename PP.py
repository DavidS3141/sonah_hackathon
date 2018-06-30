""" Image Preprocessing """
import cv2 as cv
import numpy as np
import imutils

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

def find_rectangles(frame):
    frame = imutils.resize(frame, width=500)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("1 - Grayscale Conversion", gray)

    gray = cv.bilateralFilter(gray, 11, 17, 17)
    cv.imshow("2 - Bilateral Filter", gray)

    edged = cv.Canny(gray, 170, 200)
    cv.imshow("4 - Canny Edges", edged)

    # rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 4))
    # tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, rectKernel)
    # edges = cv.Canny(frame,100,200)

    # blurred = cv.GaussianBlur(gray, (5, 5), 0.2)
    # thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]

    # se_shape = (16,4)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gray = enhance(gray)
    # gray = cv.GaussianBlur(gray, (5,5), 0)
    # gray = cv.Sobel(gray, -1, 1, 0)
    # h,sobel = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # frame2,contours,_=cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Find contours based on Edges
    (new, cnts, _) = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours=sorted(cnts, key = cv.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    NumberPlateCnt = None #we currently have no Number plate contour


    # frame2, contours, h = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    num_squares = 0
    squares = []
    print("Len(contours) = %i" % len(contours))
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.08*cv.arcLength(cnt,True), True)

        if len(approx)==4:
            # print("square")
            if True: #cv.contourArea(cnt) > 50*50 and cv.contourArea(cnt) < 50*50:
                num_squares += 1
                squares.append(cnt)
                cv.drawContours(frame,[approx],0,(0,0,255),-1)

    cv.imshow('frame',frame)
    cv.imwrite('output_poly/%i.png' % np.random.randint(0,10000), frame)
    print(num_squares)
    # cv.waitKey(0)
    cv.destroyAllWindows()

    print("Num Squares = %i" % num_squares)
    return squares
