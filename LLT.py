"""
The Low Level Trigger (LLT) should detect all possible plates in scene
"""

import cv2 as cv
import numpy as np
from PP import *

class LLT:
    def __init__(self):
        self.thresh = {}


    def detectPossiblePlates(self, frame):
        height, width, channels = frame.shape

        hsv_mask = pp_hsv_mask(frame)
        masked_frame = cv.bitwise_and(frame, frame, mask=hsv_mask)
        rectangles = find_rectangles(masked_frame)
