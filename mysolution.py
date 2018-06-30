#!/usr/bin/env python3

"""This is an example module showing how the API should be used."""
from api.hackathon import HackathonApi, RunModes
from LLT import LLT, deselect_rects
import os
import time
import hashlib
import json
import random
import numpy as np
import cv2 as cv
from PIL import Image
import pytesseract
import re


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


class MySolution(HackathonApi):
    """
    This class implements your solution to the two tasks.

    When running the agorithms via the HackathonAPI's run method,
    the methods below are automatically called when needed.
    However you can also execute them yourself, by directly invoking them with
    the necessary parameters.

    The HackathonApi class does not implement a __init__ constructor,
    therefore you can implement your own here and don't have to care about
    matching any super-class method signature.
    Instead take care to call the initializeApi method of this class' object,
    if you want to use the Hackathon API. (See below for an example)

    You can also create other files and classes and do whatever you like, as
    long as the methods below return the required values.
    """
    xlen = 4*520
    ylen = 4*110
    minSmall = 2000

    def isRectConvex(self, rect):
        return cv.isContourConvex(np.multiply(rect, 10000).astype(int))

    def make_rect_convex(self, rect):
        xpy = rect[:, 0] + rect[:, 1]
        startIdx = np.argsort(xpy)[0]
        other3 = rect[np.argsort(xpy)[1:], :]
        start = rect[startIdx, :].reshape(1, 2)
        relOther3 = other3 - start
        phi = np.arctan2(relOther3[:, 1], relOther3[:, 0])
        other3sort = other3[np.argsort(phi), :]
        convex_rect = np.concatenate([start, other3sort], axis=0)
        if self.isRectConvex(convex_rect):
            return convex_rect
        else:
            return None

    def handleFrameForTaskA(self, frame):
        """
        See the documentation in the parent class for a whole lot of information on this method.

        We will just stupidly return random ROIs here
        to show what the result has to look like.
        """
        LowLevelTrigger = LLT()
        possiblePlates = LowLevelTrigger.detectPossiblePlates(frame)
        wh = np.array(frame.shape)[1::-1]

        possiblePlates = [plate/wh for plate in possiblePlates]
        possiblePlates = [plate for plate in possiblePlates if len(np.unique(plate, axis=0)) == 4]
        possiblePlates = [self.make_rect_convex(plate) for plate in possiblePlates]
        plates = [plate for plate in possiblePlates if plate is not None]
        if len(plates) == 0:
            plates = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
        return deselect_rects(plates)

    def wordImage2ListOfLetterImages(self, image):
        # convert to gray scale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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
            if dr > self.ylen / 3 and dc > self.xlen / 40 and dc < self.xlen / 5:
                # possibly one or more chars
                n_chars = int(round(dc/dr))
                ci = np.round(np.linspace(c1, c2, n_chars+1)).astype(int)
                for k in range(n_chars):
                    listOfBB.append((r1-delta, r2+delta+1, ci[k]-delta, ci[k+1]+delta+1))
        if len(listOfBB) < 3:
            return []
        listOfBB = np.array(listOfBB)
        pos = 0.5 * (listOfBB[:, 2] + listOfBB[:, 3])
        listOfBB = listOfBB[np.argsort(pos), :]
        pos = np.sort(pos)
        largeDiffIdx = np.sort(np.argsort(pos[1:] - pos[:-1])[-2:])
        groups = [listOfBB[0:largeDiffIdx[0]+1, :],
                  listOfBB[largeDiffIdx[0]+1:largeDiffIdx[1]+1, :],
                  listOfBB[largeDiffIdx[1]+1:, :]]
        listOfLetters = []
        for group in groups:
            group = group.reshape(-1, 4)
            groupOfLetters = []
            for k in range(group.shape[0]):
                r1, r2, c1, c2 = tuple(group[k, :])
                if not (r1>0 and r2>0 and c1>0 and c2>0):
                    return []
                if not (r1<=image.shape[0] and r2<=image.shape[0]):
                    return []
                if not (c1<=image.shape[1] and c2<=image.shape[1]):
                    return []
                groupOfLetters.append(image[r1:r2, c1:c2, :])
            listOfLetters.append(groupOfLetters)

        return listOfLetters

    def letterImageToChar(self, letter_image, type):
        # convert to gray scale
        gray = cv.cvtColor(letter_image, cv.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        # gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
        gray = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)[1]

        # make a check to see if median blurring should be done to remove
        # noise
        # gray = cv.medianBlur(gray, 3)
        # compute text
        if type == 'letter':
            text = pytesseract.image_to_string(Image.fromarray(gray), config='--tessdata-dir ../tessdata --psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        elif type == 'digit':
            text = pytesseract.image_to_string(Image.fromarray(gray), config='--tessdata-dir ../tessdata --psm 10 -c tessedit_char_whitelist=0123456789')
        elif type == 'posDigit':
            text = pytesseract.image_to_string(Image.fromarray(gray), config='--tessdata-dir ../tessdata --psm 10 -c tessedit_char_whitelist=123456789')
        elif type == 'digitE':
            text = pytesseract.image_to_string(Image.fromarray(gray), config='--tessdata-dir ../tessdata --psm 10 -c tessedit_char_whitelist=0123456789E')

        # show image
        # cv.imshow(text, gray)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # print(text)
        return text

    def handleFrameForTaskB(self, frame, regionCoordinates):
        """
        See the documentation in the parent class for a whole lot of information on this method.

        We will just stupidly return None here,
        which basically stands for "I can't read this".
        """
        small = np.min(frame.shape[:-1])
        su = self.minSmall//small + 1
        h, w = frame.shape[:-1]
        frame = cv.resize(frame, (su * w, su * h), interpolation=cv.INTER_CUBIC)

        # # Noise removal with iterative bilateral filter(removes noise while preserving edges)
        # bilateralFilterScale = frame.shape[1] / 500
        # frame = cv.bilateralFilter(frame, int(11*bilateralFilterScale), 17, 17)
        # # cv.imshow("2 - Bilateral Filter", frame)

        targetSize = np.float32([[0, 0], [self.xlen, 0], [self.xlen, self.ylen], [0, self.ylen]])
        scaledCoordinates = np.multiply(regionCoordinates, np.array([frame.shape[1], frame.shape[0]])).astype(int)
        scaledCoordinates = np.float32(scaledCoordinates)
        M = cv.getPerspectiveTransform(scaledCoordinates, targetSize)
        target = cv.warpPerspective(frame, M, (self.xlen, self.ylen))
        # cv.imshow('all', frame)
        # cv.imshow('region', target)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        letterGroups = self.wordImage2ListOfLetterImages(target)
        if len(letterGroups) != 3:
            print("LetterGroups != 3")
            return None
        result = ''
        for k, letters in enumerate(letterGroups):
            for i, letter in enumerate(letters):
                #print(len(letter))
                type = "digit" if k == 2 else "letter"
                if k == 2:
                    if i + 1 == len(letters):
                        type = "digitE"
                    elif i == 0:
                        type = "posDigit"
                result += self.letterImageToChar(letter, type)

                # cv.imshow(self.letterImageToChar(letter)+str(i), letter)
            result += '-'
            # cv.waitKey(0)
            # cv.destroyAllWindows()

        resultParts = result[:-1].split('-')
        if len(resultParts[2]) > 4:
            resultParts[1] += resultParts[2][0]
            resultParts[1] = resultParts[1][:-1] + 'X' # TODO hacky
            resultParts[2] = resultParts[2][1:]
        if len(resultParts[1]) >= 3:
            resultParts[0] += resultParts[1][0]
            resultParts[1] = resultParts[1][1:]
        result = '-'.join(resultParts)
        print(result)
        if re.match("^[a-zA-ZüÖöÜäÄ]{1,3}-[a-zA-ZöÖüÜäÄ]{1,2}-[1-9][0-9]{0,3}[eE]?$", result):
            print("Found possible numberplate")
            return result
        else:
            print("Did not find numberplate")
            return None

        # rng = random.Random()
        # m = hashlib.md5()
        # m.update(json.dumps(regionCoordinates.tolist()).encode())
        # rng.seed(m.hexdigest())
        # resultSet = [
        #     None,
        #     "AC-FT-774",
        #     "MU-YG-728",
        #     "HB-KZ-3124"
        # ]
        # return resultSet[int(((time.time() / 10.0) + rng.randrange(0, len(resultSet))) % len(resultSet))]


if __name__ == "__main__":
    """This is an example of how to use the Hackathon API."""
    # We instantiate an object of our implemented solution first.
    solution = MySolution()
    # Before running the code, the hackathon API has to be initialized.
    # This loads the metadata, needed for running things automatically.
    # Make sure you downloaded all the frames with the download_labeldata.sh script.
    datasetWrapper = solution.initializeApi(os.path.abspath("./metadata.json"), os.path.abspath("./data/"))
    print("MySolution begins here...")
    print("The total number of frames is {:d}".format(datasetWrapper.getTotalFrameCount()))
    # We can test our implementation in multiple modes, by supplying a RunMode
    # to the run method. This will automatically print a few stats after running.
    # You may however implement your own stats, by using the methods
    # of the datasetWrapper directly. You can get frames with its
    # getFrame(frameId) method for example. Have a look at the class' documentation
    # inside the ./api/hackathon.py file!
    # solution.run(RunModes.TASK_A_FULL)
    solution.run(RunModes.TASK_B_FULL)
    # solution.run(RunModes.INTEGRATED_FULL)
    # The visualization run mode only shows the algorithm performing live on
    # a video. The only thing it really tests is whether your algorithm can
    # run in real-time. Its primary purpose is to provide a visualization however.
    # solution.run(RunModes.VISUALIZATION, videoFilePath=os.path.abspath("./data/demovideo.avi"))
