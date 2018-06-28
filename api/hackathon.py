"""
You may change this file, but at the evaluation it will be replaced
by the original file :^)
Its documentation may help you though.
Any data structure with at least one leading underscore is explicitly intended
to not be used from outside this module.
"""
from enum import Enum
import abc
import json
import cv2 as cv
import numpy as np
import time
import re
import sys
import os
import copy
import random
from munkres import Munkres


class DatasetWrapper:
    """
    Wraps input from the dataset.

    You can use this class to obtain frames and region metadata.
    If you want to debug your code, you may use this class to execute your algorithm
    on a specific frame and check its result against the target values yourself.

    Evaluation is done with the run-methods from the HackathonApi though.
    """

    def __init__(self, labelFilePath, dataFolderPath):
        """Constructor."""
        self.__metadata = self.__loadMetadata(labelFilePath, dataFolderPath)

    def __loadMetadata(self, metadataFilePath, dataFolderPath):
        """Load all metadata."""
        result = []
        with open(metadataFilePath, "r") as metadataFile:
            jsonData = json.load(metadataFile)
            jsonData = jsonData["imagepath"]
            for i in range(len(jsonData)):
                framePath = os.path.join(dataFolderPath, jsonData[i]["path"])
                for j in range(len(jsonData[i]["frames"])):
                    resultObj = {}
                    resultObj["filepath"] = os.path.join(framePath, jsonData[i]["frames"][j]["filename"])
                    # resultObj["framesize"] = jsonData[i]["frames"][j]["framesize"]
                    resultObj["rois"] = []
                    for k in range(len(jsonData[i]["frames"][j]["rois"])):
                        if jsonData[i]["frames"][j]["rois"][k]["type"] == "Numberplate_Label":
                            roiObj = {}
                            roiObj["coordinates"] = np.array(jsonData[i]["frames"][j]["rois"][k]["coordinates"])
                            roiObj["label"] = jsonData[i]["frames"][j]["rois"][k]["label"]
                            resultObj["rois"].append(roiObj)
                    if len(resultObj["rois"]) > 0:
                        result.append(resultObj)
        return result

    def getTotalFrameCount(self):
        """Return total number of frames specified in the metadata file."""
        return len(self.__metadata)

    def getTotalRoisForFrame(self, frameId):
        """Return total number of regions in the frame with ID frameId."""
        return len(self.__metadata[frameId]["rois"])

    def getFrame(self, frameId):
        """Read the frame with the specified ID from disk and return it as a numpy matrix."""
        return cv.imread(self.__metadata[frameId]["filepath"])

    def getRois(self, frameId):
        """Return the regions of interest from the metadata for the specified frame."""
        return copy.deepcopy(self.__metadata[frameId]["rois"])


class RunModes(Enum):
    """Different run modes, to test certain tasks only or only perform on a random subset of all available data files."""

    # Test Task A's implementation on a single selectable or random frame
    TASK_A_SINGLE = 0
    # Test Task A's implementation on the whole dataset
    TASK_A_FULL = 1
    # Test Task B's implementation on a single selectable or random frame
    TASK_B_SINGLE = 2
    # Test Task B's implementation on the whole dataset
    TASK_B_FULL = 3
    # Test Task B on Task A's output on a single selectable or random frame
    INTEGRATED_SINGLE = 4
    # Test Task B on Task A's output on the whole dataset
    INTEGRATED_FULL = 5
    # Run both tasks in integrated fashion, but use a video as input and visualize the results.
    VISUALIZATION = 6


class HackathonApi:
    """
    Use this class as a parent class for your implementation.

    The methods you need to implement are at the top of the class.
    """

    __metaclass__ = abc.ABCMeta

    ###
    #   Required API for child classes
    ###

    @abc.abstractmethod
    def handleFrameForTaskA(self, frame):
        """
        Implement a solution to task A in this method.

        The method receives a frame (a three dimensional numpy array).


        Result requirements:
        - The method is expected to return a list of two dimensional numpy arrays with exactly 4
          elements on their first dimension and 2 elements on their second dimension.
          I.e. the form must be [ [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [...] ].
          (It describes a list of regions, which are in turn defined by their four
          edge points.)
        - It must be a python list of numpy arrays.
        - The four points must always be convex.
        - Each point's coordinates must be relative (i.e. in the range 0-1),
          describing the location of the point in relation to the width and height
          of the frame.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def handleFrameForTaskB(self, frame, regionCoordinates):
        """
        Implement a solution to task B in this method.

        The method receives a frame (a three dimensional numpy array) and a list
        of four points, which in turn a lists of two numbers.
        The form of the coordinates is therefore:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] and describes a convex quadriliteral
        in which a number plate is present.


        Result requirements:
        - The method is expected to return a string, that equals the number plate
          in the input region or None.
        - The output string must match the regular expression:
          ^[a-zA-ZöÖüÜäÄ]{1,3}-[a-zA-ZöÖüÜäÄ]{1,2}-[1-9][0-9]{0,3}[eE]?$
        - Special number plates, that are not of this format, have to return None.
        - If the number plate can not be read the output must be None
        """
        raise NotImplementedError

    ###
    #   Public methods
    ###

    def initializeApi(self, metadataFilePath, dataFolderPath):
        """Load all metadata and set up everything needed to run."""
        print("Python version: {}\nOpenCV version: {}".format(sys.version.replace("\n", ""), cv.__version__))
        self.__runModeFunctions = {
            RunModes.TASK_A_SINGLE: self.__runTaskASingle,
            RunModes.TASK_A_FULL: self.__runTaskAFull,
            RunModes.TASK_B_SINGLE: self.__runTaskBSingle,
            RunModes.TASK_B_FULL: self.__runTaskBFull,
            RunModes.INTEGRATED_SINGLE: self.__runIntegratedSingle,
            RunModes.INTEGRATED_FULL: self.__runIntegratedFull,
            RunModes.VISUALIZATION: self.__runVisualization
        }
        self.__datasetWrapper = DatasetWrapper(metadataFilePath, dataFolderPath)
        return self.__datasetWrapper

    def run(self, runMode, **kwargs):
        """
        Run the algorithm(s) in the provided mode.

        See the run modes documentation for info on what exactly they run.
        """
        if self.__runModeFunctions is not None and runMode in self.__runModeFunctions:
            print("Running {}".format(runMode.name))
            timeBegin = time.time()
            resultMetrics = self.__runModeFunctions[runMode](kwargs)
            if resultMetrics is not None:
                self.__printResultMetrics(resultMetrics)
            print("Ran {} in {:.3f} seconds.".format(runMode.name, time.time() - timeBegin))
        else:
            print("ERROR: No such mode: {}".format(runMode))

    ###
    #   Private methods
    ###

    def __checkOutputOfTaskA(self, image, regions, output):
        """Check the output of the child algorithm against the correct coordinates."""
        # Check the requirements first
        if not isinstance(output, list):
            print("ERROR: Output is not a list.")
            return None, 0, 0
        for i in range(len(output)):
            if not isinstance(output[i], np.ndarray):
                print("ERROR: Index {:d} is not a numpy array.".format(i))
                return None, 0, 0
            if output[i].shape != (4, 2):
                print("ERROR: Index {:d} expects shape (4, 2) but has shape {}".format(i, output[i].shape))
                return None, 0, 0
            for j in range(len(output[i])):
                for k in range(len(output[i][j])):
                    if output[i][j][k] < 0 or output[i][j][k] > 1:
                        print("ERROR: Index {:d} contains at least one non-relative coordinate.".format(i))
                        return None, 0, 0
            if not cv.isContourConvex(np.multiply(output[i], 10000).astype(int)):
                print("ERROR: Index {:d} is not a set of convex points.".format(i))
                return None, 0, 0
        # Match the ROI's to their best matching coordinates from the dataset
        # Create masks
        imageShape = image.shape
        outputMasks = []
        outputAreas = []
        for i in range(len(output)):
            outputMasks.append(np.zeros(imageShape, np.uint8))
            outputMasks[i] = cv.fillConvexPoly(outputMasks[i], np.multiply(output[i], np.array([imageShape[1], imageShape[0]])).astype(int), (255, 255, 255))
            outputMasks[i] = cv.threshold(outputMasks[i], 128, 255, cv.THRESH_BINARY)[1]
            outputAreas.append(np.count_nonzero(outputMasks[i]))
        regionMasks = []
        regionAreas = []
        for i in range(len(regions)):
            regionMasks.append(np.zeros(imageShape, np.uint8))
            regionMasks[i] = cv.fillConvexPoly(regionMasks[i], np.multiply(regions[i]["coordinates"], np.array([imageShape[1], imageShape[0]])).astype(int), (255, 255, 255))
            regionMasks[i] = cv.threshold(regionMasks[i], 128, 255, cv.THRESH_BINARY)[1]
            regionAreas.append(np.count_nonzero(regionMasks[i]))
        # Calculate cost matrix for the hungarian algorithm
        maxLength = max(len(outputMasks), len(regionMasks))
        matchingMatrix = np.ones((maxLength, maxLength), float).tolist()
        for i in range(len(outputMasks)):
            for j in range(len(regionMasks)):
                # Calculate matching coefficient
                intersectionArea = np.count_nonzero(cv.bitwise_and(outputMasks[i], regionMasks[j]))
                # Invert, since hung.alg. finds minimal cost
                matchingMatrix[i][j] = 1.0 - (float(intersectionArea) / float(outputAreas[i] + regionAreas[j] - intersectionArea))
        # Find best match with the hungarian algorithm
        m = Munkres()
        bestMatches = m.compute(matchingMatrix)
        result = []
        missedRois = 0
        unmatchedRois = 0
        for matchIndex in range(len(bestMatches)):
            outputIndex = bestMatches[matchIndex][0]
            regionIndex = bestMatches[matchIndex][1]
            if outputIndex < len(outputMasks):
                if regionIndex < len(regionMasks):
                    result.append([outputIndex, regionIndex, 1.0 - matchingMatrix[outputIndex][regionIndex]])
                else:
                    unmatchedRois = unmatchedRois + 1
            else:
                missedRois = missedRois + 1
        result = np.array(result)
        return result[result[:, 0].argsort()], missedRois, unmatchedRois

    def __checkOutputOfTaskB(self, correctLabel, output):
        """Check the output of the child algorithm against the correct coordinates."""
        # Check the requirements first
        if output is None:
            output = "XX-XX-1"
        if not isinstance(output, str):
            print("ERROR: Output is not a string.")
            return None
        if not re.match("^[a-zA-ZüÖöÜäÄ]{1,3}-[a-zA-ZöÖüÜäÄ]{1,2}-[1-9][0-9]{0,3}$", output):
            print("ERROR: Output does not match regular expression of a number plate.")
            return None
        if output.upper() != correctLabel.upper():
            return False
        return True

    def __printResultMetrics(self, resultMetrics):
        """Print the results of a run."""
        print("Results:")
        for metricName, metricValue in resultMetrics.items():
            if isinstance(metricValue, float):
                print("\t{}:\n\t{:.3f}".format(metricName, metricValue))
            elif isinstance(metricValue, int):
                print("\t{}:\n\t{:d}".format(metricName, metricValue))
            else:
                print("\t{}:\n\t{}".format(metricName, metricValue))

    def __runTaskASingle(self, kwargs):
        """
        Take the frame with the ID specified in the kwargs and execute the algorithm only on this frame.

        If no frameId was specified, a random one is selected.
        """
        if "frameId" not in kwargs or (not isinstance(kwargs["frameId"], int)) or kwargs["frameId"] < 0 or kwargs["frameId"] >= self.__datasetWrapper.getTotalFrameCount():
            print("INFO: No or invalid frameId, selecting random frame!")
            frameId = random.randint(0, self.__datasetWrapper.getTotalFrameCount()-1)
        else:
            frameId = kwargs["frameId"]
        resultMetrics = {}
        ###
        image = self.__datasetWrapper.getFrame(frameId)
        if image is None:
            print("ERROR: File for frame {:d} not found.".format(frameId))
            return None
        output = self.handleFrameForTaskA(image)
        regions = self.__datasetWrapper.getRois(frameId)
        result, missedRois, unmatchedRois = self.__checkOutputOfTaskA(image, regions, output)
        if result is None:
            return None
        ###
        resultMetrics["Frame ID"] = frameId
        resultMetrics["Average overlap of matched ROIs"] = np.mean(result[:, 2])
        resultMetrics["Missing ROIs"] = missedRois
        resultMetrics["Unmatched output ROIs"] = unmatchedRois
        resultMetrics["Final evaluation score"] = "{:.3f}%".format(((np.mean(result[:, 2]) * len(result)) / (len(result) + missedRois + unmatchedRois)) * 100)
        return resultMetrics

    def __runTaskAFull(self, kwargs):
        """Execute the algorithm on all frames in succession and measure outputs."""
        resultMetrics = {}
        allAverageAreaOverlaps = []
        allMissedRois = []
        allUnmatchedRois = []
        allEvaluationScores = []
        ###
        for i in range(self.__datasetWrapper.getTotalFrameCount()):
            image = self.__datasetWrapper.getFrame(i)
            if image is None:
                print("ERROR: File for frame {:d} not found.".format(i))
                return None
            output = self.handleFrameForTaskA(image)
            regions = self.__datasetWrapper.getRois(i)
            result, missedRois, unmatchedRois = self.__checkOutputOfTaskA(image, regions, output)
            if result is None:
                return None
            allAverageAreaOverlaps.append(np.mean(result[:, 2]))
            allMissedRois.append(missedRois)
            allUnmatchedRois.append(unmatchedRois)
            allEvaluationScores.append((np.mean(result[:, 2]) * len(result)) / (len(result) + missedRois + unmatchedRois))
        ###
        resultMetrics["Total frames"] = self.__datasetWrapper.getTotalFrameCount()
        resultMetrics["Average overlap of matched ROIs"] = np.mean(allAverageAreaOverlaps)
        resultMetrics["Missing ROIs"] = np.sum(allMissedRois)
        resultMetrics["Unmatched output ROIs"] = np.sum(allUnmatchedRois)
        # resultMetrics["Final evaluation score per frame"] = np.divide(np.multiply(np.array(allEvaluationScores), 1000).astype(int).astype(float), 1000).tolist()
        resultMetrics["Final evaluation score"] = "{:.3f}%".format(np.mean(allEvaluationScores) * 100)
        return resultMetrics

    def __runTaskBSingle(self, kwargs):
        """
        Take the frame with the ID specified in the kwargs and execute the algorithm only on all ROIs in this frame.

        If no frameId was specified, a random one is selected.
        """
        if "frameId" not in kwargs or (not isinstance(kwargs["frameId"], int)) or kwargs["frameId"] < 0 or kwargs["frameId"] >= self.__datasetWrapper.getTotalFrameCount():
            print("INFO: No or invalid frameId, selecting random frame!")
            frameId = random.randint(0, self.__datasetWrapper.getTotalFrameCount()-1)
        else:
            frameId = kwargs["frameId"]
        resultMetrics = {}
        totalRegions = self.__datasetWrapper.getTotalRoisForFrame(frameId)
        totalReadable = 0
        correctRegions = 0
        correctNonNone = 0
        ###
        image = self.__datasetWrapper.getFrame(frameId)
        if image is None:
            print("ERROR: File for frame {:d} not found.".format(frameId))
            return None
        regions = self.__datasetWrapper.getRois(frameId)
        for i in range(len(regions)):
            coordinates = regions[i]["coordinates"]
            label = regions[i]["label"]
            if label.upper() != "XX-XX-1":
                totalReadable = totalReadable + 1
            output = self.handleFrameForTaskB(image, coordinates)
            result = self.__checkOutputOfTaskB(label, output)
            if result is None:
                return None
            correctRegions = correctRegions + (1 if result else 0)
            correctNonNone = correctNonNone + (1 if result and output is not None and (not isinstance(output, str) or output.upper() != "XX-XX-1") else 0)
        ###
        resultMetrics["Frame ID"] = frameId
        resultMetrics["Total regions"] = totalRegions
        resultMetrics["Total readable regions"] = totalReadable
        resultMetrics["Correctly classified"] = "{:d} ({:.2f}%)".format(correctRegions, (float(correctRegions) / totalRegions) * 100)
        resultMetrics["Correctly classified readable"] = "{:d} ({:.2f}%)".format(correctNonNone, (float(correctNonNone) / totalReadable) * 100)
        return resultMetrics

    def __runTaskBFull(self, kwargs):
        """Execute the algorithm on all frames in succession and measure outputs."""
        resultMetrics = {}
        totalRegions = 0
        totalReadable = 0
        correctRegions = 0
        correctNonNone = 0
        ###
        for i in range(self.__datasetWrapper.getTotalFrameCount()):
            image = self.__datasetWrapper.getFrame(i)
            if image is None:
                print("ERROR: File for frame {:d} not found.".format(i))
                return None
            regions = self.__datasetWrapper.getRois(i)
            totalRegions = totalRegions + len(regions)
            for i in range(len(regions)):
                coordinates = regions[i]["coordinates"]
                label = regions[i]["label"]
                if label.upper() != "XX-XX-1":
                    totalReadable = totalReadable + 1
                output = self.handleFrameForTaskB(image, coordinates)
                result = self.__checkOutputOfTaskB(label, output)
                if result is None:
                    return None
                correctRegions = correctRegions + (1 if result else 0)
                correctNonNone = correctNonNone + (1 if result and output is not None and (not isinstance(output, str) or output.upper() != "XX-XX-1") else 0)
        ###
        resultMetrics["Total frames"] = self.__datasetWrapper.getTotalFrameCount()
        resultMetrics["Total regions"] = totalRegions
        resultMetrics["Total readable regions"] = totalReadable
        resultMetrics["Correctly classified"] = "{:d} ({:.2f}%)".format(correctRegions, (float(correctRegions) / totalRegions) * 100)
        resultMetrics["Correctly classified readable"] = "{:d} ({:.2f}%)".format(correctNonNone, (float(correctNonNone) / totalReadable) * 100)
        return resultMetrics

    def __runIntegratedSingle(self, kwargs):
        """
        Take the frame with the ID specified in the kwargs and execute the algorithm only on this frame.

        Both algorithms are executed in sequence. I.e. the output of task A is fed as an input to task B.
        If no frameId was specified, a random one is selected.
        """
        if "frameId" not in kwargs or (not isinstance(kwargs["frameId"], int)) or kwargs["frameId"] < 0 or kwargs["frameId"] >= self.__datasetWrapper.getTotalFrameCount():
            print("INFO: No or invalid frameId, selecting random frame!")
            frameId = random.randint(0, self.__datasetWrapper.getTotalFrameCount()-1)
        else:
            frameId = kwargs["frameId"]
        resultMetrics = {}
        totalRegionsMatched = 0
        correctRegions = 0
        ###
        image = self.__datasetWrapper.getFrame(frameId)
        if image is None:
            print("ERROR: File for frame {:d} not found.".format(frameId))
            return None
        outputTaskA = self.handleFrameForTaskA(image)
        regions = self.__datasetWrapper.getRois(frameId)
        matches, missedRois, unmatchedRois = self.__checkOutputOfTaskA(image, regions, outputTaskA)
        if matches is None:
            return None
        totalRegionsMatched = totalRegionsMatched + len(matches)
        for i in range(len(matches)):
            coordinates = outputTaskA[int(matches[i][0])]
            label = regions[int(matches[i][1])]["label"]
            outputTaskB = self.handleFrameForTaskB(image, coordinates)
            result = self.__checkOutputOfTaskB(label, outputTaskB)
            if result is None:
                return None
            correctRegions = correctRegions + (1 if result else 0)
        ###
        taskAScore = (np.mean(matches[:, 2]) * len(matches)) / (len(matches) + missedRois + unmatchedRois)
        taskBScore = float(correctRegions) / totalRegionsMatched
        resultMetrics["Frame ID"] = frameId
        resultMetrics["Total num of matched regions"] = "{:d} (+{:d}/-{:d})".format(totalRegionsMatched, unmatchedRois, missedRois)
        resultMetrics["Task A - Final evaluation score"] = "{:.3f}%".format(taskAScore * 100)
        resultMetrics["Task B - Correctly classified regions of Task A"] = "{:.3f}%".format(taskBScore * 100)
        resultMetrics["Combined score"] = "{:.3f}%".format((taskAScore * taskBScore) * 100)
        return resultMetrics

    def __runIntegratedFull(self, kwargs):
        """
        Execute the algorithm on all frames in succession and measure outputs.

        Both algorithms are executed in sequence. I.e. the output of task A is fed as an input to task B.
        """
        resultMetrics = {}
        allEvaluationScores = []
        allMissedRois = []
        allUnmatchedRois = []
        totalRegionsMatched = 0
        correctRegions = 0
        ###
        for i in range(self.__datasetWrapper.getTotalFrameCount()):
            image = self.__datasetWrapper.getFrame(i)
            if image is None:
                print("ERROR: File for frame {:d} not found.".format(i))
                return None
            outputTaskA = self.handleFrameForTaskA(image)
            regions = self.__datasetWrapper.getRois(i)
            matches, missedRois, unmatchedRois = self.__checkOutputOfTaskA(image, regions, outputTaskA)
            if matches is None:
                return None
            allMissedRois.append(missedRois)
            allUnmatchedRois.append(unmatchedRois)
            allEvaluationScores.append((np.mean(matches[:, 2]) * len(matches)) / (len(matches) + missedRois + unmatchedRois))
            totalRegionsMatched = totalRegionsMatched + len(matches)
            for i in range(len(matches)):
                coordinates = outputTaskA[int(matches[i][0])]
                label = regions[int(matches[i][1])]["label"]
                outputTaskB = self.handleFrameForTaskB(image, coordinates)
                result = self.__checkOutputOfTaskB(label, outputTaskB)
                if result is None:
                    return None
                correctRegions = correctRegions + (1 if result else 0)
        ###
        taskAScore = np.mean(allEvaluationScores)
        taskBScore = float(correctRegions) / totalRegionsMatched
        resultMetrics["Total frames"] = self.__datasetWrapper.getTotalFrameCount()
        resultMetrics["Total num of matched regions"] = "{:d} (+{:d}/-{:d})".format(totalRegionsMatched, np.sum(allUnmatchedRois), np.sum(allMissedRois))
        resultMetrics["Task A - Final evaluation score"] = "{:.3f}%".format(taskAScore * 100)
        resultMetrics["Task B - Correctly classified regions of Task A"] = "{:.3f}%".format(taskBScore * 100)
        resultMetrics["Combined score"] = "{:.3f}%".format((taskAScore * taskBScore) * 100)
        return resultMetrics

    def __runVisualization(self, kwargs):
        """Run a visualization showing the results of the algorithm live in a video."""
        if "videoFilePath" not in kwargs or (not isinstance(kwargs["videoFilePath"], str)) or (not os.path.isfile(kwargs["videoFilePath"])):
            print("ERROR: No or invalid file path to video!")
            return None
        videoCapture = cv.VideoCapture(kwargs["videoFilePath"])
        windowCreated = False
        while videoCapture.isOpened():
            # Read video frame
            ret, image = videoCapture.read()
            imageShape = image.shape
            if not windowCreated:
                windowCreated = True
                cv.namedWindow("Demo video", cv.WINDOW_NORMAL)
                aspectRatio = float(imageShape[1]) / float(imageShape[0])
                targetWidth = 600
                targetHeight = int(targetWidth * aspectRatio)
                cv.resizeWindow("Demo video", targetHeight, targetWidth)
            # Run algorithm
            output = []
            outputTaskA = self.handleFrameForTaskA(image)
            for i in range(len(outputTaskA)):
                coordinates = outputTaskA[i]
                outputTaskB = self.handleFrameForTaskB(image, coordinates)
                output.append({
                    "coordinates": coordinates,
                    "label": outputTaskB if outputTaskB is not None else "XX-XX-1"
                })
            # Visualize output
            for i in range(len(output)):
                scaledCoordinates = np.multiply(output[i]["coordinates"], np.array([imageShape[1], imageShape[0]])).astype(int)
                cv.drawContours(image, [scaledCoordinates], -1, (255, 255, 255), 5, cv.LINE_8)
                textSize = cv.getTextSize(output[i]["label"], cv.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                cv.putText(image, output[i]["label"], (scaledCoordinates[0][0] + 10, scaledCoordinates[0][1] + 10 + textSize[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_8, False)
            # Show result
            cv.imshow('Demo video', image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        videoCapture.release()
        cv.destroyAllWindows()
        return None
