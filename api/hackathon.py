"""DOCSTRING"""
from enum import Enum
import abc
import json
import cv2 as cv
import sys
import os


class DataReader:
    """Wraps input from the dataset."""

    def __init__(self, labelFilePath, dataFolderPath):
        """Constructor."""
        self.__metadata = self.__loadMetadata(labelFilePath, dataFolderPath)

    def __del__(self):
        """Destructor."""
        pass

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
                            roiObj["coordinates"] = jsonData[i]["frames"][j]["rois"][k]["coordinates"]
                            roiObj["label"] = jsonData[i]["frames"][j]["rois"][k]["label"]
                            resultObj["rois"].append(roiObj)
                    if len(resultObj["rois"]) > 0:
                        result.append(resultObj)
        return result

    def getTotalFrameCount(self):
        """Return total number of frames specified in the metadata file."""
        return len(self.__metadata)


class RunModes(Enum):
    """Different run modes, to test certain tasks only or only perform on a random subset of all available data files."""

    TASK_A_SINGLE = 0
    TASK_A_FULL = 1
    TASK_B_SINGLE = 2
    TASK_B_FULL = 3
    FULL_RUN = 4


class HackathonApi:
    """Use this class as a parent class to ..."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, metadataFilePath, dataFolderPath):
        """Constructor."""
        print("Python version: {}\nOpenCV version: {}".format(sys.version.replace("\n", ""), cv.__version__))
        self.__runModeFunctions = {
            RunModes.TASK_A_SINGLE: self.__runTaskASingle,
            RunModes.TASK_A_FULL: self.__runTaskAFull,
            RunModes.TASK_B_SINGLE: self.__runTaskBSingle,
            RunModes.TASK_B_FULL: self.__runTaskBFull,
            RunModes.FULL_RUN: self.__runFullRun,
        }
        self.__dataReader = DataReader(metadataFilePath, dataFolderPath)

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

    def __runTaskASingle(self, **kwargs):
        """Take the frame with the ID specified in the kwargs and execute the algorithm only on this frame."""
        if "frameId" not in kwargs or (not isinstance(kwargs["frameId"], int)) or kwargs["frameId"] < 0 or kwargs["frameId"] >= self.getTotalFrameCount():
            print("ERROR: No or invalid frameId!")
            return None
        resultMetrics = {
            "Total frames", 1
        }
        averageAreaOverlap = 0
        ###

        ###
        resultMetrics["Average overlap of result and target area"] = averageAreaOverlap
        return resultMetrics

    def __runTaskAFull(self, kwargs):
        """DOCSTRING"""
        resultMetrics = {}
        totalFrames = 0
        averageAreaOverlap = 0
        ###
        ###
        resultMetrics["Total frames"] = totalFrames
        resultMetrics["Average overlap of result and target area"] = averageAreaOverlap
        return resultMetrics

    def __runTaskBSingle(self, kwargs):
        """DOCSTRING"""
        resultMetrics = {
            "Total frames", 1
        }
        ###
        ###
        return resultMetrics

    def __runTaskBFull(self, kwargs):
        """DOCSTRING"""
        resultMetrics = {}
        totalFrames = 0
        ###
        ###
        resultMetrics["Total frames"] = totalFrames
        return resultMetrics

    def __runFullRun(self, kwargs):
        """DOCSTRING"""
        resultMetrics = {
            "Total frames", self.getTotalFrameCount()
        }
        ###
        ###
        return resultMetrics

    def getTotalFrameCount(self):
        """Return the number of total frames specified in the metadata."""
        return self.__dataReader.getTotalFrameCount()

    def run(self, runMode, **kwargs):
        """Run the selected mode."""
        if runMode in self.__runModeFunctions:
            print("Running {}".format(runMode))
            resultMetrics = self.__runModeFunctions[runMode](kwargs)
            if resultMetrics is not None:
                self.__printResultMetrics(resultMetrics)
        else:
            print("ERROR: No such mode: {}".format(runMode))

    @abc.abstractmethod
    def construct(self):
        """Constructor for child classes."""
        pass

    @abc.abstractmethod
    def destruct(self):
        """Destructor for child classes."""
        pass
