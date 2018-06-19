"""This is an example module showing how the API should be used."""
from api.hackathon import HackathonApi, RunModes
import os
import numpy as np


class MySolution(HackathonApi):
    """DOCSTRING"""

    def handleFrameForTaskA(self, frame):
        """See the documentation in the parent class for information on this method."""
        x = [
            np.array([[
                0.26593406593406593,
                0.7065217391304348
            ], [
                0.4642857142857143,
                0.6956521739130435
            ], [
                0.46373626373626375,
                0.7815217391304348
            ], [
                0.2681318681318681,
                0.7945652173913044
            ]])]
        y = [np.array([[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]])]
        z = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]
        w = [
            np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),
            np.array([[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]),
            np.array([[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]),
            np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]])
        ]
        return w

    def handleFrameForTaskB(self, frame, regionCoordinates):
        """See the documentation in the parent class for information on this method."""
        return "XX-XX-1"


if __name__ == "__main__":
    solution = None
    try:
        solution = MySolution()
        datasetWrapper = solution.initializeApi(os.path.abspath("./metadata.json"), os.path.abspath("./data/"))
        print("MySolution begins here...")
        print("The total number of frames is {:d}".format(datasetWrapper.getTotalFrameCount()))
        solution.run(RunModes.TASK_B_SINGLE, frameId=0)
    finally:
        del solution
