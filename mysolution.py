"""This is an example module showing how the API should be used."""
from api.hackathon import HackathonApi, RunModes
import os
import numpy as np


class MySolution(HackathonApi):
    """DOCSTRING"""

    def handleFrameForTaskA(self, frame):
        """See the documentation in the parent class for information on this method."""
        w = [
            np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),
            np.array([[0.5, 0], [1, 0], [1, 0.5], [0.5, 0.5]]),
            np.array([[0, 0.5], [0.5, 0.5], [0.5, 1], [0, 1]]),
            np.array([[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]])
        ]
        return w

    def handleFrameForTaskB(self, frame, regionCoordinates):
        """See the documentation in the parent class for information on this method."""
        return None


if __name__ == "__main__":
    solution = None
    try:
        solution = MySolution()
        datasetWrapper = solution.initializeApi(os.path.abspath("./metadata.json"), os.path.abspath("./data/"))
        print("MySolution begins here...")
        print("The total number of frames is {:d}".format(datasetWrapper.getTotalFrameCount()))
        solution.run(RunModes.TASK_B_FULL)
        solution.run(RunModes.INTEGRATED_FULL)
    finally:
        del solution
