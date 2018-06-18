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
        return z

    def handleFrameForTaskB(self):
        """See the documentation in the parent class for information on this method."""
        return None


if __name__ == "__main__":
    solution = None
    try:
        solution = MySolution()
        solution.initializeApi(os.path.abspath("./result.json"), os.path.abspath("./data/"))
        print("MySolution begins here...")
        print("The total number of frames is {:d}".format(solution.getTotalFrameCount()))
        solution.run(RunModes.TASK_A_SINGLE, frameId=0)
    finally:
        del solution
