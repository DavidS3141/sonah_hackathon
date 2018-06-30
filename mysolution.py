"""This is an example module showing how the API should be used."""
from api.hackathon import HackathonApi, RunModes
from LLT import LLT
import os
import time
import hashlib
import json
import random
import numpy as np


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

    def handleFrameForTaskA(self, frame):
        """
        See the documentation in the parent class for a whole lot of information on this method.

        We will just stupidly return random ROIs here
        to show what the result has to look like.
        """
        LowLevelTrigger = LLT()
        possiblePlates = LowLevelTrigger.detectPossiblePlates(frame)
        wh = np.array(frame.shape)[1::-1]

        plates = [plate/wh for plate in possiblePlates if plate is not None]
        if len(plates) == 0:
            plates = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
        return plates # relative positions of rectangles

    def handleFrameForTaskB(self, frame, regionCoordinates):
        """
        See the documentation in the parent class for a whole lot of information on this method.

        We will just stupidly return None here,
        which basically stands for "I can't read this".
        """
        rng = random.Random()
        m = hashlib.md5()
        m.update(json.dumps(regionCoordinates.tolist()).encode())
        rng.seed(m.hexdigest())
        resultSet = [
            None,
            "AC-FT-774",
            "MU-YG-728",
            "HB-KZ-3124"
        ]
        return resultSet[int(((time.time() / 10.0) + rng.randrange(0, len(resultSet))) % len(resultSet))]


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
    solution.run(RunModes.TASK_A_FULL)
    # solution.run(RunModes.TASK_B_FULL)
    # solution.run(RunModes.INTEGRATED_FULL)
    # The visualization run mode only shows the algorithm performing live on
    # a video. The only thing it really tests is whether your algorithm can
    # run in real-time. Its primary purpose is to provide a visualization however.
    # solution.run(RunModes.VISUALIZATION, videoFilePath=os.path.abspath("./data/demovideo.avi"))
