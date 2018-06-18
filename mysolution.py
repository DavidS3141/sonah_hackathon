"""This is an example module showing how the API can be used."""
from api.hackathon import HackathonApi, RunModes
import os


class MySolution(HackathonApi):
    """This is the class implementing my solution."""


if __name__ == "__main__":
    solution = None
    try:
        solution = MySolution(os.path.abspath("./labels.json"), os.path.abspath("./data/"))
        print("MySolution begins here...")
        print("The total number of frames is {:d}".format(solution.getTotalFrameCount()))
        solution.run(RunModes.TASK_A_FULL)
    finally:
        del solution
