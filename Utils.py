"""Implements utility functions"""
import numpy as np

def konvex_rectangle(rectangle):
    xpy = rectangle[:,0]+rectangle[:,1]
    xmy = rectangle[:,0]-rectangle[:,1]
    idx1 = np.argsort(xpy)[0]
    idx2 = np.argsort(xmy)[-1]
    idx3 = np.argsort(xpy)[-1]
    idx4 = np.argsort(xmy)[0]
    if len(set([idx1, idx2, idx3, idx4])) < 4:
        return None
    return rectangle[[idx1, idx2, idx3, idx4], :]

def rescale_squares(squares, wh, width):
    return [square/width*wh[0] for square in squares]
