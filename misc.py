"""Miscellaneous functions"""
import numpy as np


def sigma(x):
    """standard deviation """
    return np.linalg.norm(x - np.average(x))


def cov(x, y):
    """covariance of two arrays"""
    return sum((x - np.average(x)) * (y - np.average(y)))


def correlation(x, y):
    """correlation of two arrays"""
    den = sigma(x) * sigma(y)
    if den != 0:
        return cov(x, y) / den
    else:
        return 0


def histogram_equalized(arr):
    """histogram-equalization normalizes the image and enhances contrast"""
    n = len(arr)
    v = sorted(range(n), key=arr.__getitem__)
    w = sorted(range(n), key=v.__getitem__)
    return np.array(w) / (n-1)


def cos_angle_between_unit_vectors(theta1, phi1, theta2, phi2):
    """angle between two unitary vectors"""
    out = np.sin(theta1) * np.sin(theta2) * (np.cos(phi1-phi2)-1)
    out += np.cos(theta1-theta2)
    return out
