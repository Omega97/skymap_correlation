"""Useful classes and functions"""
import healpy as hp
from astropy.stats import bayesian_blocks
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from misc import correlation


class Mask(np.ndarray):
    """
    This class subclasses np.ndarray to create a masked array given a list of indices.
    indices: list of indices to be set to True
    length: length of the array
    """
    def __new__(cls, indices: list, length: int):
        # cls.indices = indices
        arr = np.zeros(length, dtype=np.int8)
        arr[indices] = True
        # cast the array to the subclass
        obj = np.asarray(arr).view(cls)
        obj.indices = indices
        return obj


def masked_correlation(map1, map2, mask: Mask):
    """
    Calculates the correlation of the two maps, but ignore
    pixels that don't belong to the map

    map1, map2: arrays
    mask: Mask
    returns: float
    """
    return correlation(map1[mask.indices], map2[mask.indices])


def correlation_map(map1, map2, proximity_matrix: List[Mask]):
    """Compute the correlation map of two input maps given a proximity matrix"""
    if len(map1) != len(map2):
        raise IndexError('Maps must share the same size')
    v = [masked_correlation(map1, map2, proximity_matrix[i]) for i in range(len(map1))]
    return np.array(v)


def compute_proximity_masks(n_side, angle_rad):
    """
    For each pixel, compute the mask of adjacent pixels
    Formally equivalent to a proximity matrix
    O(n_pix^2)
    """
    cos_angle = np.cos(angle_rad)
    npix = hp.nside2npix(n_side)
    out = [[i] for i in range(npix)]

    # Convert pixel indices to spherical coordinates
    theta, phi = hp.pix2ang(n_side, np.arange(npix))

    # pre-computing
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    ss = sin_theta * sin_phi
    sc = sin_theta * cos_phi

    # compute matrix
    for row in range(npix):

        for col in range(row):
            cos_ang = sc[row] * sc[col]
            cos_ang += ss[row] * ss[col]
            cos_ang += cos_theta[row] * cos_theta[col]

            if cos_ang >= cos_angle:
                out[row].append(col)
                out[col].append(row)

    return [Mask(v, npix) for v in out]


def get_temperature_range_mask(sky_map, x_min, x_max) -> Mask:
    """ returns a mask of the pixels in a given temperature range"""
    return Mask([i for i, x in enumerate(sky_map) if x_min <= x <= x_max], len(sky_map))


def temperature_bands_plot(map1, map2, n_bands=5):
    """ plot sky-maps segmented by temperature """
    _ = plt.figure(figsize=(18, 4))
    for i in range(n_bands):
        x_min = i/n_bands
        x_max = (i+1)/n_bands
        mask = get_temperature_range_mask(map1, x_min, x_max)

        map_a = map1 * mask
        map_b = map2 * mask

        title = f'{x_min:.0%} -- {x_max:.0%}'
        hp.mollview(map_a, title=title, cmap='jet', cbar=False,
                    bgcolor='0.1', sub=(2, n_bands, 1+i))
        hp.mollview(map_b, title='', cmap='jet', cbar=False,
                    bgcolor='0.1', sub=(2, n_bands, 1+i+n_bands))


def adaptive_proximity(sky_map, prox, p_temp):
    """
    returns a list of pixels in proximity to each given pixel, that also
    satisfy some temperature requirement
    """
    temp = [np.average([sky_map[j] for j in v]) * p_temp for v in prox]
    return [[j for j in v if sky_map[j] >= temp[j]] for v in prox]


def corr_analysis(map1, map2, n_bands=5):
    """compute the correlation between sky-maps for reach temperature band"""
    mat = np.zeros(n_bands)
    for i in range(n_bands):
        x_max = (i+1) / n_bands
        x_min = i / n_bands
        mask = get_temperature_range_mask(map1, x_min, x_max)
        mat[i] = masked_correlation(map1, map2, mask)
    return mat


def correlation_plot(map1, map2, n_bands=5):
    """plot correlation between sky-maps for reach temperature band"""
    _x = np.linspace(1, n_bands, n_bands)
    ticks = [f'{i/n_bands:.0%} -- {(i+1)/n_bands:.0%}' for i in range(n_bands)]
    ax = plt.axes()
    ax.set_facecolor("0.1")
    plt.bar(_x, corr_analysis(map1, map2, n_bands))
    plt.xticks(_x, ticks)
    plt.title('Temperature band correlation')


def partition_brightness_spectrum(sky_map):
    edges = bayesian_blocks(sky_map, p0=10 ** -7)
    print(', '.join([f'{x:.2f}' for x in edges]))
    plt.hist(sky_map, bins=list(edges), density=True)
    plt.title(f'Bayesian Blocks 408 Mhz spectrum')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.show()
