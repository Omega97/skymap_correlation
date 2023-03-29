"""Useful classes and functions"""
import healpy as hp
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from misc import correlation, ProgressBar


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

    def get_indices(self) -> list:
        """returns list of indices"""
        return self.indices


def masked_correlation(map1, map2, mask: Mask):
    """
    Calculates the correlation of the two maps, but ignore
    pixels that don't belong to the map

    map1, map2: arrays
    mask: Mask
    returns: float
    """
    return correlation(map1[mask.get_indices()], map2[mask.get_indices()])


def masked_correlation_2(map1, map2, indices):
    """
    Calculates the correlation of the two maps, but ignore
    pixels that don't belong to the map

    map1, map2: arrays
    mask: Mask
    returns: float
    """
    return correlation(map1[indices], map2[indices])


def correlation_map(map1, map2, proximity_matrix):
    """Compute the correlation map of two input maps given a proximity matrix"""
    bar = ProgressBar()
    npix = proximity_matrix.shape[0]
    if len(map1) != npix or len(map2) != npix:
        raise IndexError('Maps must share the same size as the proximity matrix')

    v = []
    for i in range(npix):
        indices = proximity_matrix[i].nonzero()[1]
        v.append(masked_correlation_2(map1, map2, indices))
        bar(i / (len(map1) - 1))
    return np.array(v)


def compute_proximity_masks(n_side, angle_rad, show_progress=False):     # todo can be made faster, save locally
    """
    For each pixel, compute the mask of adjacent pixels
    Formally equivalent to a proximity matrix
    O(n_pix^2)
    """
    progress_bar = ProgressBar() if show_progress else False

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
        if show_progress:
            progress_bar(row**2 / (npix-1)**2)

        for col in range(row):
            cos_ang = sc[row] * sc[col]
            cos_ang += ss[row] * ss[col]
            cos_ang += cos_theta[row] * cos_theta[col]

            if cos_ang >= cos_angle:
                out[row].append(col)
                out[col].append(row)

    if progress_bar:
        print()

    return [Mask(v, npix) for v in out]


def compute_proximity_matrix(n_side, angle_rad, show_progress=False):
    progress_bar = ProgressBar() if show_progress else False

    cos_angle = np.cos(angle_rad)
    npix = hp.nside2npix(n_side)
    rows = list(range(npix))
    cols = list(range(npix))

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
        if show_progress:
            progress_bar(row**2 / (npix-1)**2)

        for col in range(row):
            cos_ang = sc[row] * sc[col]
            cos_ang += ss[row] * ss[col]
            cos_ang += cos_theta[row] * cos_theta[col]

            if cos_ang >= cos_angle:
                rows += [row, col]
                cols += [col, row]

    if progress_bar:
        print()

    data = np.ones(len(cols))
    mat = csr_matrix((data, (rows, cols)), shape=(npix, npix), dtype=np.uint8)
    return mat


def compute_proximity_matrix_fast(n_side, angle_rad, show_progress=False):
    progress_bar = ProgressBar() if show_progress else False

    max_dist = np.cos(angle_rad)
    npix = hp.nside2npix(n_side)

    # Convert pixel indices to spherical coordinates
    theta, phi = hp.pix2ang(n_side, np.arange(npix))

    # pre-computing
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    ss = sin_theta * sin_phi
    sc = sin_theta * cos_phi

    neighbors_cache = dict()

    def dist(i, j):
        return sc[i] * sc[j] + ss[i] * ss[j] + cos_theta[i] * cos_theta[j]

    def get_close_indices(i) -> list:
        """get indices of pixels close to the i-th pixel"""
        out = {i}
        new = {i}
        while len(new):
            item = new.pop()
            if item in neighbors_cache:
                neighbors = neighbors_cache[item]
            else:
                neighbors = hp.get_all_neighbours(n_side, item)
                neighbors_cache[item] = neighbors
            for j in neighbors:
                if j not in out:
                    if dist(i, j) >= max_dist:
                        out.add(j)
                        new.add(j)
        return [i % npix for i in out]

    def compute_matrix_indices():
        rows = []
        cols = []
        for row in range(npix):
            if show_progress:
                progress_bar(row / (npix-1))

            indices = get_close_indices(row)
            rows += [row] * len(indices) + indices
            cols += indices + [row] * len(indices)

        if progress_bar:
            print()

        return rows, cols

    return compute_matrix_indices()


def get_temperature_range_mask(sky_map, x_min, x_max) -> Mask:
    """ returns a mask of the pixels in a given temperature range"""
    return Mask([i for i, x in enumerate(sky_map) if x_min <= x <= x_max], len(sky_map))


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
