"""Sky-map"""
import healpy as hp
from astropy.io import fits
import numpy as np
from typing import List

try:
    from misc import histogram_equalized
    from utils import correlation, proximity_masks, Mask
except ImportError:
    from .misc import histogram_equalized
    from .utils import correlation, proximity_masks, Mask


class SkyMapAnalysis:
    """
    This class performs the analysis of the correlation between sky-maps
    """
    def __init__(self, *map_paths, verbose=False):
        self.paths = map_paths
        self.maps = None
        self.normalized_maps = None
        self.nside = None
        self.n_pixels = None

        self._load_data(verbose=verbose)
        self._normalize_maps()

    def __len__(self):
        return len(self.paths)

    def _load_data(self, verbose=False):
        """load locally-saved data"""
        self.maps = []
        for i in range(len(self)):
            with fits.open(self.paths[i]) as lambda_list_1:
                if verbose:
                    lambda_list_1.info()
                self.maps.append(hp.read_map(lambda_list_1))

    def _normalize_maps(self):
        self.normalized_maps = []
        for i in range(len(self)):
            self.normalized_maps.append(histogram_equalized(self.maps[i]))

    def downscale(self, nside):
        self.nside = nside
        self.n_pixels = hp.nside2npix(nside)
        for i in range(len(self)):
            self.maps[i] = hp.ud_grade(map_in=self.maps[i], nside_out=nside)
        self._normalize_maps()

    def compute_global_correlation(self):
        """global correlation of the whole normalized maps"""
        return correlation(self.normalized_maps[0], self.normalized_maps[1])

    def compute_custom_correlation(self, proximity_masks: List[Mask]):
        """
        compute the correlation between the maps for each pixel given a list of proximity masks
        :param proximity_masks: list of masks that define whether two pixels are close or not
        :return: sky-map of correlation values
        """
        v = [correlation(self.normalized_maps[0][mask.indices],
                         self.normalized_maps[1][mask.indices])
             for mask in proximity_masks]
        return np.array(v)

    def compute_local_correlation(self, angle_rad=.2):
        """for each pixel, compute the correlation between the maps for pixels within a certain angle"""
        masks = proximity_masks(self.nside, angle_rad)
        return self.compute_custom_correlation(masks)
