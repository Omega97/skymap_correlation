"""Sky-map"""
import healpy as hp
from astropy.io import fits
from typing import List
from misc import histogram_equalized, take_time
from utils import correlation, compute_proximity_masks, Mask, \
    correlation_map, get_temperature_range_mask, masked_correlation


class SkyMapAnalyzer:
    """
    This class performs the analysis of the correlation between sky-maps
    """
    def __init__(self, *map_paths, verbose=False):
        self.paths = map_paths
        self.verbose = verbose
        self.maps = None
        self.n_side = None
        self.n_pixels = None
        self._load_data()

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        return iter(self.maps)

    @take_time
    def _load_data(self):
        """load locally-saved data"""
        self.maps = []
        for i in range(len(self)):
            with fits.open(self.paths[i]) as lambda_list_1:
                if self.verbose:
                    print()
                    lambda_list_1.info()
                self.maps.append(hp.read_map(lambda_list_1))

    @take_time
    def normalize(self):
        normalized_maps = []
        for sky_map in self:
            normalized_maps.append(histogram_equalized(sky_map))
        self.maps = normalized_maps

    @take_time
    def downscale(self, new_n_side):
        self.n_side = new_n_side
        self.n_pixels = hp.nside2npix(new_n_side)
        for i in range(len(self)):
            self.maps[i] = hp.ud_grade(map_in=self.maps[i], nside_out=new_n_side)

    @take_time
    def compute_global_correlation(self, i=0, j=1):
        """ Global correlation of the whole normalized maps of indices i and j """
        return correlation(self.maps[i], self.maps[j])

    @take_time
    def compute_custom_correlation_maps(self, proximity_matrix: List[Mask], i=0, j=1):
        """
        Given a list of proximity masks, compute the correlation
        between masked maps of indices i and j

        :param proximity_matrix: list of masks that define whether two pixels are close or not
        :param i: first map index
        :param j: second map index
        :return: sky-map of correlation values
        """
        return correlation_map(self.maps[i], self.maps[j], proximity_matrix)

    @take_time
    def compute_local_correlation(self, angle_rad=.2, i=0, j=1):
        """for each pixel, compute the correlation between the maps for pixels within a certain angle
        :param angle_rad: distance within which the correlation is considered
        :param i: first map index
        :param j: second map index
        """
        masks = compute_proximity_masks(self.n_side, angle_rad)
        return self.compute_custom_correlation_maps(masks, i=i, j=j)

    @take_time
    def compute_temperature_correlation(self, spectral_segmentation_method, i=0, j=1, **kwargs) -> dict:
        """
        The first map is separated in temperature bands, then the correlation
        between the two maps is calculated in those regions
        :param spectral_segmentation_method: a method that separates the value distribution of a list
        by returning a list of edges of bins
        :param i: index of first map
        :param j: index of second map
        :param kwargs: keyword arguments for the spectral_segmentation_method
        :return: dictionary containing list of edges, list of masks, list of correlations
        between temperature bands, and number of bins
        """
        # compute edges
        edges = spectral_segmentation_method(self.maps[i], **kwargs)
        masks = []
        correlations = []

        for k in range(len(edges)-1):
            # find mask
            mask = get_temperature_range_mask(self.maps[i], edges[k], edges[k+1])
            masks.append(mask)

            # compute correlation
            c = masked_correlation(self.maps[i], self.maps[j], mask)
            correlations.append(c)

        return {'edges': edges, 'masks': masks, 'correlations': correlations, 'n_bins': len(masks)}
