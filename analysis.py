"""Sky-map"""
import healpy as hp
import numpy as np
from astropy.io import fits
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix
from misc import histogram_equalized, take_time, make_dir
from utils import correlation, correlation_map, get_temperature_range_mask, \
    masked_correlation, compute_proximity_matrix_fast
import os


class SkyMapAnalyzer(list):
    """
    This class performs the analysis of the correlation between sky-maps
    """
    def __init__(self, *map_paths, verbose=False):
        super().__init__()
        self.paths = map_paths
        self.verbose = verbose
        self.n_side = None
        self.n_pixels = None
        self.proximity_masks = None
        self.proximity_matrix = None
        self._load_data()

    @take_time
    def plot(self):
        for i, sky_map in enumerate(self):
            hp.mollview(sky_map, cmap='jet', title=f'map {i}', sub=(1, len(self), i + 1))

    def _update_n_side(self):
        self.n_side = None
        if len(self):
            sizes = [hp.npix2nside(len(m)) for m in self]
            if min(sizes) == max(sizes):
                self.n_side = min(sizes)

    def add_maps(self, *maps):
        """add new sky_maps"""
        for sky_map in maps:
            self.append(sky_map)
        self._update_n_side()

    @take_time
    def _load_data(self):
        """
        Load locally-saved data
        Each map is saved in the list maps as a numpy array
        """
        self.clear()
        for path in self.paths:
            with fits.open(path) as lambda_list_1:
                if self.verbose:
                    print()
                    lambda_list_1.info()
                self.add_maps(hp.read_map(lambda_list_1))

    @take_time
    def normalize(self):
        """
        Maps are normalized using histogram equalization to enhance contrast
        The resulting maps have uniform distribution of values between 0 an 1
        """
        normalized_maps = []
        for sky_map in self:
            normalized_maps.append(histogram_equalized(sky_map))
        self.clear()
        self.add_maps(*normalized_maps)

    @take_time
    def downscale(self, new_n_side):
        self.n_side = new_n_side
        self.n_pixels = hp.nside2npix(new_n_side)
        for i in range(len(self)):
            self[i] = hp.ud_grade(map_in=self[i], nside_out=new_n_side)

    @take_time
    def compute_global_correlation(self, i=0, j=1):
        """ Global correlation of the whole normalized maps of indices i and j """
        return correlation(self[i], self[j])

    @take_time
    def compute_custom_correlation_maps(self, proximity_matrix, i=0, j=1):
        return correlation_map(self[i], self[j], proximity_matrix)

    def get_local_matrix_path(self, angle_deg):
        make_dir('local')
        return f'local/prox_mat_{self.n_side}_{angle_deg:.0f}.npz'

    @take_time
    def _load_proximity_matrix(self, angle_deg):
        path = self.get_local_matrix_path(angle_deg)
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                self.proximity_matrix = load_npz(file)
                assert self.proximity_matrix.shape == (self.n_pixels, self.n_pixels)

    @take_time
    def _save_proximity_matrix(self, angle_deg):
        if self.verbose:
            print('Saving proximity matrix...')
        path = self.get_local_matrix_path(angle_deg)
        save_npz(path, self.proximity_matrix)

    @take_time
    def compute_local_correlation(self, angle_deg=11., i=0, j=1):
        """for each pixel, compute the correlation between the maps for pixels within a certain angle
        :param angle_deg: distance within which the correlation is considered
        :param i: first map index
        :param j: second map index
        """
        self._load_proximity_matrix(angle_deg)
        if self.proximity_matrix is None:
            angle_rad = np.deg2rad(angle_deg)
            rows, cols = compute_proximity_matrix_fast(self.n_side, angle_rad,
                                                       show_progress=self.verbose)
            data = [1 for _ in range(len(rows))]
            if self.verbose:
                print('Building proximity matrix...')
            self.proximity_matrix = csr_matrix((data, (rows, cols)),
                                               shape=(self.n_pixels, self.n_pixels),
                                               dtype=np.uint8)
            self._save_proximity_matrix(angle_deg)

        return self.compute_custom_correlation_maps(self.proximity_matrix, i=i, j=j)

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
        edges = spectral_segmentation_method(self[i], **kwargs)
        masks = []
        correlations = []

        for k in range(len(edges)-1):
            # find mask
            mask = get_temperature_range_mask(self[i], edges[k], edges[k+1])
            masks.append(mask)

            # compute correlation
            c = masked_correlation(self[i], self[j], mask)
            correlations.append(c)

        return {'edges': edges, 'masks': masks, 'correlations': correlations, 'n_bins': len(masks)}
