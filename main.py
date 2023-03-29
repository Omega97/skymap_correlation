import healpy as hp
import matplotlib.pyplot as plt
from analysis import SkyMapAnalyzer


PATHS = ['skymaps/haslam408_dsds_Remazeilles2014.fits',
         'skymaps/wmap_band_smth_iqumap_r9_9yr_K_v5.fits']


def main(new_n_side=32, angle_deg=5):
    """compute the sky-map that represents the local correlation of the two input maps"""
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    sky_maps.downscale(new_n_side=new_n_side)
    sky_maps.normalize()
    c = sky_maps.compute_local_correlation(angle_deg=angle_deg)
    title = f'Local correlation (n_side={sky_maps.n_side}, ang={angle_deg:.0f})'
    hp.mollview(c, title=title, cmap='jet_r', min=-1, max=+1)
    plt.show()


if __name__ == '__main__':
    main()
