import healpy as hp
import matplotlib.pyplot as plt
from analysis import SkyMapAnalyzer
import numpy as np
from utils import compute_proximity_masks, masked_correlation


PATHS = ['skymaps/haslam408_dsds_Remazeilles2014.fits',
         'skymaps/wmap_band_smth_iqumap_r9_9yr_K_v5.fits']


def test_loading():
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    sky_maps.downscale(new_n_side=64)
    sky_maps.normalize()

    for i, sky_map in enumerate(sky_maps):
        hp.mollview(sky_map, cmap='jet', title=f'map {i}', sub=(1, len(PATHS), i+1))
    plt.show()


def test_global_correlation():
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    sky_maps.downscale(new_n_side=32)
    sky_maps.normalize()

    cor = sky_maps.compute_global_correlation()
    print(f'\n corr = {cor:.2%}')


def test_proximity(rows=3, cols=3, n_side=12, angle_rad=.3):
    np.random.seed(0)
    v = compute_proximity_masks(n_side=n_side, angle_rad=angle_rad)
    for i in range(rows):
        for j in range(cols):
            k = np.random.randint(0, hp.nside2npix(n_side))
            sky_map = v[k]
            n = i*cols+j+1
            print(sky_map.indices)
            hp.mollview(sky_map, title=f'{k}', cmap='bone', sub=(rows, cols, n))

    plt.show()


def test_masked_correlation(n_side=8):
    np.random.seed(1)

    analysis = SkyMapAnalyzer(*PATHS, verbose=True)
    analysis.downscale(new_n_side=n_side)
    analysis.normalize()

    hp.mollview(analysis.maps[0], title='mask A', cmap='jet', sub=(2, 2, 1))
    hp.mollview(analysis.maps[1], title='mask B', cmap='jet', sub=(2, 2, 2))

    masks = compute_proximity_masks(n_side, angle_rad=.3)

    for j in range(2):
        i = np.random.randint(0, hp.nside2npix(n_side))
        mask = masks[i]
        c = masked_correlation(analysis.maps[0], analysis.maps[1], mask)
        hp.mollview(mask, title=f'corr = {c:.2%}', cmap='bone', sub=(2, 2, 3 + j))
    plt.show()


def test_local_correlation(new_n_side=16):
    analysis = SkyMapAnalyzer(*PATHS, verbose=True)
    analysis.downscale(new_n_side=new_n_side)
    analysis.normalize()

    c = analysis.compute_local_correlation(angle_rad=.2)
    hp.mollview(c, title='Local correlation', cmap='bwr_r', min=-1, max=+1)
    plt.show()


if __name__ == '__main__':
    # test_loading()
    # test_global_correlation()
    # test_proximity()
    # test_masked_correlation()
    test_local_correlation()
