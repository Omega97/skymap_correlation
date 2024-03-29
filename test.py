import healpy as hp
import matplotlib.pyplot as plt
from astropy.stats import bayesian_blocks
from analysis import SkyMapAnalyzer
import numpy as np
from utils import compute_proximity_masks, masked_correlation, get_temperature_range_mask
import sys


PATHS = ['skymaps/haslam408_dsds_Remazeilles2014.fits',
         'skymaps/wmap_band_smth_iqumap_r9_9yr_K_v5.fits']


def test_loading(new_n_side=32):
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    print('\n nside:', ' '.join(f'{hp.npix2nside(len(m))}' for m in sky_maps))
    sky_maps.downscale(new_n_side=new_n_side)
    sky_maps.normalize()
    sky_maps.plot()
    plt.show()


def test_global_correlation(new_n_side=None):
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    if new_n_side:
        sky_maps.downscale(new_n_side=new_n_side)
    sky_maps.normalize()
    cor = sky_maps.compute_global_correlation()
    print(f'\n corr = {cor:.2%}')


def test_proximity(rows=3, cols=3, n_side=12, angle_rad=.3):
    np.random.seed(0)
    v = compute_proximity_masks(n_side=n_side, angle_rad=angle_rad)
    print('indices:')
    for i in range(rows):
        for j in range(cols):
            k = np.random.randint(0, hp.nside2npix(n_side))
            sky_map = v[k]
            n = i*cols+j+1
            print(sky_map.get_indices())
            hp.mollview(sky_map, title=f'{k}', cmap='bone', sub=(rows, cols, n))

    plt.show()


def test_masked_correlation(n_side=16):
    """correlation between maps in the specified areas"""
    np.random.seed(1)

    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    sky_maps.downscale(new_n_side=n_side)
    sky_maps.normalize()

    hp.mollview(sky_maps[0], title='mask A', cmap='jet', sub=(2, 2, 1))
    hp.mollview(sky_maps[1], title='mask B', cmap='jet', sub=(2, 2, 2))

    masks = compute_proximity_masks(n_side, angle_rad=.3)

    for j in range(2):
        i = np.random.randint(0, hp.nside2npix(n_side))
        mask = masks[i]
        c = masked_correlation(sky_maps[0], sky_maps[1], mask)
        hp.mollview(mask, title=f'corr = {c:.2%}', cmap='bone', sub=(2, 2, 3 + j))
    plt.show()


def test_local_correlation(new_n_side=8, angle_deg=20):
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    sky_maps.downscale(new_n_side=new_n_side)
    sky_maps.normalize()
    c = sky_maps.compute_local_correlation(angle_deg=angle_deg)
    title = f'Local correlation (n_side={sky_maps.n_side}, ang={angle_deg:.0f})'
    hp.mollview(c, title=title, cmap='jet_r', min=-1, max=+1)
    plt.show()
    print('All done!')


def test_local_correlation_random(n_side=16, angle_deg=10):
    npix = hp.nside2npix(n_side)
    sky_maps = SkyMapAnalyzer()
    sky_maps.append(np.random.random(npix))
    sky_maps.append(np.random.random(npix))
    sky_maps.normalize()
    c = sky_maps.compute_local_correlation(angle_deg=angle_deg)
    title = f'Local correlation (n_side={sky_maps.n_side}, ang={angle_deg:.0f})'
    hp.mollview(c, title=title, cmap='bwr_r', min=-1, max=+1)
    plt.show()


def partition_brightness_spectrum(sky_map, verbose=False):
    """separate image into masks spanning over regions on certain intensity"""
    edges = bayesian_blocks(sky_map, p0=10 ** -7)

    if verbose:
        print(', '.join([f'{x:.2f}' for x in edges]))

    plt.hist(sky_map, bins=list(edges), density=True)
    plt.title(f'Bayesian Blocks 408 Mhz spectrum')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.show()


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


def test_temperature_separation(new_n_side=16, ncols=6):
    """ The first map is separated in temperature bands, then the correlation
    between the two maps is calculated in those regions """
    analysis = SkyMapAnalyzer(*PATHS, verbose=True)
    analysis.downscale(new_n_side=new_n_side)
    results = analysis.compute_temperature_correlation(bayesian_blocks, i=0, j=1, p0=10**-7)
    analysis.normalize()

    n = results['n_bins']
    nrows = np.ceil(2 * n / ncols)

    for i in range(n):
        x_min, x_max = results['edges'][i:i+2]

        map_a = (1 + analysis[0]) * results['masks'][i] - 1
        map_b = (1 + analysis[1]) * results['masks'][i] - 1

        title = f'({x_min:.1f} -- {x_max:.1f})'
        hp.mollview(map_a, title=title, cmap='hot', min=-.5, max=1, cbar=False, sub=(nrows, ncols, 2 * i + 1))
        title = f'corr={results["correlations"][i]:.1%}'
        hp.mollview(map_b, title=title, cmap='hot', min=-.5, max=1, cbar=False, sub=(nrows, ncols, 2 * i + 2))

    plt.show()


def test_local_correlation_sys():
    """run from console with arguments new_n_side, angle_deg"""
    test_local_correlation(*[int(i) for i in sys.argv[1:3]])


if __name__ == '__main__':
    # test_loading()
    # test_global_correlation(new_n_side=16)
    # test_proximity()
    # test_masked_correlation()

    test_local_correlation()
    # test_local_correlation_random()
    # test_temperature_separation()

    # test_local_correlation_sys()
