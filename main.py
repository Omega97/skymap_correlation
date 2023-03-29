import healpy as hp
import matplotlib.pyplot as plt
from analysis import SkyMapAnalyzer
from utils import Mask


PATHS = ['skymaps/haslam408_dsds_Remazeilles2014.fits',
         'skymaps/wmap_band_smth_iqumap_r9_9yr_K_v5.fits']


def main(new_n_side=32, rows=4, cols=5, angle_deg=11, k_alpha=.3, n_samples=150):
    sky_maps = SkyMapAnalyzer(*PATHS, verbose=True)
    sky_maps.downscale(new_n_side=new_n_side)
    sky_maps.normalize()

    npix = hp.nside2npix(new_n_side)
    n_mod = int(npix / n_samples)
    sample_mask = Mask([i for i in range(npix) if (i+1) % n_mod == 0], npix)
    print(len(sample_mask.get_indices()))

    corr_map = sky_maps.compute_local_correlation(angle_deg=angle_deg)
    v = [(i, corr_map[i]) for i in sample_mask.get_indices()]

    v.sort(key=lambda a: a[1], reverse=True)
    v = v[:(rows-1)*cols]

    hp.mollview(sky_maps[0], cmap='jet', title='map A', sub=(rows, cols, 1))
    hp.mollview(sky_maps[1], cmap='jet', title='map B', sub=(rows, cols, 2))
    hp.mollview(corr_map, title=f'Local correlation ang={angle_deg:.0f}°',
                cmap='bwr_r', min=-1, max=+1, sub=(rows, cols, 3))
    alpha = sample_mask + corr_map * k_alpha
    hp.mollview(corr_map * (1-sample_mask) + sample_mask, title='samples', cmap='bwr_r', min=-1, max=1,
                alpha=alpha, cbar=False, sub=(rows, cols, 4))

    for i in range(len(v)):
        j, c = v[i]
        mask = sky_maps.proximity_masks[j]
        sub = (rows, cols, 1+cols+i)
        alpha = mask + (1-mask) * k_alpha
        hp.mollview(sky_maps[0], cmap='jet', alpha=alpha, sub=sub, cbar=False, title=f'corr = {c:.1%}')

    plt.show()


if __name__ == '__main__':
    main()
