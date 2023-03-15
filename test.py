import healpy as hp
import matplotlib.pyplot as plt

try:
    from analysis import SkyMapAnalysis
    from utils import correlation, get_temperature_range_mask
    from misc import histogram_equalized
except ImportError:
    from .analysis import SkyMapAnalysis
    from .utils import correlation, get_temperature_range_mask
    from .misc import histogram_equalized


def test_temperature_partition(map1, map2, edges):
    fig = plt.figure(figsize=(24, 4))

    for i in range(len(edges)-1):
        indices = [j for j, x in enumerate(map1) if edges[i] <= x <= edges[i+1]]

        corr = correlation(map1[indices], map2[indices])

        mask = get_temperature_range_mask(map1, edges[i], edges[i+1])
        hp.mollview(histogram_equalized(map1) * mask,
                    title=f'range = ({edges[i]:.2f}, {edges[i+1]:.2f})',
                    cbar=False, bgcolor='0.1', cmap='jet',
                    sub=(2, len(edges)-1, 1+i))
        hp.mollview(histogram_equalized(map2) * mask,
                    title=f'corr = {corr:.1%}',
                    cbar=False, bgcolor='0.1', cmap='jet',
                    sub=(2, len(edges)-1, i+len(edges)))


def main():
    file_path_1 = ...
    file_path_2 = ...
    analysis = SkyMapAnalysis(file_path_1, file_path_2)
    analysis.downscale(nside=32)


if __name__ == '__main__':
    main()
