
from PIL import Image
import numpy as np
import torch
from torchvision.utils import save_image
def gaussian(sigma=6):
    """
    2D Gaussian Kernel Generation.
    """
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g
def center_map_gen(center_map, x, y, label, sigma, g):
    """
    Center map generation. point to heatmap.
    Arguments:
        center_map: A Tensor of shape [C, H, W].
        x: A Int type value. x-coordinate for the center point.
        y: A Int type value. y-coordinate for the center point.
        label: A Int type value. class for the center point.
        sigma: A Int type value. sigma for 2D gaussian kernel.
        g: A numpy array. predefined 2D gaussian kernel to be encoded in the center_map.

    Returns:
        A numpy array of shape [C, H, W]. center map in which points are encoded in 2D gaussian kernel.
    """

    channel, height, width = center_map.shape

    # outside image boundary
    if x < 0 or y < 0 or x >= width or y >= height:
        return center_map

    # upper left
    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
    # bottom right
    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

    c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
    a, b = max(0, -ul[1]), min(br[1], height) - ul[1]

    cc, dd = max(0, ul[0]), min(br[0], width)
    aa, bb = max(0, ul[1]), min(br[1], height)

    center_map[label, aa:bb, cc:dd] = np.maximum(center_map[label, aa:bb, cc:dd], g[a:b, c:d])

    return center_map