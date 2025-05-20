from __future__ import annotations

import io
import os
import pickle
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatuses
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from PIL import Image
from pyquaternion import Quaternion

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")




import torch
from torchvision import transforms

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
import warnings
import numpy as np

# -*- coding: utf-8 -*-

import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
from torchvision import transforms

from PIL import Image

# /////////////// Data Loader ///////////////


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, method, severity, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.method = method
        self.severity = severity
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            img = self.method(img, self.severity)
        if self.target_transform is not None:
            target = self.target_transform(target)

        save_path = '/share/data/vision-greg/DistortedImageNet/JPEG/' + self.method.__name__ + \
                    '/' + str(self.severity) + '/' + self.idx_to_class[target]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += path[path.rindex('/'):]

        Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)

        return 0  # we do not care about returning the data

    def __len__(self):
        return len(self.imgs)


warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):

    h, w = img.shape[:2]

    new_h = int(np.round(h / zoom_factor))
    new_w = int(np.round(w / zoom_factor))

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    img = scizoom(img[top:top+new_h, left:left+new_w], 
                (zoom_factor, zoom_factor, 1), 
                order=1)

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def fgsm(x, source_net, severity=1):
    c = [8, 16, 32, 64, 128][severity - 1]

    x = V(x, requires_grad=True)
    logits = source_net(x)
    source_net.zero_grad()
    loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
    loss.backward()

    return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def clipped_zoom(img, zoom_factor):
    h, w = img.shape[:2]
    
    new_h = int(np.round(h / zoom_factor))
    new_w = int(np.round(w / zoom_factor))
    
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    
    img = scizoom(img[top:top+new_h, left:left+new_w], 
                (zoom_factor, zoom_factor, 1), 
                order=1)
    
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    h, w = x.shape[:2]  

    out = np.zeros_like(x)
    
    for zoom_factor in c:
        zoomed = clipped_zoom(x, zoom_factor)
        out += zoomed

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


# def barrel(x, severity=1):
#     c = [(0,0.03,0.03), (0.05,0.05,0.05), (0.1,0.1,0.1),
#          (0.2,0.2,0.2), (0.1,0.3,0.6)][severity - 1]
#
#     output = BytesIO()
#     x.save(output, format='PNG')
#
#     x = WandImage(blob=output.getvalue())
#     x.distort('barrel', c)
#
#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#
#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    # 获取原始图像尺寸
    if isinstance(x, Image.Image):
        w, h = x.size
        x = np.array(x).astype(np.float32) / 255.
    else:
        h, w = x.shape[:2]
        x = x.astype(np.float32) / 255.

    # 生成雪层（适配任意尺寸）
    snow_layer = np.random.normal(size=(h, w), loc=c[0], scale=c[1])
    
    # 缩放雪层（保持宽高比）
    zoom_factor = c[2]
    snow_layer_zoomed = clipped_zoom(snow_layer[..., np.newaxis], zoom_factor)
    snow_layer = snow_layer_zoomed.squeeze()
    
    # 应用阈值
    snow_layer[snow_layer < c[3]] = 0

    # 创建运动模糊雪层
    snow_pil = Image.fromarray((np.clip(snow_layer, 0, 1) * 255).astype(np.uint8))
    with WandImage() as wand_img:
        wand_img.read(blob=snow_pil.tobytes())
        wand_img.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
        snow_layer = np.array(wand_img).astype(np.float32) / 255.

    # 适配不同通道情况
    if snow_layer.ndim == 2:
        snow_layer = snow_layer[..., np.newaxis]

    # 动态调整灰度转换
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    gray_reshaped = gray.reshape(h, w, 1)  # 使用动态尺寸

    # 合成最终图像
    blended = c[6] * x + (1 - c[6]) * np.maximum(x, gray_reshaped * 1.5 + 0.5)
    result = np.clip(blended + snow_layer + np.rot90(snow_layer, k=2), 0, 1)
    
    return (result * 255).astype(np.uint8)


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////


# /////////////// Further Setup ///////////////


def save_distorted(method=gaussian_noise):
    for severity in range(1, 6):
        print(method.__name__, severity)
        distorted_dataset = DistortImageFolder(
            root="/share/data/vision-greg/ImageNet/clsloc/images/val",
            method=method, severity=severity,
            transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224)]))
        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=100, shuffle=False, num_workers=4)

        for _ in distorted_dataset_loader: continue


# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
import collections

# print('\nUsing ImageNet data')

d = collections.OrderedDict()
    # d['Zoom Blur'] = zoom_blur   
    # d['JPEG'] = jpeg_compression
    # d['Pixelate'] = pixelate
    # d['Motion Blur'] = motion_blur
    # d['Defocus Blur'] = defocus_blur
    # d['Elastic'] = elastic_transform




# d['Gaussian Noise'] = gaussian_noise
# d['Shot Noise'] = shot_noise
# d['Impulse Noise'] = impulse_noise
# d['Brightness'] = brightness
# d['Contrast'] = contrast
# d['Speckle Noise'] = speckle_noise
# d['Spatter'] = spatter
# d['Saturate'] = saturate
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as trn

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

def save_distorted_images_from_numpy(
    input_image: np.ndarray,  # h w c
    methods: dict,
    severity: int = 4,
):

    img_pil = Image.fromarray(input_image.astype(np.uint8)).convert('RGB')
    for method_name, method in methods.items():
        distorted_img = method(img_pil, severity)
        to_pil = transforms.ToPILImage()
        if not isinstance(distorted_img, Image.Image):
            if isinstance(distorted_img, np.ndarray):
                distorted_img = Image.fromarray(distorted_img.astype(np.uint8))
            else:
                distorted_img = to_pil(distorted_img)
                    
        if distorted_img.size != img_pil.size:
            distorted_img = distorted_img.resize(img_pil.size, Image.LANCZOS)

        distorted_tensor = transforms.ToTensor()(distorted_img)

    return distorted_tensor

def camera_feature_improve(image): 
        dd = collections.OrderedDict()
        dd['Motion Blur'] = motion_blur      
        tensor_image_noise = save_distorted_images_from_numpy(image, dd)
        tensor_image = tensor_image_noise.detach().cpu()
        denormalized_image = tensor_image * 255.0
        denormalized_image = denormalized_image.byte()
        np_image_noise = denormalized_image.numpy()
        return np_image_noise


@dataclass
class Camera:
    """Camera dataclass for image and parameters."""

    image: Optional[npt.NDArray[np.float32]] = None

    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_translation: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float32]] = None
    distortion: Optional[npt.NDArray[np.float32]] = None

    camera_path: Optional[Path] = None


@dataclass
class Cameras:
    """Multi-camera dataclass."""

    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_camera_dict(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
    ) -> Cameras:
        """
        Load camera dataclass from dictionary.
        :param sensor_blobs_path: root directory of sensor data.
        :param camera_dict: dictionary containing camera specifications.
        :param sensor_names: list of camera identifiers to include.
        :return: Cameras dataclass.
        """

        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                # # oringin
                # data_dict[camera_identifier] = Camera(
                #     image=np.array(Image.open(image_path)),
                #     sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                #     sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                #     intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                #     distortion=camera_dict[camera_name]["distortion"],
                #     camera_path=camera_dict[camera_name]["data_path"],
                # )
                # #noise
                image=np.array(Image.open(image_path))
                # import random;
                # idx=random.randint(1,10)
                # if idx<5:
                    # image=camera_feature_improve(image).transpose(1, 2, 0)
                data_dict[camera_identifier] = Camera(
                    image=image,
                    sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                    sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                    intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                    distortion=camera_dict[camera_name]["distortion"],
                    camera_path=camera_dict[camera_name]["data_path"],
                )
            else:
                data_dict[camera_identifier] = Camera()  # empty camera

        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@dataclass
class Lidar:
    """Lidar point cloud dataclass."""

    # NOTE:
    # merged lidar point cloud as (6,n) float32 array with n points
    # first axis: (x, y, z, intensity, ring, lidar_id), see LidarIndex
    lidar_pc: Optional[npt.NDArray[np.float32]] = None
    lidar_path: Optional[Path] = None

    @staticmethod
    def _load_bytes(lidar_path: Path) -> BinaryIO:
        """Helper static method to load lidar point cloud stream."""
        with open(lidar_path, "rb") as fp:
            return io.BytesIO(fp.read())

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, lidar_path: Path, sensor_names: List[str]) -> Lidar:
        """
        Loads lidar point cloud dataclass in log loading.
        :param sensor_blobs_path: root directory to sensor data
        :param lidar_path: relative lidar path from logs.
        :param sensor_names: list of sensor identifiers to load`
        :return: lidar point cloud dataclass
        """

        # NOTE: this could be extended to load specific LiDARs in the merged pc
        if "lidar_pc" in sensor_names:
            global_lidar_path = sensor_blobs_path / lidar_path
            lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
            return Lidar(lidar_pc, lidar_path)
        return Lidar()  # empty lidar


@dataclass
class EgoStatus:
    """Ego vehicle status dataclass."""

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    in_global_frame: bool = False  # False for AgentInput


@dataclass
class AgentInput:
    """Dataclass for agent inputs with current and past ego statuses and sensors."""

    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [
                    ego_translation[0],
                    ego_translation[1],
                    ego_quaternion.yaw_pitch_roll[0],
                ],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]),
            np.array(global_ego_poses, dtype=np.float64),
        )

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []

        for frame_idx in range(num_history_frames):

            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            # # 修改变成一维
            # import math
            # vle=math.sqrt(ego_dynamic_state[0]**2 + ego_dynamic_state[1]**2)
            # acc=math.sqrt(ego_dynamic_state[2]**2 + ego_dynamic_state[3]**2)
            # tag_vle=1
            # tag_acc=1
            # if ego_dynamic_state[0]<0:
            #     tag_vle=-1
            # if ego_dynamic_state[2]<0:
            #     tag_acc=-1
            # ego_status = EgoStatus(
            #     ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
            #     ego_velocity=np.array(tag_vle*vle, dtype=np.float32),
            #     ego_acceleration=np.array(tag_acc*acc, dtype=np.float32),
            #     driving_command=scene_dict_list[frame_idx]["driving_command"],
            # )
            # print("<<<<<")
            # import pdb;pdb.set_trace()
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                    sensor_names=sensor_names,
                )
            )

        return AgentInput(ego_statuses, cameras, lidars)


@dataclass
class Annotations:
    """Dataclass of annotations (e.g. bounding boxes) per frame."""

    boxes: npt.NDArray[np.float32]
    names: List[str]
    velocity_3d: npt.NDArray[np.float32]
    instance_tokens: List[str]
    track_tokens: List[str]

    def __post_init__(self):
        annotation_lengths: Dict[str, int] = {
            attribute_name: len(attribute) for attribute_name, attribute in vars(self).items()
        }
        assert (
            len(set(annotation_lengths.values())) == 1
        ), f"Annotations expects all attributes to have equal length, but got {annotation_lengths}"


@dataclass
class Trajectory:
    """Trajectory dataclass in NAVSIM."""

    poses: npt.NDArray[np.float32]  # local coordinates
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    """Dataclass of scene metadata (e.g. location) per scene."""

    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int

    #  maps between synthetic scenes and the corresponding original scene
    #  with the same timestamp in the same log.
    #  NOTE: this is not the corresponding first stage scene token
    #  for original scenes this is None
    corresponding_original_scene: str = None

    # maps to the initial frame token (at 0.0s) of the corresponding original scene
    # for original scenes this is None
    corresponding_original_initial_token: str = None


@dataclass
class Frame:
    """Frame dataclass with privileged information."""

    token: str
    timestamp: int
    roadblock_ids: List[str]
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Lidar
    cameras: Cameras


@dataclass
class Scene:
    """Scene dataclass defining a single sample in NAVSIM."""

    # Ground truth information
    scene_metadata: SceneMetadata
    map_api: AbstractMap
    frames: List[Frame]
    extended_traffic_light_data: Optional[List[TrafficLightStatuses]] = None
    extended_detections_tracks: Optional[List[DetectionsTracks]] = None
    """
    scene_metadata (SceneMetadata): Metadata describing the scene, including its unique identifiers and attributes.
    map_api (AbstractMap): Map API interface providing access to map-related information such as lane geometry and topology.
    frames (List[Frame]): A sequence of frames describing the state of the ego-vehicle and its surroundings.
    extended_traffic_light_data (Optional[List[TrafficLightStatuses]], optional):
        A list containing traffic light status information for each future frame after the scene ends.
        Each `TrafficLightStatuses` entry includes a `TrafficLightStatusData` object for every lane connector
        controlled by a traffic light. Defaults to None.
    extended_detections_tracks (Optional[List[DetectionsTracks]], optional):
        A list containing detection tracks for each future frame after the scene ends.
        This can be used to provide future detections of pedestrians and objects in synthetic scenarios
        where future frames are unavailable. Defaults to None.
    """
    
    def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts future trajectory of the human operator in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """
        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_future_frames
        # if num_trajectory_frames is None:
        #     num_trajectory_frames = 8
        start_frame_idx = self.scene_metadata.num_history_frames - 1

        global_ego_poses = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_trajectory_frames + 1):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)
        # print(len(StateSE2(*global_ego_poses[0])))
        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]),
            np.array(global_ego_poses[1:], dtype=np.float64),
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_history_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts past trajectory of ego vehicles in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_history_frames

        global_ego_poses = []
        for frame_idx in range(num_trajectory_frames):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[-1])
        local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_agent_input(self) -> AgentInput:
        """
        Extracts agents input dataclass (without privileged information) from scene.
        :return: agent input dataclass
        """

        local_ego_poses = self.get_history_trajectory().poses
        ego_statuses: List[EgoStatus] = []
        cameras: List[Cameras] = []
        lidars: List[Lidar] = []

        for frame_idx in range(self.scene_metadata.num_history_frames):
            frame_ego_status = self.frames[frame_idx].ego_status

            ego_statuses.append(
                EgoStatus(
                    ego_pose=local_ego_poses[frame_idx],
                    ego_velocity=frame_ego_status.ego_velocity,
                    ego_acceleration=frame_ego_status.ego_acceleration,
                    driving_command=frame_ego_status.driving_command,
                )
            )
            cameras.append(self.frames[frame_idx].cameras)
            lidars.append(self.frames[frame_idx].lidar)

        return AgentInput(ego_statuses, cameras, lidars)

    @classmethod
    def _build_map_api(cls, map_name: str) -> AbstractMap:
        """Helper classmethod to load map api from name."""
        assert map_name in MAP_LOCATIONS, f"The map name {map_name} is invalid, must be in {MAP_LOCATIONS}"
        return get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)

    @classmethod
    def _build_annotations(cls, scene_frame: Dict) -> Annotations:
        """Helper classmethod to load annotation dataclass from logs."""
        return Annotations(
            boxes=scene_frame["anns"]["gt_boxes"],
            names=scene_frame["anns"]["gt_names"],
            velocity_3d=scene_frame["anns"]["gt_velocity_3d"],
            instance_tokens=scene_frame["anns"]["instance_tokens"],
            track_tokens=scene_frame["anns"]["track_tokens"],
        )

    @classmethod
    def _build_ego_status(cls, scene_frame: Dict) -> EgoStatus:
        """Helper classmethod to load ego status dataclass from logs."""
        ego_translation = scene_frame["ego2global_translation"]
        ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )
        ego_dynamic_state = scene_frame["ego_dynamic_state"]
        return EgoStatus(
            ego_pose=global_ego_pose,
            ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
            ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
            driving_command=scene_frame["driving_command"],
            in_global_frame=True,
        )

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
    ) -> Scene:
        """
        Load scene dataclass from scene dictionary list (for log loading).
        :param scene_dict_list: list of scene frames (in logs)
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of past and current frames to load
        :param num_future_frames: number of future frames to load
        :param sensor_config: sensor config dataclass
        :return: scene dataclass
        """
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        map_api = cls._build_map_api(scene_metadata.map_name)

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = cls._build_ego_status(scene_dict_list[frame_idx])
            annotations = cls._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)

            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=scene_dict_list[frame_idx]["cams"],
                sensor_names=sensor_names,
            )

            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                sensor_names=sensor_names,
            )

            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
            )
            frames.append(frame)

        return Scene(scene_metadata=scene_metadata, map_api=map_api, frames=frames)

    def save_to_disk(self, data_path: Path):
        """
        Save scene dataclass to disk.
        Note: this will NOT save the images or point clouds.
        :param data_path: root directory to save scene data
        :param sensor_blobs_path: root directory to sensor data
        """

        assert self.scene_metadata.scene_token is not None, "Scene token cannot be 'None', when saving to disk."
        assert data_path.is_dir(), f"Data path {data_path} is not a directory."

        # collect all the relevant data for the frames
        frames_data = []
        for frame in self.frames:
            camera_dict = {}
            for camera_field in fields(frame.cameras):
                camera_name = camera_field.name
                camera: Camera = getattr(frame.cameras, camera_name)
                if camera.image is not None:
                    camera_dict[camera_name] = {
                        "data_path": camera.camera_path,
                        "sensor2lidar_rotation": camera.sensor2lidar_rotation,
                        "sensor2lidar_translation": camera.sensor2lidar_translation,
                        "cam_intrinsic": camera.intrinsics,
                        "distortion": camera.distortion,
                    }
                else:
                    camera_dict[camera_name] = {}

            if frame.lidar.lidar_pc is not None:
                lidar_path = frame.lidar.lidar_path
            else:
                lidar_path = None

            frames_data.append(
                {
                    "token": frame.token,
                    "timestamp": frame.timestamp,
                    "roadblock_ids": frame.roadblock_ids,
                    "traffic_lights": frame.traffic_lights,
                    "annotations": asdict(frame.annotations),
                    "ego_status": asdict(frame.ego_status),
                    "lidar_path": lidar_path,
                    "camera_dict": camera_dict,
                }
            )

        # collect all the relevant data for the scene
        scene_dict = {
            "scene_metadata": asdict(self.scene_metadata),
            "frames": frames_data,
            "extended_traffic_light_data": self.extended_traffic_light_data,
            "extended_detections_tracks": self.extended_detections_tracks,
        }

        # save the scene_dict to disk
        save_path = data_path / f"{self.scene_metadata.scene_token}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(scene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_disk(
        cls,
        file_path: Path,
        sensor_blobs_path: Path,
        sensor_config: SensorConfig = None,
    ) -> Scene:
        """
        Load scene dataclass from disk. Only used for synthesized views.
        Regular scenes are loaded from logs.
        :return: scene dataclass
        """
        if sensor_config is None:
            sensor_config = SensorConfig.build_no_sensors()
        # Load the metadata
        with open(file_path, "rb") as f:
            scene_data = pickle.load(f)

        scene_metadata = SceneMetadata(**scene_data["scene_metadata"])
        # build the map from the map_path
        map_api = cls._build_map_api(scene_metadata.map_name)

        scene_frames: List[Frame] = []
        for frame_idx, frame_data in enumerate(scene_data["frames"]):
            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            lidar_path = Path(frame_data["lidar_path"]) if frame_data["lidar_path"] else None
            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=lidar_path,
                sensor_names=sensor_names,
            )
            # 
            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=frame_data["camera_dict"],
                sensor_names=sensor_names,
            )

            scene_frames.append(
                Frame(
                    token=frame_data["token"],
                    timestamp=frame_data["timestamp"],
                    roadblock_ids=frame_data["roadblock_ids"],
                    traffic_lights=frame_data["traffic_lights"],
                    annotations=Annotations(**frame_data["annotations"]),
                    ego_status=EgoStatus(**frame_data["ego_status"]),
                    lidar=lidar,
                    cameras=cameras,
                )
            )

        return Scene(
            scene_metadata=scene_metadata,
            map_api=map_api,
            frames=scene_frames,
            extended_traffic_light_data=scene_data["extended_traffic_light_data"],
            extended_detections_tracks=scene_data["extended_detections_tracks"],
        )


@dataclass
class SceneFilter:
    """Scene filtering configuration for scene loading."""

    num_history_frames: int = 4
    num_future_frames: int = 10
    frame_interval: Optional[int] = None
    has_route: bool = True

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    include_synthetic_scenes: bool = False
    synthetic_scene_tokens: Optional[List[str]] = None
    # TODO: expand filter options
    reactive_synthetic_initial_tokens: Optional[List[str]] = None
    non_reactive_synthetic_initial_tokens: Optional[List[str]] = None
    def __post_init__(self):

        if self.frame_interval is None:
            self.frame_interval = self.num_frames

        assert self.num_history_frames >= 1, "SceneFilter: num_history_frames must greater equal one."
        assert self.num_future_frames >= 0, "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."

        if (
            not self.include_synthetic_scenes
            and self.synthetic_scene_tokens is not None
            and len(self.synthetic_scene_tokens) > 0
        ):
            warnings.warn(
                "SceneFilter: synthetic_scene_tokens are provided but include_synthetic_scenes is False. No synthetic scenes will be loaded."
            )

    @property
    def num_frames(self) -> int:
        """
        :return: total number for frames for scenes to extract.
        """
        return self.num_history_frames + self.num_future_frames


@dataclass
class SensorConfig:
    """Configuration dataclass of agent sensors for memory management."""

    # Config values of sensors are either
    # - bool: Whether to load history or not
    # - List[int]: For loading specific history steps
    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]

    def get_sensors_at_iteration(self, iteration: int) -> List[str]:
        """
        Creates a list of sensor identifiers given iteration.
        :param iteration: integer indicating the history iteration.
        :return: list of sensor identifiers to load.
        """
        sensors_at_iteration: List[str] = []
        for sensor_name, sensor_include in asdict(self).items():
            if isinstance(sensor_include, bool) and sensor_include:
                sensors_at_iteration.append(sensor_name)
            elif isinstance(sensor_include, list) and iteration in sensor_include:
                sensors_at_iteration.append(sensor_name)
        return sensors_at_iteration

    @classmethod
    def build_all_sensors(cls, include: Union[bool, List[int]] = True) -> SensorConfig:
        """
        Classmethod to load all sensors with the same specification.
        :param include: boolean or integers for sensors to include, defaults to True
        :return: sensor configuration dataclass
        """
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=include,
            cam_l2=include,
            cam_r0=include,
            cam_r1=include,
            cam_r2=include,
            cam_b0=include,
            lidar_pc=False, #include 
        )

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        """
        Classmethod to load no sensors.
        :return: sensor configuration dataclass
        """
        return cls.build_all_sensors(include=False)


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float
    driving_direction_compliance: float
    traffic_light_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    lane_keeping: float
    history_comfort: float

    multiplicative_metrics_prod: float
    weighted_metrics: npt.NDArray[np.float64]
    weighted_metrics_array: npt.NDArray[np.float64]

    pdm_score: float

    @classmethod
    def get_empty_results(cls) -> PDMResults:
        """
        Returns an instance of the class where all values are NaN.
        :return: empty PDM results dataclass.
        """
        return PDMResults(
            no_at_fault_collisions=np.nan,
            drivable_area_compliance=np.nan,
            driving_direction_compliance=np.nan,
            traffic_light_compliance=np.nan,
            ego_progress=np.nan,
            time_to_collision_within_bound=np.nan,
            lane_keeping=np.nan,
            history_comfort=np.nan,
            multiplicative_metrics_prod=np.nan,
            weighted_metrics=np.nan,
            weighted_metrics_array=np.nan,
            pdm_score=np.nan,
        )
