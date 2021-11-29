"""Image transformations for augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import random
from typing import Callable, Dict, Tuple

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import torchvision.transforms.functional as F
from PIL.Image import Image

import albumentations

from src.utils.data import get_rand_bbox_coord

FILLCOLOR = (128, 128, 128)
FILLCOLOR_RGBA = (128, 128, 128, 128)


def transforms_info() -> Dict[
    str, Tuple[Callable[[Image, float], Image], float, float]
]:
    """Return augmentation functions and their ranges."""
    transforms_list = [
        (Identity, 0.0, 0.0),
        (Invert, 0.0, 0.0),
        (Contrast, 0.0, 0.9),
        (AutoContrast, 0.0, 0.0),
        (Rotate, 0.0, 30.0),
        (TranslateX, 0.0, 150 / 331),
        (TranslateY, 0.0, 150 / 331),
        (Sharpness, 0.0, 0.9),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (Color, 0.0, 0.9),
        (Brightness, 0.0, 0.9),
        (Equalize, 0.0, 0.0),
        (Solarize, 256.0, 0.0),
        (Posterize, 8, 4),
        (Cutout, 0, 0.5),
        (RandomBrightnessContrast, 0.0, 0.0),
    ]
    return {f.__name__: (f, low, high) for f, low, high in transforms_list}


def Identity(img: Image, _: float) -> Image:
    """Identity map."""
    return img


def Invert(img: Image, _: float) -> Image:
    """Invert the image."""
    imgArray = np.array(img)
    aug = albumentations.InvertImg()
    imgArray = aug.apply(imgArray)
    img = PIL.Image.fromarray(imgArray)
    return img


def Contrast(img: Image, magnitude: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageEnhance.Contrast(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def RandomBrightnessContrast(img: Image, _: float) -> Image:
    imgArray = np.array(img)
    aug = albumentations.RandomBrightnessContrast()
    imgArray = aug.apply(imgArray)
    img = PIL.Image.fromarray(imgArray)
    return img


def AutoContrast(img: Image, _: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageOps.autocontrast(img)


def Rotate(img: Image, magnitude: float) -> Image:
    """Rotate the image (degree)."""
    imgArray = np.array(img)
    aug = albumentations.Rotate()
    imgArray = aug.apply(imgArray, magnitude)
    img = PIL.Image.fromarray(imgArray)
    return img
    

def TranslateX(img: Image, magnitude: float) -> Image:
    """Translate the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
        fillcolor=FILLCOLOR,
    )


def TranslateY(img: Image, magnitude: float) -> Image:
    """Translate the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
        fillcolor=FILLCOLOR,
    )


def Sharpness(img: Image, magnitude: float) -> Image:
    """Adjust the sharpness of the image."""
    imgArray = np.array(img)
    aug = albumentations.Sharpen()
    imgArray = aug.apply(imgArray, magnitude)
    img = PIL.Image.fromarray(imgArray)
    return img


def ShearX(img: Image, magnitude: float) -> Image:
    """Shear the image on x-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def ShearY(img: Image, magnitude: float) -> Image:
    """Shear the image on y-axis."""
    return img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )


def Color(img: Image, magnitude: float) -> Image:
    """Adjust the color balance of the image."""
    return PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def Brightness(img: Image, magnitude: float) -> Image:
    """Adjust brightness of the image."""
    return PIL.ImageEnhance.Brightness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    )


def Equalize(img: Image, _: float) -> Image:
    """Equalize the image."""
    imgArray = np.array(img)
    aug = albumentations.Equalize()
    imgArray = aug.apply(imgArray)
    img = PIL.Image.fromarray(imgArray)
    return img


def Solarize(img: Image, magnitude: float) -> Image:
    """Solarize the image."""

    imgArray = np.array(img)
    aug = albumentations.Solarize(magnitude)
    imgArray = aug.apply(imgArray)
    img = PIL.Image.fromarray(imgArray)
    return img


def Posterize(img: Image, magnitude: float) -> Image:
    """Posterize the image."""
    bits = int(magnitude)
    mask = ~(2 ** (8 - bits) - 1)

    imgArray = np.array(img)
    aug = albumentations.Posterize(num_bits = mask)
    imgArray = aug.apply(imgArray)
    img = PIL.Image.fromarray(imgArray)
    return img


def Cutout(img: Image, magnitude: float) -> Image:
    """Cutout some region of the image."""
    if magnitude == 0.0:
        return img
    w, h = img.size
    xy = get_rand_bbox_coord(w, h, magnitude)

    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, fill=FILLCOLOR)
    return img


class SquarePad:
    """Square pad to make torch resize to keep aspect ratio."""

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")
