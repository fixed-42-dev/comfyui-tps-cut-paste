"""Shared pieces for the colour-match proof of concept (not part of the node pack)."""
import numpy as np
import cv2

# ------------------------------------------------ colour spaces

def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def srgb_to_oklab(x):
    lms = srgb_to_linear(x) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_srgb(lab):
    lms = (lab @ np.linalg.inv(_M2).T) ** 3
    return linear_to_srgb(lms @ np.linalg.inv(_M1).T)


LUMA = np.array([0.2126, 0.7152, 0.0722])

# ------------------------------------------------ maps

def nconv(value, weight, sigma):
    """Normalized convolution: a weighted local mean that fills gaps from around them."""
    num = cv2.GaussianBlur((value * weight).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(weight.astype(np.float32), (0, 0), sigma)
    return num / np.maximum(den, 1e-6), den


def signed_distance(mask):
    """Distance from the mask edge: positive outside, negative inside (px)."""
    m = (mask > 0).astype(np.uint8)
    out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    ins = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return out - ins


def flatness(rgb, sigma=2.0):
    """Gradient magnitude of luma, lightly smoothed; small = flat."""
    y = (rgb @ LUMA).astype(np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigma)


def load_pair(results, prefix="fin_mona", rough="demo_mona_lisa_shoulder.png"):
    """The aligned sparrow pair from finish_test.py, in the edit's frame."""
    from PIL import Image
    O = cv2.imread(results + f"/{prefix}_orig_editframe.png")[..., ::-1].astype(np.float32) / 255
    E = np.load(results + f"/{prefix}_aligned.npy").astype(np.float32)
    change = np.load(results + f"/{prefix}_soft.npy").astype(np.float32)
    h, w = O.shape[:2]
    ell = np.asarray(Image.open(results + "/../" + rough))[..., 3] == 0
    rough = cv2.resize(ell.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return O, E, change, rough
