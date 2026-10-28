"""Stage 3: measurements taken from a decoded frame.

Every function here takes a plain image array (no files, no windows), so it
can be tested with tiny made-up images. Frames from decode.py are BGR arrays
of shape (height, width, 3) with values from 0 to 255.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Frames wider than this are shrunk before measuring. That makes analysis much
# faster and barely changes the averages.
ANALYSIS_WIDTH = 640

# How much each color adds to the brightness the eye sees, using the standard
# for HD video (Rec. 709). Green counts the most and blue the least.
LUMA_B, LUMA_G, LUMA_R = 0.0722, 0.7152, 0.2126


def _check_color_image(image):
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty color image (height x width x 3).")


def prepare(image):
    """Shrink a frame to ANALYSIS_WIDTH pixels wide. Never enlarges it."""
    _check_color_image(image)
    height, width = image.shape[:2]
    if width <= ANALYSIS_WIDTH:
        return image
    new_height = max(1, round(height * ANALYSIS_WIDTH / width))
    return cv2.resize(image, (ANALYSIS_WIDTH, new_height),
                      interpolation=cv2.INTER_AREA)


def luma(image):
    """The brightness of every pixel, from 0 (black) to 255 (white).

    Returns a 2-D uint8 array. Most measurements are built on this.
    """
    _check_color_image(image)
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    values = LUMA_B * blue + LUMA_G * green + LUMA_R * red
    return np.rint(values).astype(np.uint8)


def brightness_percent(gray):
    """Average brightness of a luma array: 0 is black, 100 is white."""
    if gray.size == 0:
        raise ValueError("Cannot measure an empty image.")
    return float(gray.mean()) / 255 * 100


# Contrast compares the darkest and brightest of the picture, but ignores the
# most extreme 5% at each end so a few stray pixels (a tiny logo, a speck of
# noise) can't change the answer.
CONTRAST_LOW_PERCENTILE = 5
CONTRAST_HIGH_PERCENTILE = 95


def contrast_percent(gray):
    """How far apart the dark and bright parts of a luma array are.

    0 means flat: nearly every pixel has the same brightness. 100 means the
    dark end reaches black and the bright end reaches white.
    """
    if gray.size == 0:
        raise ValueError("Cannot measure an empty image.")
    low, high = np.percentile(
        gray, [CONTRAST_LOW_PERCENTILE, CONTRAST_HIGH_PERCENTILE])
    return float(high - low) / 255 * 100


@dataclass
class FrameStats:
    brightness: float   # percent, 0 (black) to 100 (white)
    contrast: float     # percent, 0 (flat) to 100 (black to white)


def analyze_frame(image):
    """Take every measurement we have for one frame (a BGR array)."""
    small = prepare(image)
    gray = luma(small)
    return FrameStats(
        brightness=brightness_percent(gray),
        contrast=contrast_percent(gray),
    )