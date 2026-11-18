"""The color barcode: a stretch of the video as one strip of color.

The chosen segment is sampled at evenly spaced moments. Each moment becomes
one stripe, the average color of that frame. Side by side, from the start of
the segment to its end, the stripes show how the colors and brightness change
over time.
"""
import numpy as np

from blackboard.analyze import LUMA_B, LUMA_G, LUMA_R, average_color
from blackboard.decode import DecodeError, read_frames

# How many stripes a barcode has (fewer for a segment with fewer frames)
BARCODE_COLUMNS = 400

# The progress callback is called after every this many stripes
UPDATE_EVERY = 10


def segment_length(first, last):
    """How many frames the segment first..last holds (both included)."""
    if first < 0:
        raise ValueError("Frame numbers must not be negative.")
    if last < first:
        raise ValueError("The segment end must not be before its start.")
    return last - first + 1


def barcode_columns(first, last, columns=BARCODE_COLUMNS):
    """How many stripes the barcode of this segment will have."""
    length = segment_length(first, last)
    if columns < 1:
        raise ValueError("A barcode needs at least one stripe.")
    return min(columns, length)


def sample_indices(first, last, columns=BARCODE_COLUMNS):
    """The frame numbers to sample: the middle of each of equal slices.

    Stripe j stands for the j-th slice of the segment, and the frame in the
    middle of that slice is the one measured.
    """
    length = segment_length(first, last)
    count = barcode_columns(first, last, columns)
    return [first + int((slot + 0.5) * length / count) for slot in range(count)]


def build_barcode(path, first, last, columns=BARCODE_COLUMNS,
                  on_update=None, should_stop=None):
    """Measure the stripes of the barcode of frames first..last.

    Returns an array with one (blue, green, red) row per stripe, in order.
    on_update(colors_so_far) is called every few stripes, so a window can fill
    in as the work goes on. should_stop() is checked after every stripe, and
    when it returns True the stripes measured so far are returned.

    If frames can't be read part of the way through (some videos report more
    frames than they hold), the stripes measured so far are returned. Only if
    not even one stripe could be measured is the DecodeError raised.
    """
    indices = sample_indices(first, last, columns)
    colors = []
    try:
        for frame in read_frames(path, indices):
            colors.append(average_color(frame.image))
            if on_update is not None and len(colors) % UPDATE_EVERY == 0:
                on_update(np.array(colors, np.uint8))
            if should_stop is not None and should_stop():
                break
    except DecodeError:
        if not colors:
            raise
    return np.array(colors, np.uint8).reshape(-1, 3)


def brightness_colors(colors):
    """The brightness barcode: the same stripes as gray.

    Each stripe becomes the gray with the brightness of that stripe's color
    (black for dark, white for bright), using the same weights as the
    brightness measurement. The average of the brightness of the pixels
    equals the brightness of their average color, so no frame is read again.
    Returns an (n x 3) array with equal blue, green and red values.
    """
    colors = np.asarray(colors, np.float64).reshape(-1, 3)
    level = colors @ np.array([LUMA_B, LUMA_G, LUMA_R])
    gray = np.clip(np.rint(level), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, None], 3, axis=1)