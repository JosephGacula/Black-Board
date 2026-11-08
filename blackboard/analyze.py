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


# A pixel counts as clipped when it is pure black or white, or very close.
# Video compression rarely leaves a "black" pixel at exactly 0, so we allow a
# small margin: 5 out of 255 is about 2% and 250 out of 255 is about 98%.
CLIP_BLACK_MAX = 5
CLIP_WHITE_MIN = 250


def clipping_masks(gray):
    """Mark the clipped pixels of a luma array.

    Returns two True/False arrays the same shape as the image: one that is
    True where a pixel is crushed to black, and one where it is blown out to
    white. (The exposure overlay will draw these on the preview.)
    """
    return gray <= CLIP_BLACK_MAX, gray >= CLIP_WHITE_MIN


def clipping_percent(gray):
    """The share of pixels that are clipped, as (crushed_blacks, blown_highlights).

    Each is a percentage of all the pixels, from 0 to 100.
    """
    if gray.size == 0:
        raise ValueError("Cannot measure an empty image.")
    black, white = clipping_masks(gray)
    return float(black.mean()) * 100, float(white.mean()) * 100


# The brightness range 0-255 is split into thirds for the dark / mid / bright
# summary: 0 to 85 is dark, 86 to 170 is mid, 171 to 255 is bright.
ZONE_DARK_MAX = 85
ZONE_MID_MAX = 170


def luminance_histogram(gray):
    """Count how many pixels have each brightness from 0 to 255.

    Returns an array of 256 counts: counts[0] is the number of pure black
    pixels, counts[255] the number of pure white ones, and so on.
    """
    if gray.size == 0:
        raise ValueError("Cannot measure an empty image.")
    return np.bincount(gray.ravel(), minlength=256)


def combine_histograms(histograms):
    """Add several histograms (for example one per frame) into one.

    The result describes all of those frames together, as if they were one
    big picture.
    """
    if len(histograms) == 0:
        raise ValueError("Nothing to combine.")
    return np.sum(histograms, axis=0)


def tonal_zones_percent(histogram):
    """Split a histogram into thirds: (dark, mid, bright) as percentages."""
    total = histogram.sum()
    if total == 0:
        raise ValueError("The histogram is empty.")
    dark = histogram[:ZONE_DARK_MAX + 1].sum()
    mid = histogram[ZONE_DARK_MAX + 1:ZONE_MID_MAX + 1].sum()
    bright = histogram[ZONE_MID_MAX + 1:].sum()
    return tuple(float(part) / total * 100 for part in (dark, mid, bright))


# ---------- color: saturation and hue ----------
#
# Color is measured in HSV: hue is which color it is (red, green, blue...),
# saturation is how strong it is (0% is gray, 100% is fully vivid) and value
# is how bright it is. OpenCV stores hue from 0 to 179 (every step is 2
# degrees), and saturation and value from 0 to 255.

# Pixels darker than this (out of 255, about 6%) are left out of the color
# measurements. In a very dark pixel the color is mostly camera noise, and
# the HSV math would call it strongly saturated even though nobody can see it.
COLOR_MIN_VALUE = 16

# For the dominant hue, only pixels with a clear color get a vote: at least
# this saturated (out of 255, about 20%). Grays and near-whites have no hue.
HUE_MIN_SATURATION = 51

# If fewer than this share of all pixels has a clear color, the picture is
# treated as neutral and no dominant hue is reported.
MIN_COLORED_SHARE = 0.02

# Hue is a circle, so the peak is found after smoothing: each hue is scored by
# the votes within this many steps (2 degrees each) on either side of it.
HUE_SMOOTHING = 5

# (degree where the color ends, name) in order around the color wheel.
# Red covers 345 to 360 as well as 0 to 15, because the wheel is a circle.
HUE_NAMES = [
    (15, "Red"), (40, "Orange"), (70, "Yellow"), (160, "Green"),
    (200, "Cyan"), (255, "Blue"), (290, "Purple"), (345, "Magenta"),
]


def to_hsv(image):
    """Convert a BGR frame to HSV (hue 0-179, saturation and value 0-255)."""
    _check_color_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def saturation_percent(hsv):
    """How colorful the picture is, from 0 (gray) to 100 (fully vivid).

    This is the average saturation of every pixel that isn't nearly black.
    Gray and white pixels count (as 0), so a mostly gray picture with a
    little color scores low. A picture with no visible pixels scores 0.
    """
    if hsv.size == 0:
        raise ValueError("Cannot measure an empty image.")
    visible = hsv[..., 2] >= COLOR_MIN_VALUE
    if not visible.any():
        return 0.0
    return float(hsv[..., 1][visible].mean()) / 255 * 100


def hue_histogram(hsv):
    """Count the pixels of each hue, as 180 counts (each step is 2 degrees).

    Only pixels with a clear color are counted: not nearly black and at
    least 20% saturated.
    """
    if hsv.size == 0:
        raise ValueError("Cannot measure an empty image.")
    colored = (hsv[..., 2] >= COLOR_MIN_VALUE) & (hsv[..., 1] >= HUE_MIN_SATURATION)
    return np.bincount(hsv[..., 0][colored].ravel(), minlength=180)


def dominant_hue(histogram, total_pixels):
    """The most common hue, in degrees from 0 to 358, or None for neutral.

    `total_pixels` is how many pixels the histogram was taken from. If too
    few of them have a clear color, there is no dominant hue.
    """
    if total_pixels <= 0:
        raise ValueError("The picture has no pixels.")
    if histogram.sum() < MIN_COLORED_SHARE * total_pixels:
        return None
    # Score each hue by the votes around it, wrapping from 179 back to 0.
    # Nearer votes count more (1, 2, ... 6 ... 2, 1), so a single pure color
    # scores highest exactly at its own hue.
    window = np.concatenate([np.arange(1, HUE_SMOOTHING + 2),
                             np.arange(HUE_SMOOTHING, 0, -1)]).astype(float)
    wrapped = np.concatenate(
        [histogram[-HUE_SMOOTHING:], histogram, histogram[:HUE_SMOOTHING]])
    scores = np.convolve(wrapped, window, mode="valid")
    return float(int(np.argmax(scores)) * 2)


def hue_name(degrees):
    """The everyday name of a hue: "Red", "Orange", "Yellow", and so on."""
    degrees = degrees % 360
    for end_degree, name in HUE_NAMES:
        if degrees < end_degree:
            return name
    return "Red"                              # 345 and up wraps back to red


# ---------- black bars (letterbox and pillarbox) ----------
#
# Widescreen films shown on a taller screen get black bars at the top and
# bottom (letterbox). Tall or square pictures on a wide screen get them at the
# sides (pillarbox). We find them by walking in from each edge for as long as
# every row (or column) is black.

# A pixel this dark (out of 255) counts as black. A little above zero, because
# compression leaves faint noise even in a perfectly black bar.
BAR_LUMA_MAX = 10

# A row or column counts as black when at least this share of it is. Not all
# of it, so a small logo or a speck of noise in a bar doesn't end it. A wide
# line of subtitles sitting in a bar does end it, so the bar is then reported
# as only the part between the edge and the text.
BAR_LINE_SHARE = 0.95

# Bars thinner than this percent of the picture are ignored. A pixel or two of
# black at an edge is usually a cropping or compression artifact, not a bar.
MIN_BAR_PERCENT = 1.0

# A bar thicker than this percent of its side is not counted as a bar. Real
# bars are never this big, so a picture that is black across nearly half of
# it is more likely dark content (a black sky, a dark half of a split screen).
MAX_BAR_PERCENT = 45.0


@dataclass(frozen=True)
class BlackBars:
    """How thick the black bars are, as a percent of the picture.

    top and bottom are percents of the picture's height. left and right are
    percents of its width. 0 means no bar on that side.
    """
    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0

    @property
    def any(self):
        return bool(self.top or self.bottom or self.left or self.right)


def _black_lines_from_start(is_black):
    """How many lines in a row, counting from the first, are black."""
    if is_black.all():
        return len(is_black)
    return int(np.argmin(is_black))          # position of the first False


def black_bar_lines(gray):
    """Find black bars at the four edges of a luma array.

    Returns how many pixel rows or columns each bar covers, as
    (top, bottom, left, right). 0 means no bar on that side.

    A picture that is black all the way across (a fade to black, say) has no
    bars: there is no picture for them to frame. Dark scenes with real
    picture in them can still look like bars if their edges are pure black.
    """
    if gray.size == 0:
        raise ValueError("Cannot measure an empty image.")
    height, width = gray.shape
    dark = gray <= BAR_LUMA_MAX
    black_rows = dark.mean(axis=1) >= BAR_LINE_SHARE
    black_columns = dark.mean(axis=0) >= BAR_LINE_SHARE
    if black_rows.all() or black_columns.all():
        return 0, 0, 0, 0

    def valid(lines, total):
        percent = lines / total * 100
        return lines if MIN_BAR_PERCENT <= percent <= MAX_BAR_PERCENT else 0

    return (
        valid(_black_lines_from_start(black_rows), height),
        valid(_black_lines_from_start(black_rows[::-1]), height),
        valid(_black_lines_from_start(black_columns), width),
        valid(_black_lines_from_start(black_columns[::-1]), width),
    )


def black_bars_percent(gray):
    """Black bars at the four edges, as a percent of the picture's size."""
    top, bottom, left, right = black_bar_lines(gray)
    height, width = gray.shape
    return BlackBars(
        top=top / height * 100,
        bottom=bottom / height * 100,
        left=left / width * 100,
        right=right / width * 100,
    )


def crop_bars(image, lines):
    """Cut the black bars off an image (or luma array).

    `lines` is the (top, bottom, left, right) result of black_bar_lines.
    What is left is the picture itself.
    """
    top, bottom, left, right = lines
    height, width = image.shape[:2]
    return image[top:height - bottom, left:width - right]


# ---------- waveform ----------
#
# A waveform shows how bright the picture is at each position across its
# width. Each column of the picture gets its own little brightness histogram,
# and the histograms are lined up side by side: left of the picture on the
# left, right on the right, black at the bottom and white at the top.

# How many columns the waveform has, at most. A narrower picture has fewer.
WAVEFORM_COLUMNS = 320


def luma_waveform(gray, columns=WAVEFORM_COLUMNS):
    """Count the pixels at every brightness for every part of the width.

    Returns an array with 256 rows (row 0 is black, row 255 is white) and one
    column per slice of the picture. The picture's columns are grouped into
    at most `columns` slices, as evenly as possible, and each slice counts
    every pixel in the columns it covers.
    """
    if gray.size == 0:
        raise ValueError("Cannot measure an empty image.")
    if columns < 1:
        raise ValueError("The waveform needs at least one column.")
    width = gray.shape[1]
    slices = min(columns, width)
    slice_of_column = np.arange(width) * slices // width
    combined = gray.astype(np.int64) * slices + slice_of_column[np.newaxis, :]
    counts = np.bincount(combined.ravel(), minlength=256 * slices)
    return counts.reshape(256, slices)


# ---------- vectorscope ----------
#
# A vectorscope shows which colors are in the picture and how strong they
# are. Every pixel is placed on a flat color map by its two "color
# differences": how far its blue is above or below its brightness (Cb, left
# to right) and how far its red is (Cr, bottom to top). Gray pixels land in
# the middle, and the more vivid a color, the farther from the middle it sits.
# The direction says which color it is.

# The map is this many cells wide and tall
VECTORSCOPE_SIZE = 128


def chroma(image):
    """The color differences (Cb, Cr) of every pixel of a BGR image.

    Returns two float arrays shaped like the picture. Each runs from about
    -127.5 to +127.5, and both are 0 for any gray pixel. They are scaled the
    way HD video (Rec. 709) does it, so pure blue reaches +127.5 on Cb and
    pure red reaches +127.5 on Cr.
    """
    _check_color_image(image)
    blue = image[..., 0].astype(float)
    green = image[..., 1].astype(float)
    red = image[..., 2].astype(float)
    brightness = LUMA_B * blue + LUMA_G * green + LUMA_R * red
    cb = (blue - brightness) / (2 * (1 - LUMA_B))
    cr = (red - brightness) / (2 * (1 - LUMA_R))
    return cb, cr


def chroma_of_color(blue, green, red):
    """Where one color sits on the map, as (Cb, Cr)."""
    cb, cr = chroma(np.array([[[blue, green, red]]], dtype=np.uint8))
    return float(cb[0, 0]), float(cr[0, 0])


def chroma_vectorscope(image, size=VECTORSCOPE_SIZE):
    """Count the pixels in every cell of the color map.

    Returns a (size x size) array of counts. Columns go from left to right
    with Cb, and rows go from the bottom (row 0) to the top with Cr, so the
    array reads like the picture drawn on screen once it is flipped upright.
    """
    if size < 1:
        raise ValueError("The vectorscope needs at least one cell.")
    cb, cr = chroma(image)
    columns = ((cb + 128) * size / 256).astype(np.int64).clip(0, size - 1)
    rows = ((cr + 128) * size / 256).astype(np.int64).clip(0, size - 1)
    counts = np.bincount((rows * size + columns).ravel(), minlength=size * size)
    return counts.reshape(size, size)


# ---------- color palette ----------
#
# The palette is the handful of colors that make up the picture, with how much
# of the picture each one covers. It groups similar pixels together with
# k-means in the Lab color space, where a distance between two colors is close
# to how different they look to a person, then reports the average color of
# each group.

# How many colors the palette has, at most
PALETTE_SIZE = 5

# At most this many pixels are used, picked evenly across the picture. It is
# plenty for a stable palette and keeps the grouping fast.
PALETTE_SAMPLE = 5000

# Two groups whose colors are closer than this (in Lab units, where about 2
# is the smallest difference a person can see and 10 is clearly different)
# are merged into one. Otherwise one noisy color would fill the palette with
# five almost identical shades.
PALETTE_MERGE_DISTANCE = 10.0

# k-means starts from random centers. A fixed seed makes the same picture give
# the same palette every time.
PALETTE_SEED = 7


@dataclass(frozen=True)
class PaletteColor:
    """One color of the palette and the share of the picture it covers."""
    blue: int
    green: int
    red: int
    share: float              # percent of the picture, 0 to 100

    @property
    def bgr(self):
        return (self.blue, self.green, self.red)

    @property
    def hex(self):
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"


def lab_of(lab, wanted):
    """The sum of the Lab colors of the pixels picked by a True/False array."""
    return lab[wanted].sum(axis=0)


def merge_close_colors(found):
    """Merge groups whose colors are nearly the same.

    `found` is a list of [pixel count, Lab sum, BGR sum], one per group. Going
    from the biggest group down, a group joins the nearest bigger one if their
    average colors are closer than PALETTE_MERGE_DISTANCE.
    """
    kept = []
    for count, lab_sum, bgr_sum in sorted(found, key=lambda group: -group[0]):
        lab_mean = lab_sum / count
        nearest, nearest_distance = None, PALETTE_MERGE_DISTANCE
        for group in kept:
            distance = float(np.linalg.norm(lab_mean - group[1] / group[0]))
            if distance < nearest_distance:
                nearest, nearest_distance = group, distance
        if nearest is None:
            kept.append([count, lab_sum, bgr_sum])
        else:
            nearest[0] += count
            nearest[1] = nearest[1] + lab_sum
            nearest[2] = nearest[2] + bgr_sum
    return kept


def palette_colors(pixels, count=PALETTE_SIZE, sample=PALETTE_SAMPLE):
    """The main colors of a picture, biggest share first.

    `pixels` is a BGR picture (height x width x 3) or a plain list of BGR
    pixels (n x 3), as uint8. A list is handy for pooling pixels from several
    frames. A picture with fewer than `count` different colors gets fewer
    colors in its palette. The shares add up to 100.
    """
    pixels = np.asarray(pixels)
    if pixels.ndim == 3:
        pixels = pixels.reshape(-1, pixels.shape[2])
    if pixels.ndim != 2 or pixels.shape[1] != 3 or pixels.dtype != np.uint8:
        raise ValueError("Expected a color picture (or a list of pixels) of 8-bit B, G, R values.")
    if len(pixels) == 0:
        raise ValueError("Cannot measure an empty image.")
    if count < 1:
        raise ValueError("The palette needs at least one color.")

    if len(pixels) > sample:
        picked = np.linspace(0, len(pixels) - 1, sample).astype(np.int64)
        pixels = pixels[picked]
    groups = min(count, len(np.unique(pixels, axis=0)))

    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.float32) / 255,
                       cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float64)
    if groups == 1:
        labels = np.zeros(len(pixels), dtype=np.int64)
    else:
        cv2.setRNGSeed(PALETTE_SEED)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, _ = cv2.kmeans(lab.astype(np.float32), groups, None, criteria, 3,
                                  cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()

    # Average color of each group, in Lab (to compare) and in BGR (to show)
    found = []                                   # [count, lab sum, bgr sum]
    for group in range(groups):
        wanted = labels == group
        count = int(wanted.sum())
        if count:
            found.append([count, lab_of(lab, wanted), pixels[wanted].sum(axis=0, dtype=np.float64)])
    found = merge_close_colors(found)

    colors = []
    for count, _, bgr_sum in found:
        blue, green, red = (int(round(v)) for v in bgr_sum / count)
        colors.append(PaletteColor(blue, green, red, count / len(pixels) * 100))
    colors.sort(key=lambda color: -color.share)
    return colors


# ---------- exposure overlay ----------
#
# Paints the clipped pixels on a copy of the picture, so you can see where the
# shadows are crushed and the highlights are blown. The colors are in B, G, R
# order, like every frame: blue for crushed blacks, red for blown highlights.
EXPOSURE_CRUSHED_BGR = (255, 110, 30)
EXPOSURE_BLOWN_BGR = (40, 40, 255)


def exposure_overlay(image):
    """A copy of the picture with clipped pixels painted over.

    Crushed blacks turn blue and blown highlights turn red, using the same
    rules as the Crushed blacks and Blown highlights numbers. Black bars
    around the picture are left as they are, like in the measurements.
    """
    _check_color_image(image)
    gray = luma(image)
    crushed, blown = clipping_masks(gray)
    top, bottom, left, right = black_bar_lines(gray)
    height, width = gray.shape
    inside = np.zeros(gray.shape, dtype=bool)
    inside[top:height - bottom, left:width - right] = True
    result = image.copy()
    result[crushed & inside] = EXPOSURE_CRUSHED_BGR
    result[blown & inside] = EXPOSURE_BLOWN_BGR
    return result


@dataclass
class FrameStats:
    brightness: float         # percent, 0 (black) to 100 (white)
    contrast: float           # percent, 0 (flat) to 100 (black to white)
    crushed_blacks: float     # percent of pixels at or near pure black
    blown_highlights: float   # percent of pixels at or near pure white
    dark_share: float         # percent of pixels in the darkest third
    mid_share: float          # percent of pixels in the middle third
    bright_share: float       # percent of pixels in the brightest third
    saturation: float         # percent, 0 (gray) to 100 (fully vivid)
    dominant_hue: object      # degrees 0-358, or None if the frame is neutral
    black_bars: BlackBars     # thickness of black bars at each edge, in percent
    histogram: np.ndarray     # 256 pixel counts, from black to white
    waveform: object = None   # 256 x columns pixel counts (see luma_waveform)
    palette: object = None    # list of PaletteColor, biggest share first
    vectorscope: object = None  # size x size pixel counts (see chroma_vectorscope)


def analyze_frame(image):
    """Take every measurement we have for one frame (a BGR array).

    Black bars are found first and cut off, so every other measurement
    describes the picture itself. Otherwise the bars would pull the
    brightness down and pile up at the dark end of the histogram.
    """
    small = prepare(image)
    lines = black_bar_lines(luma(small))
    picture = crop_bars(small, lines)
    gray = luma(picture)
    hsv = to_hsv(picture)
    crushed, blown = clipping_percent(gray)
    histogram = luminance_histogram(gray)
    dark, mid, bright = tonal_zones_percent(histogram)
    return FrameStats(
        brightness=brightness_percent(gray),
        contrast=contrast_percent(gray),
        crushed_blacks=crushed,
        blown_highlights=blown,
        dark_share=dark,
        mid_share=mid,
        bright_share=bright,
        saturation=saturation_percent(hsv),
        dominant_hue=dominant_hue(hue_histogram(hsv), gray.size),
        black_bars=black_bars_percent(luma(small)),
        histogram=histogram,
        waveform=luma_waveform(gray),
        palette=palette_colors(picture),
        vectorscope=chroma_vectorscope(picture),
    )