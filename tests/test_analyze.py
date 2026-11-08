"""Tests for the frame measurements in analyze.py.

Most tests use tiny made-up images whose right answer we know. The ones that
need FFmpeg make real videos to check the whole path from file to number.
"""
import shutil
import subprocess

import numpy as np
import pytest

from blackboard.analyze import (
    ANALYSIS_WIDTH, BAR_LUMA_MAX, CLIP_BLACK_MAX, CLIP_WHITE_MIN,
    EXPOSURE_BLOWN_BGR, EXPOSURE_CRUSHED_BGR, MAX_BAR_PERCENT,
    PALETTE_MERGE_DISTANCE, PALETTE_SAMPLE, PALETTE_SIZE, WAVEFORM_COLUMNS, ZONE_DARK_MAX, ZONE_MID_MAX,
    VECTORSCOPE_SIZE, BlackBars, FrameStats, PaletteColor,
    analyze_frame, black_bar_lines, black_bars_percent, brightness_percent,
    chroma, chroma_of_color, chroma_vectorscope,
    clipping_masks, clipping_percent, combine_histograms, contrast_percent,
    crop_bars, dominant_hue, exposure_overlay, hue_histogram, hue_name, luma,
    luma_waveform, luminance_histogram, palette_colors, prepare,
    saturation_percent, to_hsv,
    tonal_zones_percent,
)
from blackboard.decode import read_frame

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def solid(blue, green, red, height=8, width=8):
    """A made-up image where every pixel is the same color (B, G, R order)."""
    image = np.zeros((height, width, 3), np.uint8)
    image[:] = (blue, green, red)
    return image


# ---------- luma (per-pixel brightness) ----------

def test_luma_black_and_white():
    assert (luma(solid(0, 0, 0)) == 0).all()
    assert (luma(solid(255, 255, 255)) == 255).all()


def test_luma_returns_one_value_per_pixel():
    gray = luma(solid(10, 20, 30, height=5, width=7))
    assert gray.shape == (5, 7)
    assert gray.dtype == np.uint8


def test_green_looks_brighter_than_red_and_red_than_blue():
    # Pure colors at full strength. The eye sees green as much brighter.
    red = luma(solid(0, 0, 255))[0, 0]
    green = luma(solid(0, 255, 0))[0, 0]
    blue = luma(solid(255, 0, 0))[0, 0]
    assert green > red > blue
    assert (blue, red, green) == (18, 54, 182)   # 255 x 0.0722, 0.2126, 0.7152


def test_luma_rejects_bad_images():
    with pytest.raises(ValueError):
        luma(np.zeros((0, 0, 3), np.uint8))        # empty
    with pytest.raises(ValueError):
        luma(np.zeros((8, 8), np.uint8))           # no color channels


# ---------- brightness ----------

def test_brightness_black_is_0_and_white_is_100():
    assert brightness_percent(luma(solid(0, 0, 0))) == 0
    assert brightness_percent(luma(solid(255, 255, 255))) == pytest.approx(100)


def test_brightness_mid_gray_is_about_50():
    assert brightness_percent(luma(solid(128, 128, 128))) == pytest.approx(50.2, abs=0.1)


def test_brightness_half_black_half_white_is_50():
    image = solid(0, 0, 0, height=10, width=10)
    image[:, 5:] = 255
    assert brightness_percent(luma(image)) == pytest.approx(50)


def test_brightness_of_pure_red_is_low():
    assert brightness_percent(luma(solid(0, 0, 255))) == pytest.approx(21.2, abs=0.2)


def test_brightness_of_empty_array_raises():
    with pytest.raises(ValueError):
        brightness_percent(np.zeros((0, 0), np.uint8))


# ---------- shrinking before measuring ----------

def test_prepare_shrinks_wide_frames_and_keeps_shape():
    big = np.zeros((1080, 1920, 3), np.uint8)
    small = prepare(big)
    assert small.shape == (360, ANALYSIS_WIDTH, 3)     # 16:9 stays 16:9


def test_prepare_never_enlarges():
    image = solid(1, 2, 3, height=50, width=80)
    assert prepare(image).shape == (50, 80, 3)


def test_shrinking_barely_changes_brightness():
    # A smooth left-to-right gradient, big enough to be shrunk. It starts at
    # 60 so the left edge isn't black enough to be mistaken for a bar.
    ramp = np.linspace(60, 255, 1920).astype(np.uint8)
    image = np.repeat(ramp[None, :, None], 1080, axis=0).repeat(3, axis=2)
    full = brightness_percent(luma(image))
    assert analyze_frame(image).brightness == pytest.approx(full, abs=0.5)


def test_analyze_frame_returns_stats():
    stats = analyze_frame(solid(255, 255, 255))
    assert isinstance(stats, FrameStats)
    assert stats.brightness == pytest.approx(100)
    assert stats.contrast == 0
    assert stats.crushed_blacks == 0
    assert stats.blown_highlights == pytest.approx(100)
    assert (stats.dark_share, stats.mid_share) == (0, 0)
    assert stats.bright_share == pytest.approx(100)
    assert len(stats.histogram) == 256


# ---------- contrast ----------

def gray_with_bands(*bands):
    """A luma array built from (value, how_many_pixels) pairs, e.g.
    gray_with_bands((0, 50), (255, 50)) is half black, half white."""
    return np.concatenate(
        [np.full(count, value, np.uint8) for value, count in bands]
    ).reshape(1, -1)


def test_contrast_of_a_flat_image_is_zero():
    assert contrast_percent(luma(solid(0, 0, 0))) == 0
    assert contrast_percent(luma(solid(255, 255, 255))) == 0
    assert contrast_percent(luma(solid(128, 128, 128))) == 0


def test_contrast_of_black_and_white_is_100():
    assert contrast_percent(gray_with_bands((0, 50), (255, 50))) == pytest.approx(100)


def test_contrast_of_a_narrow_range_is_low():
    # Everything between 100 and 140: 40 steps out of 255 is about 15.7%
    gray = gray_with_bands((100, 50), (140, 50))
    assert contrast_percent(gray) == pytest.approx(40 / 255 * 100)


def test_contrast_is_the_same_for_dark_and_bright_versions_of_a_picture():
    # Brightness changes, but the gap between dark and light stays the same
    dark = gray_with_bands((10, 50), (60, 50))
    bright = gray_with_bands((150, 50), (200, 50))
    assert contrast_percent(dark) == pytest.approx(contrast_percent(bright))
    assert brightness_percent(dark) < brightness_percent(bright)


def test_contrast_of_a_smooth_ramp_is_about_90():
    # The 5th to 95th percentile of an even ramp covers 90% of the range
    ramp = np.arange(256, dtype=np.uint8).reshape(1, -1)
    assert contrast_percent(ramp) == pytest.approx(90, abs=0.5)


def test_contrast_ignores_a_few_stray_pixels():
    # 3% white specks on black: too few to count, so the picture is "flat"
    specks = gray_with_bands((0, 97), (255, 3))
    assert contrast_percent(specks) == 0


def test_contrast_counts_a_real_bright_area():
    # 10% of the picture is white, which is enough to count
    patch = gray_with_bands((0, 90), (255, 10))
    assert contrast_percent(patch) == pytest.approx(100)


def test_contrast_of_empty_array_raises():
    with pytest.raises(ValueError):
        contrast_percent(np.zeros((0, 0), np.uint8))


# ---------- the whole path: video file -> decoded frame -> number ----------

def make_color_clip(path, color):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d=1:r=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
@pytest.mark.parametrize("color, expected", [
    ("black", 0.0), ("white", 100.0), ("gray", 50.2),
])
def test_decoded_video_brightness(tmp_path, color, expected):
    # Video files often store black as 16 and white as 235. If decoding
    # didn't stretch that back to 0-255, white would measure about 92%.
    clip = make_color_clip(tmp_path / f"{color}.mp4", color)
    stats = analyze_frame(read_frame(clip, 0).image)
    assert stats.brightness == pytest.approx(expected, abs=2.0)


@needs_ffmpeg
def test_brightness_matches_for_brighter_and_darker_clips(tmp_path):
    dark = analyze_frame(read_frame(make_color_clip(tmp_path / "d.mp4", "0x202020"), 0).image)
    light = analyze_frame(read_frame(make_color_clip(tmp_path / "l.mp4", "0xE0E0E0"), 0).image)
    assert dark.brightness < 20 < 80 < light.brightness


def make_clip(path, ffmpeg_inputs, extra=()):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *ffmpeg_inputs, *extra,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_decoded_video_contrast(tmp_path):
    flat = make_color_clip(tmp_path / "flat.mp4", "gray")
    split = make_clip(
        tmp_path / "split.mp4",
        ["-f", "lavfi", "-i", "color=c=black:s=160x240:d=1:r=30",
         "-f", "lavfi", "-i", "color=c=white:s=160x240:d=1:r=30"],
        ["-filter_complex", "[0:v][1:v]hstack"],
    )
    assert analyze_frame(read_frame(flat, 0).image).contrast < 2
    assert analyze_frame(read_frame(split, 0).image).contrast > 98


# ---------- clipping ----------

def test_clipping_of_black_and_white_images():
    assert clipping_percent(luma(solid(0, 0, 0))) == (100, 0)
    assert clipping_percent(luma(solid(255, 255, 255))) == (0, 100)


def test_clipping_of_mid_gray_is_zero():
    assert clipping_percent(luma(solid(128, 128, 128))) == (0, 0)


def test_clipping_half_black_half_white():
    crushed, blown = clipping_percent(gray_with_bands((0, 50), (255, 50)))
    assert (crushed, blown) == (50, 50)


def test_clipping_counts_a_small_patch():
    # 10% white patch on mid-gray: no black, a tenth of the picture blown out
    gray = gray_with_bands((128, 90), (255, 10))
    assert clipping_percent(gray) == (0, pytest.approx(10))


def test_clipping_includes_pixels_close_to_black_and_white():
    # Right on the margins counts as clipped...
    near = gray_with_bands((CLIP_BLACK_MAX, 25), (CLIP_WHITE_MIN, 25), (128, 50))
    assert clipping_percent(near) == (pytest.approx(25), pytest.approx(25))
    # ...and one step inside the margins does not
    inside = gray_with_bands(
        (CLIP_BLACK_MAX + 1, 25), (CLIP_WHITE_MIN - 1, 25), (128, 50))
    assert clipping_percent(inside) == (0, 0)


def test_clipping_masks_mark_the_right_pixels():
    gray = np.array([[0, 128, 255]], np.uint8)
    black, white = clipping_masks(gray)
    assert black.tolist() == [[True, False, False]]
    assert white.tolist() == [[False, False, True]]
    assert black.shape == white.shape == gray.shape


def test_clipping_of_empty_array_raises():
    with pytest.raises(ValueError):
        clipping_percent(np.zeros((0, 0), np.uint8))


def test_shrinking_barely_changes_clipping():
    # A black stretch, a gradient and a white stretch, with gray at both edges
    # so the black isn't mistaken for a bar
    ramp = np.concatenate([
        np.full(160, 128, np.uint8), np.zeros(480, np.uint8),
        np.linspace(0, 255, 640).astype(np.uint8),
        np.full(480, 255, np.uint8), np.full(160, 128, np.uint8),
    ])
    image = np.repeat(ramp[None, :, None], 1080, axis=0).repeat(3, axis=2)
    full = clipping_percent(luma(image))
    small = analyze_frame(image)
    assert small.crushed_blacks == pytest.approx(full[0], abs=1.0)
    assert small.blown_highlights == pytest.approx(full[1], abs=1.0)


@needs_ffmpeg
def test_decoded_video_clipping(tmp_path):
    black = make_color_clip(tmp_path / "black.mp4", "black")
    white = make_color_clip(tmp_path / "white.mp4", "white")
    gray = make_color_clip(tmp_path / "gray.mp4", "gray")
    split = make_clip(
        tmp_path / "split.mp4",
        ["-f", "lavfi", "-i", "color=c=black:s=160x240:d=1:r=30",
         "-f", "lavfi", "-i", "color=c=white:s=160x240:d=1:r=30"],
        ["-filter_complex", "[0:v][1:v]hstack"],
    )
    b = analyze_frame(read_frame(black, 0).image)
    w = analyze_frame(read_frame(white, 0).image)
    g = analyze_frame(read_frame(gray, 0).image)
    s = analyze_frame(read_frame(split, 0).image)
    assert (b.crushed_blacks, b.blown_highlights) == (pytest.approx(100, abs=1), 0)
    assert (w.crushed_blacks, w.blown_highlights) == (0, pytest.approx(100, abs=1))
    assert (g.crushed_blacks, g.blown_highlights) == (0, 0)
    assert s.crushed_blacks == pytest.approx(50, abs=3)
    assert s.blown_highlights == pytest.approx(50, abs=3)


# ---------- luminance histogram ----------

def test_histogram_has_256_counts_that_add_up_to_the_pixel_count():
    hist = luminance_histogram(luma(solid(10, 200, 30, height=6, width=7)))
    assert len(hist) == 256
    assert hist.sum() == 6 * 7


def test_histogram_of_black_and_white_images():
    black = luminance_histogram(luma(solid(0, 0, 0, height=10, width=10)))
    white = luminance_histogram(luma(solid(255, 255, 255, height=10, width=10)))
    assert black[0] == 100 and black.sum() == 100
    assert white[255] == 100 and white.sum() == 100


def test_histogram_of_half_black_half_white():
    hist = luminance_histogram(gray_with_bands((0, 50), (255, 50)))
    assert hist[0] == 50 and hist[255] == 50 and hist.sum() == 100


def test_histogram_of_a_ramp_has_one_pixel_in_every_bin():
    ramp = np.arange(256, dtype=np.uint8).reshape(1, -1)
    assert (luminance_histogram(ramp) == 1).all()


def test_histogram_of_empty_array_raises():
    with pytest.raises(ValueError):
        luminance_histogram(np.zeros((0, 0), np.uint8))


def test_combining_histograms_adds_them_up():
    black = luminance_histogram(gray_with_bands((0, 100)))
    white = luminance_histogram(gray_with_bands((255, 100)))
    both = combine_histograms([black, white])
    assert both[0] == 100 and both[255] == 100 and both.sum() == 200
    assert len(both) == 256


def test_combined_histogram_describes_all_the_frames_together():
    # Three black frames and one white frame: three quarters of all the
    # pixels are dark
    frames = [gray_with_bands((0, 100))] * 3 + [gray_with_bands((255, 100))]
    combined = combine_histograms([luminance_histogram(f) for f in frames])
    dark, mid, bright = tonal_zones_percent(combined)
    assert (dark, mid, bright) == (pytest.approx(75), 0, pytest.approx(25))


def test_combining_nothing_raises():
    with pytest.raises(ValueError):
        combine_histograms([])


# ---------- dark / mid / bright thirds ----------

def zones_of(*bands):
    return tonal_zones_percent(luminance_histogram(gray_with_bands(*bands)))


def test_zones_of_black_gray_and_white():
    assert zones_of((0, 10)) == (100, 0, 0)
    assert zones_of((128, 10)) == (0, 100, 0)
    assert zones_of((255, 10)) == (0, 0, 100)


def test_zone_edges():
    assert zones_of((ZONE_DARK_MAX, 10))[0] == 100          # still dark
    assert zones_of((ZONE_DARK_MAX + 1, 10))[1] == 100      # first mid value
    assert zones_of((ZONE_MID_MAX, 10))[1] == 100           # last mid value
    assert zones_of((ZONE_MID_MAX + 1, 10))[2] == 100       # first bright value


def test_zones_always_add_up_to_100():
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    assert sum(tonal_zones_percent(luminance_histogram(gray))) == pytest.approx(100)


def test_zones_of_an_empty_histogram_raise():
    with pytest.raises(ValueError):
        tonal_zones_percent(np.zeros(256, np.int64))


def test_analyze_frame_histogram_covers_the_shrunken_picture():
    stats = analyze_frame(np.zeros((1080, 1920, 3), np.uint8))
    assert stats.histogram.sum() == 360 * ANALYSIS_WIDTH      # shrunk first
    assert stats.histogram[0] == stats.histogram.sum()
    assert stats.dark_share == 100


@needs_ffmpeg
def test_decoded_video_tonal_zones(tmp_path):
    black = analyze_frame(read_frame(make_color_clip(tmp_path / "b.mp4", "black"), 0).image)
    gray = analyze_frame(read_frame(make_color_clip(tmp_path / "g.mp4", "gray"), 0).image)
    white = analyze_frame(read_frame(make_color_clip(tmp_path / "w.mp4", "white"), 0).image)
    assert (black.dark_share, black.mid_share, black.bright_share) == (pytest.approx(100, abs=1), pytest.approx(0, abs=1), 0)
    assert (gray.dark_share, gray.mid_share, gray.bright_share) == (0, pytest.approx(100), 0)
    assert (white.dark_share, white.mid_share, white.bright_share) == (0, 0, pytest.approx(100, abs=1))



# ---------- saturation ----------

def test_gray_black_and_white_have_no_saturation():
    for value in (0, 128, 255):
        assert saturation_percent(to_hsv(solid(value, value, value))) == 0


def test_a_pure_color_is_fully_saturated():
    assert saturation_percent(to_hsv(solid(0, 0, 255))) == pytest.approx(100)
    assert saturation_percent(to_hsv(solid(0, 255, 0))) == pytest.approx(100)


def test_a_pastel_color_is_about_half_saturated():
    # Red at full strength mixed half way with white
    assert saturation_percent(to_hsv(solid(128, 128, 255))) == pytest.approx(50, abs=1)


def test_half_color_half_gray_is_about_50():
    image = solid(128, 128, 128, height=8, width=8)
    image[:, :4] = (0, 0, 255)
    assert saturation_percent(to_hsv(image)) == pytest.approx(50, abs=1)


def test_almost_black_pixels_are_left_out():
    # A "color" this dark is really noise. Without leaving it out, these
    # pixels would count as fully saturated and drag the average up.
    image = solid(0, 0, 10, height=8, width=8)       # looks black, 100% "saturated"
    image[:, :4] = (128, 128, 128)                    # plain gray
    assert saturation_percent(to_hsv(image)) == 0


def test_saturation_of_a_black_picture_is_zero():
    assert saturation_percent(to_hsv(solid(0, 0, 0))) == 0


def test_saturation_of_empty_array_raises():
    with pytest.raises(ValueError):
        saturation_percent(np.zeros((0, 0, 3), np.uint8))


def test_to_hsv_rejects_bad_images():
    with pytest.raises(ValueError):
        to_hsv(np.zeros((4, 4), np.uint8))


def test_shrinking_barely_changes_saturation():
    rng = np.random.default_rng(1)
    big = np.zeros((1080, 1920, 3), np.uint8)
    big[:, :960] = (40, 120, 220)
    big[:, 960:] = (200, 160, 90)
    noisy = np.clip(big.astype(int) + rng.integers(-6, 7, big.shape), 0, 255).astype(np.uint8)
    full = saturation_percent(to_hsv(noisy))
    assert analyze_frame(noisy).saturation == pytest.approx(full, abs=2.0)


# ---------- hue ----------

def circular_gap(a, b):
    """The distance between two hues in degrees, going the short way round."""
    gap = abs(a - b) % 360
    return min(gap, 360 - gap)


def hue_of(image):
    hsv = to_hsv(image)
    return dominant_hue(hue_histogram(hsv), hsv.shape[0] * hsv.shape[1])


@pytest.mark.parametrize("bgr, degrees", [
    ((0, 0, 255), 0),        # red
    ((0, 128, 255), 30),     # orange
    ((0, 255, 255), 60),     # yellow
    ((0, 255, 0), 120),      # green
    ((255, 255, 0), 180),    # cyan
    ((255, 0, 0), 240),      # blue
    ((255, 0, 128), 270),    # purple
    ((255, 0, 255), 300),    # magenta
])
def test_dominant_hue_of_pure_colors(bgr, degrees):
    assert circular_gap(hue_of(solid(*bgr)), degrees) <= 2


def test_hue_histogram_has_180_counts():
    histogram = hue_histogram(to_hsv(solid(0, 255, 0)))
    assert histogram.shape == (180,)
    assert histogram.sum() == 64                 # every pixel of an 8 x 8 image
    assert histogram[60] == 64                   # green is hue 120 degrees = step 60


def test_grays_and_dark_pixels_do_not_vote_for_a_hue():
    assert hue_histogram(to_hsv(solid(128, 128, 128))).sum() == 0
    assert hue_histogram(to_hsv(solid(0, 0, 10))).sum() == 0
    assert hue_histogram(to_hsv(solid(255, 255, 255))).sum() == 0


def test_a_neutral_picture_has_no_dominant_hue():
    assert hue_of(solid(128, 128, 128)) is None
    assert hue_of(solid(0, 0, 0)) is None


def test_a_tiny_patch_of_color_is_not_enough():
    # One pixel in 100 is under the 2% minimum, so the picture counts as neutral
    image = solid(128, 128, 128, height=10, width=10)
    image[0, 0] = (0, 0, 255)
    assert hue_of(image) is None


def test_a_small_but_big_enough_patch_gives_a_hue():
    # Five pixels in 100 is over the minimum
    image = solid(128, 128, 128, height=10, width=10)
    image[0, :5] = (0, 0, 255)
    assert circular_gap(hue_of(image), 0) <= 2


def test_the_most_common_hue_wins():
    image = solid(0, 255, 0, height=10, width=10)    # green
    image[:3] = (255, 0, 0)                          # blue on 30% of the rows
    assert circular_gap(hue_of(image), 120) <= 2


def test_gray_background_does_not_hide_a_color():
    image = solid(128, 128, 128, height=10, width=10)
    image[:3] = (255, 0, 0)                          # blue on 30%, gray on 70%
    assert circular_gap(hue_of(image), 240) <= 2


def test_hue_wraps_around_red():
    # Two reds either side of 0 degrees (hue steps 178 and 2) are the same
    # color family. A plain average would land on green. The circle gives red.
    hsv = np.zeros((8, 8, 3), np.uint8)
    hsv[..., 1] = hsv[..., 2] = 255
    hsv[:, :4, 0] = 178
    hsv[:, 4:, 0] = 2
    degrees = dominant_hue(hue_histogram(hsv), 64)
    assert circular_gap(degrees, 0) <= 2


def test_dominant_hue_with_no_pixels_raises():
    with pytest.raises(ValueError):
        dominant_hue(np.zeros(180, np.int64), 0)


def test_hue_histogram_of_empty_array_raises():
    with pytest.raises(ValueError):
        hue_histogram(np.zeros((0, 0, 3), np.uint8))


def test_hue_histograms_can_be_combined_for_a_segment():
    red = hue_histogram(to_hsv(solid(0, 0, 255)))
    green = hue_histogram(to_hsv(solid(0, 255, 0)))
    both = combine_histograms([red, red, green])      # two red frames, one green
    assert circular_gap(dominant_hue(both, 3 * 64), 0) <= 2


@pytest.mark.parametrize("degrees, name", [
    (0, "Red"), (10, "Red"), (30, "Orange"), (60, "Yellow"), (120, "Green"),
    (180, "Cyan"), (240, "Blue"), (270, "Purple"), (300, "Magenta"),
    (350, "Red"), (359, "Red"), (360, "Red"), (-10, "Red"),
])
def test_hue_names(degrees, name):
    assert hue_name(degrees) == name


def test_analyze_frame_reports_color():
    stats = analyze_frame(solid(0, 255, 0, height=20, width=20))
    assert stats.saturation == pytest.approx(100)
    assert circular_gap(stats.dominant_hue, 120) <= 2
    gray = analyze_frame(solid(128, 128, 128, height=20, width=20))
    assert gray.saturation == 0
    assert gray.dominant_hue is None


@needs_ffmpeg
@pytest.mark.parametrize("color, degrees", [
    ("red", 0), ("lime", 120), ("blue", 240), ("yellow", 60),
])
def test_decoded_video_hue_and_saturation(tmp_path, color, degrees):
    # Compression and the video color range shift the numbers a little, so
    # allow some slack. Red, green and blue must still land in the right place.
    clip = make_color_clip(tmp_path / f"{color}.mp4", color)
    stats = analyze_frame(read_frame(clip, 0).image)
    assert circular_gap(stats.dominant_hue, degrees) <= 6
    assert stats.saturation > 90


@needs_ffmpeg
def test_decoded_gray_video_has_no_color(tmp_path):
    stats = analyze_frame(read_frame(make_color_clip(tmp_path / "g.mp4", "gray"), 0).image)
    assert stats.saturation < 5
    assert stats.dominant_hue is None



# ---------- black bars (letterbox and pillarbox) ----------

def picture_with_bars(height=100, width=200, top=0, bottom=0, left=0, right=0, value=128):
    """A gray luma picture with solid black bars of the given thickness."""
    gray = np.full((height, width), value, np.uint8)
    if top:
        gray[:top] = 0
    if bottom:
        gray[-bottom:] = 0
    if left:
        gray[:, :left] = 0
    if right:
        gray[:, -right:] = 0
    return gray


def test_no_bars_in_a_plain_picture():
    assert black_bars_percent(picture_with_bars()) == BlackBars()
    assert not black_bars_percent(picture_with_bars()).any


def test_letterbox_bars_top_and_bottom():
    bars = black_bars_percent(picture_with_bars(top=12, bottom=12))
    assert (bars.top, bars.bottom, bars.left, bars.right) == (12, 12, 0, 0)
    assert bars.any


def test_pillarbox_bars_left_and_right():
    bars = black_bars_percent(picture_with_bars(left=20, right=30))
    assert (bars.top, bars.bottom, bars.left, bars.right) == (0, 0, 10, 15)


def test_a_bar_on_one_side_only():
    bars = black_bars_percent(picture_with_bars(top=25))
    assert (bars.top, bars.bottom) == (25, 0)


def test_bars_on_all_four_sides():
    bars = black_bars_percent(picture_with_bars(top=10, bottom=10, left=20, right=20))
    assert (bars.top, bars.bottom, bars.left, bars.right) == (10, 10, 10, 10)


def test_a_fully_black_picture_has_no_bars():
    # Nothing is framed, so nothing is a bar (a fade to black, for example)
    assert black_bars_percent(np.zeros((50, 80), np.uint8)) == BlackBars()


def test_a_fully_bright_picture_has_no_bars():
    assert black_bars_percent(np.full((50, 80), 255, np.uint8)) == BlackBars()


def test_a_single_black_line_at_the_edge_is_ignored():
    # One row in 200 is half a percent: under the 1% minimum
    assert black_bars_percent(picture_with_bars(height=200, top=1)) == BlackBars()


def test_a_bar_of_exactly_the_minimum_counts():
    assert black_bars_percent(picture_with_bars(height=100, top=1)).top == 1


def test_noisy_black_bars_still_count():
    gray = picture_with_bars(top=12, bottom=12)
    rng = np.random.default_rng(3)
    noise = rng.integers(0, BAR_LUMA_MAX + 1, gray.shape).astype(np.uint8)
    gray[:12] = noise[:12]
    gray[-12:] = noise[-12:]
    bars = black_bars_percent(gray)
    assert (bars.top, bars.bottom) == (12, 12)


def test_dark_gray_bars_are_not_black():
    # A bar of brightness 40 is dark gray, not black
    assert black_bars_percent(picture_with_bars(top=12)).top == 12
    gray = picture_with_bars(top=0)
    gray[:12] = 40
    assert black_bars_percent(gray).top == 0


def test_a_small_logo_inside_a_bar_does_not_end_it():
    gray = picture_with_bars(top=0, bottom=20)
    gray[-12:-8, 90:98] = 255               # 8 of 200 columns: 4% of those rows
    assert black_bars_percent(gray).bottom == 20


def test_a_wide_bright_line_ends_the_bar():
    gray = picture_with_bars(top=20)
    gray[10, 20:180] = 255                  # bright across most of the row
    assert black_bars_percent(gray).top == 10


def test_a_bright_line_in_the_picture_does_not_stop_the_other_edge():
    gray = picture_with_bars(top=20, bottom=20)
    gray[50, :] = 255
    bars = black_bars_percent(gray)
    assert (bars.top, bars.bottom) == (20, 20)


def test_bars_of_empty_array_raises():
    with pytest.raises(ValueError):
        black_bars_percent(np.zeros((0, 0), np.uint8))


def test_analyze_frame_reports_bars():
    image = np.full((100, 200, 3), 128, np.uint8)
    image[:15] = 0
    image[-15:] = 0
    bars = analyze_frame(image).black_bars
    assert (bars.top, bars.bottom, bars.left, bars.right) == (15, 15, 0, 0)


def test_shrinking_barely_changes_bars():
    # 140 rows of 1080 is 12.96% at the top and bottom
    image = np.full((1080, 1920, 3), 120, np.uint8)
    image[:140] = 0
    image[-140:] = 0
    bars = analyze_frame(image).black_bars
    assert bars.top == pytest.approx(12.96, abs=0.6)
    assert bars.bottom == pytest.approx(12.96, abs=0.6)
    assert bars.left == 0 and bars.right == 0


def make_padded_clip(path, pad):
    """A moving test picture with black padding added by FFmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=s=320x180:d=1:r=30",
         "-vf", pad, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_decoded_video_letterbox(tmp_path):
    # 30 black rows above and below a 180-row picture: 30 of 240 = 12.5%
    clip = make_padded_clip(tmp_path / "letterbox.mp4", "pad=320:240:0:30:black")
    bars = analyze_frame(read_frame(clip, 0).image).black_bars
    assert bars.top == pytest.approx(12.5, abs=1.0)
    assert bars.bottom == pytest.approx(12.5, abs=1.0)
    assert bars.left == 0 and bars.right == 0


@needs_ffmpeg
def test_decoded_video_pillarbox(tmp_path):
    # 40 black columns either side of a 320-column picture: 40 of 400 = 10%
    clip = make_padded_clip(tmp_path / "pillarbox.mp4", "pad=400:180:40:0:black")
    bars = analyze_frame(read_frame(clip, 0).image).black_bars
    assert bars.left == pytest.approx(10.0, abs=1.0)
    assert bars.right == pytest.approx(10.0, abs=1.0)
    assert bars.top == 0 and bars.bottom == 0


@needs_ffmpeg
def test_decoded_video_without_bars(tmp_path):
    clip = make_padded_clip(tmp_path / "plain.mp4", "null")
    assert not analyze_frame(read_frame(clip, 0).image).black_bars.any



# ---------- measuring inside the bars ----------

def colorful_picture(height=60, width=120):
    """A made-up picture with a gradient and some color, no black edges."""
    rng = np.random.default_rng(7)
    image = rng.integers(60, 230, (height, width, 3)).astype(np.uint8)
    image[:, : width // 3, 2] = 250                  # a reddish third
    return image


def add_bars(image, top=0, bottom=0, left=0, right=0):
    """The picture with black bars added around it."""
    height, width = image.shape[:2]
    out = np.zeros((height + top + bottom, width + left + right, 3), np.uint8)
    out[top:top + height, left:left + width] = image
    return out


def test_crop_bars_cuts_the_right_amount():
    image = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    cropped = crop_bars(image, (2, 3, 4, 5))
    assert cropped.shape == (5, 11, 3)
    assert (cropped == image[2:7, 4:15]).all()


def test_crop_bars_with_no_bars_changes_nothing():
    image = np.zeros((10, 20, 3), np.uint8)
    assert crop_bars(image, (0, 0, 0, 0)).shape == image.shape


def test_crop_bars_works_on_a_luma_array_too():
    assert crop_bars(np.zeros((10, 20), np.uint8), (1, 1, 2, 2)).shape == (8, 16)


def test_bar_lines_are_whole_pixel_counts():
    lines = black_bar_lines(picture_with_bars(top=12, bottom=7, left=9, right=0))
    assert lines == (12, 7, 9, 0)


def test_a_bar_just_under_the_maximum_counts():
    height = 100
    assert black_bar_lines(picture_with_bars(height=height, top=int(MAX_BAR_PERCENT)))[0] == int(MAX_BAR_PERCENT)


def test_a_bar_over_the_maximum_is_probably_dark_content():
    # 46% black along the top is more likely a black sky than a bar
    assert black_bar_lines(picture_with_bars(height=100, top=46)) == (0, 0, 0, 0)


@pytest.mark.parametrize("bars", [
    dict(top=15, bottom=15),                  # letterbox
    dict(left=30, right=30),                  # pillarbox
    dict(top=10, bottom=10, left=20, right=20),
    dict(top=20),                             # one side only
])
def test_measurements_ignore_the_bars(bars):
    picture = colorful_picture()
    plain = analyze_frame(picture)
    boxed = analyze_frame(add_bars(picture, **bars))
    assert boxed.brightness == pytest.approx(plain.brightness)
    assert boxed.contrast == pytest.approx(plain.contrast)
    assert boxed.crushed_blacks == pytest.approx(plain.crushed_blacks)
    assert boxed.blown_highlights == pytest.approx(plain.blown_highlights)
    assert boxed.saturation == pytest.approx(plain.saturation)
    assert boxed.dominant_hue == plain.dominant_hue
    assert (boxed.dark_share, boxed.mid_share, boxed.bright_share) == pytest.approx(
        (plain.dark_share, plain.mid_share, plain.bright_share))
    assert (boxed.histogram == plain.histogram).all()


def test_the_bars_are_still_reported():
    boxed = analyze_frame(add_bars(colorful_picture(), top=15, bottom=15))
    assert boxed.black_bars.top == pytest.approx(15 / 90 * 100)
    assert boxed.black_bars.bottom == pytest.approx(15 / 90 * 100)


def test_the_histogram_counts_only_the_picture():
    picture = colorful_picture(height=60, width=120)
    stats = analyze_frame(add_bars(picture, top=15, bottom=15))
    assert stats.histogram.sum() == 60 * 120
    assert stats.crushed_blacks == 0                    # the bars aren't crushed blacks


def test_brightness_of_a_letterboxed_gray_picture_is_the_gray():
    gray = np.full((60, 120, 3), 128, np.uint8)
    stats = analyze_frame(add_bars(gray, top=30, bottom=30))
    assert stats.brightness == pytest.approx(128 / 255 * 100)
    assert stats.contrast == 0


def test_a_small_patch_of_color_is_judged_against_the_picture():
    # 3% of the picture is red. Counted against the whole frame, bars included,
    # it would be only 1% and fall under the minimum for a dominant hue.
    picture = np.full((100, 100, 3), 128, np.uint8)
    picture[:3] = (0, 0, 255)
    plain = analyze_frame(picture)
    boxed = analyze_frame(add_bars(picture, top=100, bottom=100))
    assert plain.dominant_hue is not None
    assert boxed.dominant_hue == plain.dominant_hue


def test_black_inside_the_picture_is_not_cropped():
    picture = np.full((60, 120, 3), 128, np.uint8)
    picture[:, 50:70] = 0                               # a black stripe in the middle
    stats = analyze_frame(add_bars(picture, top=15, bottom=15))
    assert stats.histogram.sum() == 60 * 120
    assert stats.crushed_blacks == pytest.approx(20 / 120 * 100)


def test_a_half_black_picture_is_content_not_a_bar():
    image = np.full((60, 120, 3), 255, np.uint8)
    image[:, :60] = 0                                   # left half black
    stats = analyze_frame(image)
    assert not stats.black_bars.any
    assert stats.brightness == pytest.approx(50)
    assert stats.crushed_blacks == pytest.approx(50)


def test_a_black_picture_keeps_all_its_pixels():
    stats = analyze_frame(np.zeros((60, 120, 3), np.uint8))
    assert stats.histogram.sum() == 60 * 120
    assert stats.crushed_blacks == 100
    assert not stats.black_bars.any


@needs_ffmpeg
@pytest.mark.parametrize("pad", ["pad=320:240:0:30:black", "pad=400:180:40:0:black"])
def test_decoded_video_measures_the_picture_not_the_bars(tmp_path, pad):
    plain = make_padded_clip(tmp_path / "plain.mp4", "null")
    boxed = make_padded_clip(tmp_path / "boxed.mp4", pad)
    a = analyze_frame(read_frame(plain, 0).image)
    b = analyze_frame(read_frame(boxed, 0).image)
    assert b.brightness == pytest.approx(a.brightness, abs=1.5)
    assert b.contrast == pytest.approx(a.contrast, abs=3.0)
    assert b.crushed_blacks == pytest.approx(a.crushed_blacks, abs=1.5)
    assert b.saturation == pytest.approx(a.saturation, abs=2.0)



# ---------- exposure overlay ----------

def gray_with_patches():
    """Mid-gray picture with a black patch and a white patch."""
    image = np.full((60, 120, 3), 128, np.uint8)
    image[10:20, 10:20] = 0
    image[30:40, 50:60] = 255
    return image


def test_overlay_paints_crushed_blacks_blue_and_blown_highlights_red():
    out = exposure_overlay(gray_with_patches())
    assert tuple(out[15, 15]) == EXPOSURE_CRUSHED_BGR
    assert tuple(out[35, 55]) == EXPOSURE_BLOWN_BGR


def test_overlay_leaves_everything_else_alone():
    image = gray_with_patches()
    out = exposure_overlay(image)
    painted = (out != image).any(axis=2)
    assert painted.sum() == 10 * 10 + 10 * 10            # just the two patches
    assert (out[~painted] == image[~painted]).all()


def test_overlay_does_not_change_the_original_picture():
    image = gray_with_patches()
    before = image.copy()
    exposure_overlay(image)
    assert (image == before).all()


def test_overlay_keeps_the_size_and_type():
    out = exposure_overlay(gray_with_patches())
    assert out.shape == (60, 120, 3) and out.dtype == np.uint8


def test_overlay_of_a_picture_with_no_clipping_is_identical():
    image = solid(128, 128, 128, height=20, width=20)
    assert (exposure_overlay(image) == image).all()


def test_overlay_uses_the_same_limits_as_the_measurements():
    # Gray at both edges, so the dark strips can't be mistaken for bars
    image = np.full((20, 60, 3), 128, np.uint8)
    image[:, 10:20] = CLIP_BLACK_MAX            # just dark enough to count
    image[:, 20:30] = CLIP_BLACK_MAX + 1        # just too bright to count
    image[:, 30:40] = CLIP_WHITE_MIN            # just bright enough to count
    image[:, 40:50] = CLIP_WHITE_MIN - 1        # just too dark to count
    painted = (exposure_overlay(image) != image).any(axis=2)
    assert painted[:, 10:20].all() and painted[:, 30:40].all()
    assert not painted[:, 20:30].any() and not painted[:, 40:50].any()


def test_overlay_does_not_paint_the_black_bars():
    picture = gray_with_patches()
    boxed = add_bars(picture, top=15, bottom=15)
    out = exposure_overlay(boxed)
    assert (out[:15] == 0).all() and (out[-15:] == 0).all()       # bars untouched
    assert tuple(out[15 + 15, 15]) == EXPOSURE_CRUSHED_BGR        # patch still painted


def test_overlay_does_not_paint_pillarbox_bars():
    boxed = add_bars(gray_with_patches(), left=30, right=30)
    out = exposure_overlay(boxed)
    assert (out[:, :30] == 0).all() and (out[:, -30:] == 0).all()


def test_painted_share_matches_the_crushed_and_blown_numbers():
    boxed = add_bars(gray_with_patches(), top=15, bottom=15)
    stats = analyze_frame(boxed)
    out = exposure_overlay(boxed)
    picture = out[15:-15]                                  # the picture area
    crushed = (picture == EXPOSURE_CRUSHED_BGR).all(axis=2).mean() * 100
    blown = (picture == EXPOSURE_BLOWN_BGR).all(axis=2).mean() * 100
    assert crushed == pytest.approx(stats.crushed_blacks)
    assert blown == pytest.approx(stats.blown_highlights)


def test_overlay_of_a_black_frame_paints_all_of_it():
    out = exposure_overlay(np.zeros((20, 30, 3), np.uint8))
    assert (out == EXPOSURE_CRUSHED_BGR).all()


def test_overlay_rejects_bad_images():
    with pytest.raises(ValueError):
        exposure_overlay(np.zeros((4, 4), np.uint8))



# ---------- waveform ----------

def test_waveform_has_256_rows_and_at_most_the_column_limit():
    gray = np.zeros((40, 640), np.uint8)
    assert luma_waveform(gray).shape == (256, WAVEFORM_COLUMNS)


def test_a_narrow_picture_gets_one_slice_per_column():
    gray = np.zeros((40, 50), np.uint8)
    assert luma_waveform(gray).shape == (256, 50)


def test_waveform_counts_every_pixel_once():
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (36, 200)).astype(np.uint8)
    assert luma_waveform(gray).sum() == gray.size
    assert luma_waveform(gray, columns=7).sum() == gray.size


def test_waveform_of_black_and_white_pictures():
    black = luma_waveform(np.zeros((10, 40), np.uint8))
    white = luma_waveform(np.full((10, 40), 255, np.uint8))
    assert black[0].sum() == black.sum()                 # everything on the black row
    assert white[255].sum() == white.sum()               # everything on the white row


def test_waveform_follows_left_and_right_halves():
    gray = np.zeros((10, 40), np.uint8)
    gray[:, 20:] = 255                                   # right half white
    counts = luma_waveform(gray)
    assert counts[0, :20].sum() == 10 * 20 and counts[0, 20:].sum() == 0
    assert counts[255, 20:].sum() == 10 * 20 and counts[255, :20].sum() == 0


def test_waveform_of_a_left_to_right_ramp_climbs():
    ramp = np.tile(np.linspace(0, 255, 100).astype(np.uint8), (20, 1))
    counts = luma_waveform(ramp)
    peaks = counts.argmax(axis=0)
    assert (np.diff(peaks) >= 0).all()                   # never goes back down
    assert peaks[0] <= 5 and peaks[-1] >= 250


def test_waveform_of_a_top_to_bottom_ramp_is_the_same_in_every_column():
    ramp = np.tile(np.linspace(0, 255, 60).astype(np.uint8)[:, None], (1, 80))
    counts = luma_waveform(ramp)
    assert (counts == counts[:, :1]).all()


def test_a_bright_patch_only_shows_in_its_own_columns():
    gray = np.full((20, 100), 100, np.uint8)
    gray[:, 10:20] = 230
    counts = luma_waveform(gray, columns=100)
    assert counts[230, 10:20].sum() == 20 * 10
    assert counts[230].sum() == 20 * 10                  # nowhere else
    assert counts[100, 10:20].sum() == 0


def test_every_slice_holds_the_same_number_of_pixels():
    gray = np.zeros((30, 640), np.uint8)
    totals = luma_waveform(gray).sum(axis=0)
    assert totals.min() == totals.max() == 30 * 2        # 2 picture columns per slice


def test_uneven_slices_differ_by_one_column_at_most():
    gray = np.zeros((10, 100), np.uint8)
    totals = luma_waveform(gray, columns=30).sum(axis=0)
    assert totals.sum() == gray.size
    assert (totals.max() - totals.min()) <= 10           # one picture column of 10 pixels


def test_a_single_column_waveform_is_the_histogram():
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 256, (30, 70)).astype(np.uint8)
    assert (luma_waveform(gray, columns=1)[:, 0] == luminance_histogram(gray)).all()


def test_waveform_of_empty_array_raises():
    with pytest.raises(ValueError):
        luma_waveform(np.zeros((0, 0), np.uint8))


def test_waveform_needs_at_least_one_column():
    with pytest.raises(ValueError):
        luma_waveform(np.zeros((4, 4), np.uint8), columns=0)


def test_analyze_frame_waveform_adds_up_to_the_histogram():
    rng = np.random.default_rng(5)
    image = rng.integers(40, 220, (60, 120, 3)).astype(np.uint8)
    stats = analyze_frame(image)
    assert stats.waveform.shape == (256, 120)
    assert (stats.waveform.sum(axis=1) == stats.histogram).all()


def test_analyze_frame_waveform_leaves_out_the_bars():
    picture = colorful_picture(height=60, width=100)
    stats = analyze_frame(add_bars(picture, left=40, right=40))
    assert stats.waveform.shape == (256, 100)            # the picture's width only
    assert stats.waveform[0].sum() == analyze_frame(picture).waveform[0].sum()


def test_frame_stats_waveform_defaults_to_nothing():
    stats = FrameStats(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, None, BlackBars(),
                       np.ones(256, np.int64))
    assert stats.waveform is None


def test_neighboring_picture_columns_share_a_slice():
    # 100 picture columns grouped into 10 slices: left half black, right half white
    gray = np.zeros((10, 100), np.uint8)
    gray[:, 50:] = 255
    counts = luma_waveform(gray, columns=10)
    assert counts.shape == (256, 10)
    assert counts[0, :5].sum() == 10 * 50 and counts[0, 5:].sum() == 0
    assert counts[255, 5:].sum() == 10 * 50 and counts[255, :5].sum() == 0


def test_grouped_slices_keep_the_left_to_right_order():
    ramp = np.tile(np.linspace(0, 255, 640).astype(np.uint8), (8, 1))
    peaks = luma_waveform(ramp).argmax(axis=0)           # 640 columns in 320 slices
    assert (np.diff(peaks) >= 0).all()
    assert peaks[0] <= 3 and peaks[-1] >= 252



# ---------- color palette ----------

def blocks(*colors_and_rows, width=40):
    """A picture made of horizontal blocks: blocks((B, G, R), rows), ..."""
    parts = [np.full((rows, width, 3), color, np.uint8) for color, rows in colors_and_rows]
    return np.concatenate(parts)


RED, BLUE, GREEN = (0, 0, 200), (200, 80, 0), (30, 200, 30)


def close(color, expected, tolerance=3):
    return all(abs(a - b) <= tolerance for a, b in zip(color.bgr, expected))


def test_a_single_color_picture_has_a_one_color_palette():
    palette = palette_colors(solid(*RED, height=20, width=30))
    assert len(palette) == 1
    assert palette[0].bgr == RED and palette[0].share == pytest.approx(100)


def test_two_colors_in_equal_halves():
    palette = palette_colors(blocks((RED, 20), (BLUE, 20)))
    assert len(palette) == 2
    assert sorted(c.bgr for c in palette) == sorted([RED, BLUE])
    assert [round(c.share) for c in palette] == [50, 50]


def test_palette_reports_each_colors_share_biggest_first():
    palette = palette_colors(blocks((RED, 60), (BLUE, 30), (GREEN, 10)))
    assert [round(c.share) for c in palette] == [60, 30, 10]
    assert close(palette[0], RED) and close(palette[1], BLUE) and close(palette[2], GREEN)


def test_shares_add_up_to_100():
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, (50, 60, 3)).astype(np.uint8)
    assert sum(c.share for c in palette_colors(image)) == pytest.approx(100)


def test_five_different_colors_give_five_tiles():
    picture = blocks(((0, 0, 255), 10), ((0, 255, 0), 10), ((255, 0, 0), 10),
                     ((0, 255, 255), 10), ((255, 255, 255), 10))
    assert len(palette_colors(picture)) == 5


def test_many_colors_are_cut_down_to_five():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (80, 80, 3)).astype(np.uint8)
    assert len(palette_colors(image)) == PALETTE_SIZE


def test_the_palette_size_can_be_changed():
    picture = blocks((RED, 10), (BLUE, 10), (GREEN, 10), ((255, 255, 255), 10))
    assert len(palette_colors(picture, count=2)) == 2
    assert len(palette_colors(picture, count=4)) == 4


def test_the_same_picture_always_gives_the_same_palette():
    rng = np.random.default_rng(10)
    image = rng.integers(0, 256, (60, 60, 3)).astype(np.uint8)
    assert palette_colors(image) == palette_colors(image)


def test_noisy_colors_still_give_their_true_color():
    rng = np.random.default_rng(11)
    picture = blocks((RED, 40), (BLUE, 40)).astype(int)
    noisy = np.clip(picture + rng.integers(-6, 7, picture.shape), 0, 255).astype(np.uint8)
    palette = palette_colors(noisy)
    assert len(palette) == 2
    for color, expected in zip(sorted(palette, key=lambda c: c.red), sorted([RED, BLUE], key=lambda c: c[2])):
        assert close(color, expected, tolerance=4)


def test_a_big_picture_is_sampled_but_the_shares_hold():
    picture = blocks((RED, 700), (BLUE, 300), width=400)         # 400,000 pixels
    assert picture.shape[0] * picture.shape[1] > PALETTE_SAMPLE * 10
    palette = palette_colors(picture)
    assert palette[0].share == pytest.approx(70, abs=2)
    assert palette[1].share == pytest.approx(30, abs=2)


def test_a_list_of_pixels_works_as_well_as_a_picture():
    picture = blocks((RED, 30), (BLUE, 30))
    assert palette_colors(picture.reshape(-1, 3)) == palette_colors(picture)


def test_pooled_pixels_from_several_frames_blend_their_shares():
    first = blocks((RED, 20)).reshape(-1, 3)                      # all red
    second = blocks((BLUE, 20)).reshape(-1, 3)                    # all blue
    palette = palette_colors(np.concatenate([first, first, first, second]))
    assert [round(c.share) for c in palette] == [75, 25]


def test_palette_color_hex_and_bgr():
    color = PaletteColor(blue=0, green=128, red=255, share=12.5)
    assert color.hex == "#ff8000"
    assert color.bgr == (0, 128, 255)


def test_palette_rejects_bad_input():
    with pytest.raises(ValueError):
        palette_colors(np.zeros((0, 3), np.uint8))
    with pytest.raises(ValueError):
        palette_colors(np.zeros((4, 4), np.uint8))               # gray, not color
    with pytest.raises(ValueError):
        palette_colors(np.zeros((4, 4, 3), np.float32))          # not 8-bit
    with pytest.raises(ValueError):
        palette_colors(solid(*RED), count=0)


def test_analyze_frame_includes_a_palette():
    stats = analyze_frame(blocks((RED, 30), (BLUE, 30)))
    assert [c.bgr for c in stats.palette] != []
    assert sum(c.share for c in stats.palette) == pytest.approx(100)


def test_the_palette_leaves_out_the_black_bars():
    picture = blocks((RED, 30), (BLUE, 30), width=60)
    stats = analyze_frame(add_bars(picture, top=20, bottom=20))
    assert len(stats.palette) == 2
    assert all(c.bgr != (0, 0, 0) for c in stats.palette)


def test_black_inside_the_picture_stays_in_the_palette():
    picture = blocks((RED, 20), ((0, 0, 0), 20), (BLUE, 20), width=60)
    palette = analyze_frame(picture).palette
    assert any(close(c, (0, 0, 0)) for c in palette)


def test_frame_stats_palette_defaults_to_nothing():
    stats = FrameStats(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, None, BlackBars(),
                       np.ones(256, np.int64))
    assert stats.palette is None


@needs_ffmpeg
def test_decoded_video_palette(tmp_path):
    clip = make_color_clip(tmp_path / "red.mp4", "red")
    palette = analyze_frame(read_frame(clip, 0).image).palette
    assert len(palette) == 1 and palette[0].share == pytest.approx(100)
    assert palette[0].red > 230 and palette[0].green < 25 and palette[0].blue < 25


def test_nearly_identical_colors_are_merged_into_one():
    # Two reds a few steps apart look the same, so they share one tile
    picture = blocks(((0, 0, 200), 20), ((0, 0, 206), 20), (BLUE, 20))
    palette = palette_colors(picture)
    assert len(palette) == 2
    assert palette[0].share == pytest.approx(200 / 3, abs=1)         # both reds together
    assert palette[0].red in range(200, 207)


def test_clearly_different_colors_are_kept_apart():
    picture = blocks(((0, 0, 200), 20), ((0, 0, 120), 20))
    assert len(palette_colors(picture)) == 2


def test_merged_shares_still_add_up_to_100():
    picture = blocks(((0, 0, 200), 20), ((0, 0, 204), 20), ((0, 0, 208), 20), (BLUE, 10))
    assert sum(c.share for c in palette_colors(picture)) == pytest.approx(100)


def test_the_merge_distance_is_a_clearly_visible_difference():
    assert 5 <= PALETTE_MERGE_DISTANCE <= 15


# ---------- vectorscope (color map) ----------

PURE_COLORS = {                      # (blue, green, red): expected (Cb, Cr)
    "red": ((0, 0, 255), (-29.2, 127.5)),
    "yellow": ((0, 255, 255), (-127.5, 11.7)),
    "green": ((0, 255, 0), (-98.3, -115.8)),
    "cyan": ((255, 255, 0), (29.2, -127.5)),
    "blue": ((255, 0, 0), (127.5, -11.7)),
    "magenta": ((255, 0, 255), (98.3, 115.8)),
}


@pytest.mark.parametrize("gray", [0, 1, 77, 128, 200, 255])
def test_gray_pixels_have_no_color_difference(gray):
    cb, cr = chroma(solid(gray, gray, gray))
    assert np.allclose(cb, 0, atol=1e-9) and np.allclose(cr, 0, atol=1e-9)


@pytest.mark.parametrize("name", list(PURE_COLORS))
def test_pure_colors_land_where_a_vectorscope_puts_them(name):
    bgr, (expected_cb, expected_cr) = PURE_COLORS[name]
    cb, cr = chroma_of_color(*bgr)
    assert cb == pytest.approx(expected_cb, abs=0.1)
    assert cr == pytest.approx(expected_cr, abs=0.1)


def test_chroma_has_one_value_per_pixel():
    cb, cr = chroma(solid(10, 20, 30, height=5, width=7))
    assert cb.shape == (5, 7) and cr.shape == (5, 7)


def test_more_blue_moves_right_and_more_red_moves_up():
    assert chroma_of_color(200, 100, 100)[0] > chroma_of_color(120, 100, 100)[0] > 0
    assert chroma_of_color(100, 100, 200)[1] > chroma_of_color(100, 100, 120)[1] > 0


def test_a_paler_color_sits_closer_to_the_middle():
    vivid = chroma_of_color(0, 0, 255)
    pale = chroma_of_color(128, 128, 255)
    assert abs(pale[1]) < abs(vivid[1]) and pale[1] > 0


def test_no_color_goes_past_the_edge_of_the_map():
    values = np.arange(0, 256, 15, dtype=np.uint8)
    blue, green, red = np.meshgrid(values, values, values)
    image = np.stack([blue, green, red], axis=-1).reshape(1, -1, 3)
    cb, cr = chroma(image)
    assert cb.min() >= -127.5 - 1e-9 and cb.max() <= 127.5 + 1e-9
    assert cr.min() >= -127.5 - 1e-9 and cr.max() <= 127.5 + 1e-9


def test_chroma_rejects_bad_images():
    with pytest.raises(ValueError):
        chroma(np.zeros((4, 4), np.uint8))
    with pytest.raises(ValueError):
        chroma(np.zeros((0, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        chroma_vectorscope(np.zeros((0, 4, 3), np.uint8))


def test_vectorscope_is_a_square_of_counts_that_adds_up():
    image = np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    counts = chroma_vectorscope(image)
    assert counts.shape == (VECTORSCOPE_SIZE, VECTORSCOPE_SIZE)
    assert counts.sum() == 30 * 40


def test_vectorscope_size_can_be_changed():
    counts = chroma_vectorscope(solid(10, 200, 30), size=32)
    assert counts.shape == (32, 32) and counts.sum() == 64


def test_vectorscope_needs_at_least_one_cell():
    with pytest.raises(ValueError):
        chroma_vectorscope(solid(1, 2, 3), size=0)


def test_a_gray_picture_is_one_dot_in_the_middle():
    counts = chroma_vectorscope(solid(90, 90, 90, height=10, width=10))
    middle = VECTORSCOPE_SIZE // 2
    assert counts[middle, middle] == 100 and counts.sum() == 100


def cell_of(counts):
    """(row, column) of the busiest cell."""
    return np.unravel_index(np.argmax(counts), counts.shape)


def test_colors_sit_in_the_right_direction_from_the_middle():
    middle = VECTORSCOPE_SIZE // 2
    red = cell_of(chroma_vectorscope(solid(0, 0, 255)))
    yellow = cell_of(chroma_vectorscope(solid(0, 255, 255)))
    blue = cell_of(chroma_vectorscope(solid(255, 0, 0)))
    cyan = cell_of(chroma_vectorscope(solid(255, 255, 0)))
    assert red[0] > middle                       # red: up (high rows)
    assert yellow[1] < middle                    # yellow: left
    assert blue[1] > middle                      # blue: right
    assert cyan[0] < middle                      # cyan: down (low rows)


def test_pure_red_and_blue_cells():
    assert cell_of(chroma_vectorscope(solid(0, 0, 255))) == (127, 49)
    assert cell_of(chroma_vectorscope(solid(255, 0, 0))) == (58, 127)


def test_two_colors_make_two_cells_with_their_own_counts():
    picture = blocks(((0, 0, 255), 10), ((255, 0, 0), 30), width=10)
    counts = chroma_vectorscope(picture)
    assert counts[127, 49] == 100
    assert counts[58, 127] == 300
    assert counts.sum() == 400


def test_the_vectorscope_of_two_frames_adds_up():
    one = chroma_vectorscope(solid(0, 0, 255, height=4, width=4))
    two = chroma_vectorscope(solid(0, 0, 255, height=6, width=6))
    assert (one + two)[127, 49] == 16 + 36


def test_the_extreme_colors_stay_inside_the_map():
    for bgr, _ in PURE_COLORS.values():
        counts = chroma_vectorscope(solid(*bgr, height=3, width=3))
        assert counts.sum() == 9


def test_analyze_frame_includes_a_vectorscope():
    stats = analyze_frame(solid(10, 200, 30, height=20, width=30))
    assert stats.vectorscope.shape == (VECTORSCOPE_SIZE, VECTORSCOPE_SIZE)
    assert stats.vectorscope.sum() == 20 * 30


def test_the_vectorscope_leaves_out_the_black_bars():
    # Black bars are gray, so counting them would pile a dot onto the middle
    picture = blocks(((0, 0, 0), 20), ((0, 0, 255), 60), ((0, 0, 0), 20), width=50)
    stats = analyze_frame(picture)
    middle = VECTORSCOPE_SIZE // 2
    assert stats.vectorscope[middle, middle] == 0
    assert stats.vectorscope.sum() == 60 * 50
    assert stats.vectorscope[127, 49] == 60 * 50


def test_frame_stats_vectorscope_defaults_to_nothing():
    stats = FrameStats(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, None, BlackBars(),
                       np.ones(256, np.int64))
    assert stats.vectorscope is None


@needs_ffmpeg
def test_decoded_video_vectorscope(tmp_path):
    clip = make_color_clip(tmp_path / "blue.mp4", "blue")
    counts = analyze_frame(read_frame(clip, 0).image).vectorscope
    row, column = cell_of(counts)
    assert counts.sum() == 320 * 240
    assert column >= 120 and abs(row - 58) <= 3