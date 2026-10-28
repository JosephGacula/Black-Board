"""Tests for the frame measurements in analyze.py.

Most tests use tiny made-up images whose right answer we know. The last two
make real videos with FFmpeg to check the whole path from file to number.
"""
import shutil
import subprocess

import numpy as np
import pytest

from blackboard.analyze import (
    ANALYSIS_WIDTH, FrameStats, analyze_frame, brightness_percent, luma,
    prepare,
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
    # A smooth left-to-right gradient, big enough to be shrunk
    ramp = np.linspace(0, 255, 1920).astype(np.uint8)
    image = np.repeat(ramp[None, :, None], 1080, axis=0).repeat(3, axis=2)
    full = brightness_percent(luma(image))
    assert analyze_frame(image).brightness == pytest.approx(full, abs=0.5)


def test_analyze_frame_returns_stats():
    stats = analyze_frame(solid(255, 255, 255))
    assert isinstance(stats, FrameStats)
    assert stats.brightness == pytest.approx(100)


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