"""Tests for the "Show exposure warnings" checkbox and the preview it controls."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.analyze import (
    EXPOSURE_CRUSHED_BGR, exposure_overlay,
)
from blackboard.ui.main_window import (
    PREVIEW_H, PREVIEW_W, MainWindow, bgr_hex, frame_to_qimage,
    shrink_for_preview,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def picture_with_black_patch():
    image = np.full((120, 200, 3), 128, np.uint8)
    image[40:80, 80:120] = 0                     # black patch in the middle
    return image


def load(window, image):
    """Hand the window a frame the way the background thread would."""
    window.current_path = "video.mp4"
    window.on_frame_loaded(
        "video.mp4", 0, frame_to_qimage(image), None,
        frame_to_qimage(exposure_overlay(image)))


def center_color(window):
    """The preview's center pixel as (blue, green, red)."""
    pixmap = window.preview.pixmap().toImage()
    color = pixmap.pixelColor(pixmap.width() // 2, pixmap.height() // 2)
    return color.blue(), color.green(), color.red()


# ---------- shrinking and colors ----------

def test_shrink_for_preview_fits_the_box_and_never_enlarges():
    big = np.zeros((2160, 3840, 3), np.uint8)
    small = shrink_for_preview(big)
    assert small.shape[1] <= PREVIEW_W and small.shape[0] <= PREVIEW_H
    tiny = np.zeros((50, 80, 3), np.uint8)
    assert shrink_for_preview(tiny).shape == tiny.shape


def test_bgr_hex_swaps_the_order_for_screen_colors():
    assert bgr_hex((255, 0, 0)) == "#0000ff"         # blue
    assert bgr_hex((0, 0, 255)) == "#ff0000"         # red
    assert bgr_hex((255, 110, 30)) == "#1e6eff"


# ---------- the checkbox ----------

def test_checkbox_starts_off_and_is_disabled_without_a_video(app):
    window = MainWindow()
    assert not window.exposure_check.isChecked()
    assert not window.exposure_check.isEnabled()
    assert window.exposure_check.toolTip()


def test_picture_shows_normally_while_unchecked(app):
    window = MainWindow()
    load(window, picture_with_black_patch())
    assert center_color(window) == (0, 0, 0)         # the black patch, not painted


def test_ticking_the_box_paints_the_warnings(app):
    window = MainWindow()
    load(window, picture_with_black_patch())
    window.exposure_check.setChecked(True)
    assert center_color(window) == EXPOSURE_CRUSHED_BGR


def test_unticking_goes_back_to_the_normal_picture(app):
    window = MainWindow()
    load(window, picture_with_black_patch())
    window.exposure_check.setChecked(True)
    window.exposure_check.setChecked(False)
    assert center_color(window) == (0, 0, 0)


def test_the_choice_stays_when_the_next_frame_loads(app):
    window = MainWindow()
    load(window, picture_with_black_patch())
    window.exposure_check.setChecked(True)
    load(window, picture_with_black_patch())
    assert center_color(window) == EXPOSURE_CRUSHED_BGR


def test_legend_shows_only_while_ticked_and_a_picture_is_there(app):
    window = MainWindow()
    window.show()
    window.exposure_check.setChecked(True)
    assert not window.exposure_legend.isVisible()    # no picture yet
    load(window, picture_with_black_patch())
    window.show_current_picture()
    assert window.exposure_legend.isVisible()
    window.exposure_check.setChecked(False)
    assert not window.exposure_legend.isVisible()


def test_legend_names_both_colors(app):
    window = MainWindow()
    text = window.exposure_legend.text()
    assert "crushed blacks" in text and "blown highlights" in text
    assert bgr_hex(EXPOSURE_CRUSHED_BGR) in text


def test_the_normal_picture_is_shown_if_the_overlay_failed(app):
    window = MainWindow()
    window.current_path = "video.mp4"
    window.on_frame_loaded("video.mp4", 0, frame_to_qimage(picture_with_black_patch()), None, None)
    window.exposure_check.setChecked(True)
    assert center_color(window) == (0, 0, 0)


def test_old_four_argument_calls_still_work(app):
    window = MainWindow()
    window.current_path = "video.mp4"
    window.on_frame_loaded("video.mp4", 0, frame_to_qimage(picture_with_black_patch()), None)
    assert window.preview.pixmap() is not None


def test_clearing_the_preview_resets_the_checkbox_state(app):
    window = MainWindow()
    window.show()
    load(window, picture_with_black_patch())
    window.exposure_check.setChecked(True)
    window.clear_preview()
    assert not window.exposure_check.isEnabled()
    assert not window.exposure_legend.isVisible()
    assert window.normal_picture is None and window.warning_picture is None


def test_a_late_result_for_another_video_is_ignored(app):
    window = MainWindow()
    window.current_path = "video.mp4"
    window.on_frame_loaded("other.mp4", 0, frame_to_qimage(picture_with_black_patch()), None, None)
    assert window.normal_picture is None