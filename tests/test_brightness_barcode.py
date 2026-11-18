"""Tests for the brightness barcode in the main window."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from blackboard.explanations import BRIGHTNESS_BARCODE_HELP
from blackboard.probe import probe
from blackboard.ui.main_window import MainWindow

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)

RED, BLUE = (0, 0, 255), (255, 0, 0)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(window, app, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        threads = [window.barcode_thread, window.frame_thread]
        if all(t is None or t.isFinished() for t in threads):
            break
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture(scope="module")
def dark_then_bright(tmp_path_factory):
    """60 dark gray frames, then 60 light gray frames."""
    path = tmp_path_factory.mktemp("clips") / "dark_bright.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=0x303030:s=160x120:d=2:r=30",
         "-f", "lavfi", "-i", "color=c=0xe0e0e0:s=160x120:d=2:r=30",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def test_the_window_has_a_brightness_strip_under_the_color_one(app):
    window = MainWindow()
    window.resize(1000, 860)
    window.show()
    app.processEvents()
    y = lambda widget: widget.mapTo(window, QPoint(0, 0)).y()
    assert y(window.brightness_view) >= y(window.barcode_view) + window.barcode_view.height()
    assert window.brightness_view.width() == window.barcode_view.width()


def test_everything_still_fits_in_the_window(app):
    window = MainWindow()
    window.resize(1000, 860)
    window.show()
    app.processEvents()
    bottom = window.scope_stack.mapTo(window, QPoint(0, 0)).y() + window.scope_stack.height()
    assert bottom <= window.height()


def test_the_brightness_strip_has_its_own_explanation(app):
    window = MainWindow()
    assert window.brightness_view.toolTip().replace("\n", " ") == BRIGHTNESS_BARCODE_HELP
    assert window.brightness_view.toolTip() != window.barcode_view.toolTip()


def test_it_starts_empty_with_a_hint(app):
    window = MainWindow()
    assert window.brightness_view.colors is None
    assert "brightness" in window.brightness_view.message.lower()


def test_stripes_arriving_show_up_as_gray_in_both_strips(app):
    window = MainWindow()
    window.barcode_view.begin(0, 99, 2)
    window.brightness_view.begin(0, 99, 2)
    window.on_barcode_updated(window.barcode_generation, np.array([RED, BLUE], np.uint8))
    assert len(window.barcode_view.colors) == 2
    gray = window.brightness_view.colors
    assert gray.shape == (2, 3)
    assert all(len(set(int(v) for v in row)) == 1 for row in gray)
    assert gray[0][0] > gray[1][0]                      # red is brighter than blue


def test_a_late_result_does_not_change_the_brightness_strip(app):
    window = MainWindow()
    window.brightness_view.begin(0, 99, 2)
    window.on_barcode_done(window.barcode_generation - 1, np.array([RED], np.uint8))
    assert window.brightness_view.colors is None


def test_a_failure_is_shown_on_both_strips(app):
    window = MainWindow()
    window.on_barcode_failed(window.barcode_generation, "boom")
    assert "not available" in window.brightness_view.message


def test_the_marker_follows_the_slider_on_both_strips(app):
    window = MainWindow()
    window.info = type("Info", (), {"fps": 30.0})()
    window.frame_slider.setRange(0, 99)
    window.frame_spin.setRange(0, 99)
    window.brightness_view.begin(0, 99, 4)
    window.frame_spin.setValue(40)
    assert window.brightness_view.position == 40


def test_clicking_the_brightness_strip_moves_the_slider(app):
    window = MainWindow()
    window.frame_slider.setRange(0, 999)
    window.frame_spin.setRange(0, 999)
    window.info = object()
    window.brightness_view.seek_requested.emit(321)
    assert window.frame_spin.value() == 321


@needs_ffmpeg
def test_analyzing_builds_both_barcodes_from_the_same_segment(app, dark_then_bright):
    window = MainWindow()
    window.current_path = str(dark_then_bright)
    window.setup_preview(probe(dark_then_bright))
    window.segment_picker.set_segment(10, 109)
    window.analyze_button.click()
    wait_for(window, app)
    color, gray = window.barcode_view, window.brightness_view
    assert len(gray.colors) == len(color.colors)
    assert (gray.first_frame, gray.last_frame) == (10, 109)
    assert gray.colors[0][0] < 90 and gray.colors[-1][0] > 190      # dark, then bright
    assert list(gray.colors[:, 0]) == sorted(gray.colors[:, 0])     # never gets darker


@needs_ffmpeg
def test_cancel_and_clear_empty_the_brightness_strip(app, dark_then_bright):
    window = MainWindow()
    window.current_path = str(dark_then_bright)
    window.setup_preview(probe(dark_then_bright))
    window.analyze_button.click()
    window.brightness_view.set_colors([(5, 5, 5)])
    window.analyze_button.click()                        # cancel
    assert window.brightness_view.colors is None and window.brightness_view.span == 0
    wait_for(window, app)
    window.clear_preview()
    assert window.brightness_view.colors is None