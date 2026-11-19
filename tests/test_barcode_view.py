"""Tests for the color barcode strip and its place in the main window."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from blackboard.barcode import barcode_columns
from blackboard.explanations import BARCODE_HELP, SEGMENT_HELP
from blackboard.probe import probe
from blackboard.ui.barcode_view import (
    BACKGROUND, BARCODE_HEIGHT, MARKER, BarcodeView,
)
from blackboard.ui.main_window import MainWindow

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)

BACKGROUND_RGB = (BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue())
MARKER_RGB = (MARKER.red(), MARKER.green(), MARKER.blue())

# Colors as (blue, green, red), like the frames
RED, GREEN, BLUE, WHITE = (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def render(view, width=400, height=BARCODE_HEIGHT):
    """Draw the widget and return a function: pixel(x, y) -> (R, G, B)."""
    view.resize(width, height)
    image = view.grab().toImage()

    def pixel(x, y):
        color = image.pixelColor(int(x), int(y))
        return color.red(), color.green(), color.blue()
    return pixel


def rgb(bgr):
    return bgr[2], bgr[1], bgr[0]


# ---------- the widget ----------

def test_widget_keeps_a_fixed_height_and_stretches_sideways(app):
    view = BarcodeView()
    assert view.height() == BARCODE_HEIGHT and view.minimumWidth() >= 50
    view.resize(700, 300)
    assert view.height() == BARCODE_HEIGHT


def test_empty_widget_is_all_background(app):
    pixel = render(BarcodeView())
    for point in [(0, 0), (200, 16), (399, BARCODE_HEIGHT - 1)]:
        assert pixel(*point) == BACKGROUND_RGB


def test_a_message_is_drawn_while_there_are_no_stripes(app):
    view = BarcodeView()
    view.clear("Open a video to see its color barcode")
    pixel = render(view)
    seen = {pixel(x, y) for x in range(0, 400, 2) for y in range(0, 32)}
    assert len(seen) > 1


def test_stripes_are_drawn_in_order_across_the_width(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.set_colors([RED, GREEN, BLUE, WHITE])
    pixel = render(view)                         # 100 pixels per stripe
    assert pixel(10, 5) == rgb(RED)
    assert pixel(110, 5) == rgb(GREEN)
    assert pixel(210, 5) == rgb(BLUE)
    assert pixel(310, 5) == rgb(WHITE)


def test_stripes_fill_the_whole_height(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=2)
    view.set_colors([RED, BLUE])
    pixel = render(view)
    assert pixel(50, 0) == rgb(RED) and pixel(50, BARCODE_HEIGHT - 1) == rgb(RED)


def test_stripes_not_measured_yet_stay_empty(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.set_colors([RED, GREEN])                # half done
    pixel = render(view)
    assert pixel(150, 5) == rgb(GREEN)
    assert pixel(250, 5) == BACKGROUND_RGB
    assert pixel(390, 5) == BACKGROUND_RGB


def test_the_barcode_fills_in_as_more_stripes_arrive(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.set_colors([RED])
    assert render(view)(150, 5) == BACKGROUND_RGB
    view.set_colors([RED, GREEN])
    assert render(view)(150, 5) == rgb(GREEN)


def test_more_stripes_than_pixels_pick_the_stripe_under_each_pixel(app):
    view = BarcodeView()
    view.begin(first=0, last=9999, total=400)
    view.set_colors([(i % 256, 0, 0) for i in range(400)])
    row = view.stripe_row(100)                   # four stripes per pixel
    assert [int(row[x][0]) for x in (0, 1, 2)] == [2, 6, 10]


def test_stripe_row_has_one_color_per_pixel(app):
    view = BarcodeView()
    view.begin(first=0, last=99, total=5)
    view.set_colors([RED] * 5)
    row = view.stripe_row(37)
    assert row.shape == (37, 3) and row.dtype == np.uint8


def test_clearing_removes_the_stripes_and_the_marker(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=2)
    view.set_colors([RED, BLUE])
    view.set_position(10)
    view.clear()
    pixel = render(view)
    assert pixel(50, 5) == BACKGROUND_RGB
    assert view.marker_x() is None and view.span == 0


def test_set_colors_none_clears_the_stripes(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=2)
    view.set_colors([RED, BLUE])
    view.set_colors(None)
    assert render(view)(50, 5) == BACKGROUND_RGB


# ---------- the marker ----------

def test_marker_is_at_the_current_moment(app):
    view = BarcodeView()
    view.begin(first=0, last=99, total=4)
    view.set_colors([RED, GREEN, BLUE, WHITE])
    view.set_position(50)
    pixel = render(view)                         # 400 pixels wide
    assert view.marker_x() == int(50.5 / 100 * 400)
    assert pixel(view.marker_x(), 10) == MARKER_RGB


def test_marker_moves_with_the_frame(app):
    view = BarcodeView()
    view.begin(first=0, last=99, total=4)
    view.resize(400, BARCODE_HEIGHT)
    view.set_position(10)
    first = view.marker_x()
    view.set_position(80)
    assert view.marker_x() > first


def test_marker_starts_at_the_left_and_ends_at_the_right(app):
    view = BarcodeView()
    view.begin(first=0, last=99, total=4)
    view.resize(400, BARCODE_HEIGHT)
    view.set_position(0)
    assert view.marker_x() <= 3
    view.set_position(99)
    assert view.marker_x() >= 396


def test_there_is_no_marker_without_a_position(app):
    view = BarcodeView()
    view.begin(first=0, last=99, total=4)
    assert view.marker_x() is None
    pixel = render(view)
    assert all(pixel(x, 5) != MARKER_RGB for x in range(400))


def test_marker_has_a_dark_edge_so_it_shows_on_white(app):
    view = BarcodeView()
    view.begin(first=0, last=99, total=1)
    view.set_colors([WHITE])
    view.set_position(50)
    pixel = render(view)
    x = view.marker_x()
    assert pixel(x, 10) == MARKER_RGB
    assert pixel(x - 1, 10) != rgb(WHITE) and pixel(x + 1, 10) != rgb(WHITE)


# ---------- the mouse ----------

def test_frame_at_maps_a_pixel_to_a_frame(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.resize(400, BARCODE_HEIGHT)
    assert view.frame_at(0) == 0
    assert view.frame_at(200) == 500
    assert view.frame_at(399.9) == 999


def test_frame_at_stays_inside_the_video(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.resize(400, BARCODE_HEIGHT)
    assert view.frame_at(-50) == 0
    assert view.frame_at(5000) == 999


def test_frame_at_has_no_answer_without_a_video(app):
    view = BarcodeView()
    view.resize(400, BARCODE_HEIGHT)
    assert view.frame_at(100) is None


def test_clicking_asks_to_jump_to_that_moment(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.resize(400, BARCODE_HEIGHT)
    asked = []
    view.seek_requested.connect(asked.append)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(100, 10))
    assert asked == [250]


def test_dragging_keeps_asking(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.resize(400, BARCODE_HEIGHT)
    asked = []
    view.seek_requested.connect(asked.append)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(40, 10))
    QTest.mouseMove(view, QPoint(120, 10))
    QTest.mouseMove(view, QPoint(300, 10))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(300, 10))
    assert asked[0] == 100 and asked[-1] == 750 and len(asked) >= 2


def test_clicking_does_nothing_without_a_video(app):
    view = BarcodeView()
    view.resize(400, BARCODE_HEIGHT)
    asked = []
    view.seek_requested.connect(asked.append)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(100, 10))
    assert asked == []


def test_moving_the_mouse_without_a_button_does_nothing(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    view.resize(400, BARCODE_HEIGHT)
    view.setMouseTracking(True)
    asked = []
    view.seek_requested.connect(asked.append)
    QTest.mouseMove(view, QPoint(200, 10))
    assert asked == []


def test_the_pointer_shows_the_strip_can_be_clicked(app):
    view = BarcodeView()
    view.begin(first=0, last=999, total=4)
    assert view.cursor().shape() == Qt.CursorShape.PointingHandCursor
    view.clear()
    assert view.cursor().shape() != Qt.CursorShape.PointingHandCursor


# ---------- a segment that does not start at frame 0 ----------

def test_the_marker_is_placed_inside_the_segment(app):
    view = BarcodeView()
    view.begin(first=200, last=299, total=4)
    view.resize(400, BARCODE_HEIGHT)
    view.set_position(250)
    assert view.marker_x() == int(50.5 / 100 * 400)


def test_there_is_no_marker_outside_the_segment(app):
    view = BarcodeView()
    view.begin(first=200, last=299, total=4)
    view.resize(400, BARCODE_HEIGHT)
    for outside in (0, 199, 300, 5000):
        view.set_position(outside)
        assert view.marker_x() is None
    pixel = render(view)
    assert all(pixel(x, 5) != MARKER_RGB for x in range(400))


def test_the_marker_shows_on_the_first_and_last_frame_of_the_segment(app):
    view = BarcodeView()
    view.begin(first=200, last=299, total=4)
    view.resize(400, BARCODE_HEIGHT)
    view.set_position(200)
    assert view.marker_x() <= 3
    view.set_position(299)
    assert view.marker_x() >= 396


def test_frame_at_maps_a_pixel_into_the_segment(app):
    view = BarcodeView()
    view.begin(first=200, last=1199, total=4)
    view.resize(400, BARCODE_HEIGHT)
    assert view.frame_at(0) == 200
    assert view.frame_at(200) == 700
    assert view.frame_at(399.9) == 1199
    assert view.frame_at(-50) == 200 and view.frame_at(5000) == 1199


def test_clicking_jumps_to_a_frame_of_the_segment(app):
    view = BarcodeView()
    view.begin(first=200, last=1199, total=4)
    view.resize(400, BARCODE_HEIGHT)
    asked = []
    view.seek_requested.connect(asked.append)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(100, 10))
    assert asked == [450]


def test_the_span_is_the_number_of_frames_in_the_segment(app):
    view = BarcodeView()
    assert view.span == 0
    view.begin(first=10, last=19, total=10)
    assert view.span == 10
    view.begin(first=7, last=7, total=1)
    assert view.span == 1
    view.clear()
    assert view.span == 0


def test_a_one_frame_segment_is_one_stripe_and_one_click_target(app):
    view = BarcodeView()
    view.begin(first=42, last=42, total=1)
    view.set_colors([RED])
    view.resize(400, BARCODE_HEIGHT)
    assert render(view)(200, 5) == rgb(RED)
    assert view.frame_at(0) == 42 and view.frame_at(399) == 42


# ---------- in the main window ----------

def open_clip(window, path):
    window.current_path = str(path)
    window.setup_preview(probe(path))


def test_the_window_has_a_band_under_the_slider(app):
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    app.processEvents()
    slider, band = window.frame_slider, window.segment_band
    assert band.mapTo(window, QPoint(0, 0)).y() >= slider.mapTo(window, QPoint(0, 0)).y() + slider.height()
    assert band.x() == slider.x() and band.width() == slider.width()


def test_the_barcode_sits_below_the_segment_controls(app):
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    app.processEvents()
    y = lambda widget: widget.mapTo(window, QPoint(0, 0)).y()
    assert y(window.segment_picker) > y(window.frame_slider)
    assert y(window.barcode_view) > y(window.segment_picker)
    # the button that starts the analysis is in the same card, beside its strip
    assert window.strips_card.isAncestorOf(window.analyze_button)
    assert abs(y(window.analyze_button) - y(window.barcode_view)) < 20


def test_everything_fits_in_the_window(app):
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    app.processEvents()
    bottom = window.scope_stack.mapTo(window, QPoint(0, 0)).y() + window.scope_stack.height()
    assert bottom <= window.height()
    assert window.minimumSizeHint().height() <= 900


def test_the_barcode_has_an_explanation(app):
    window = MainWindow()
    assert window.barcode_view.toolTip().replace("\n", " ") == BARCODE_HELP
    assert window.analyze_button.toolTip().replace("\n", " ") == BARCODE_HELP


def test_the_segment_controls_have_an_explanation(app):
    window = MainWindow()
    assert window.segment_picker.toolTip().replace("\n", " ") == SEGMENT_HELP
    assert window.segment_band.toolTip().replace("\n", " ") == SEGMENT_HELP


def test_the_barcode_starts_empty_with_a_hint(app):
    window = MainWindow()
    assert window.barcode_view.colors is None
    assert window.barcode_view.span == 0
    assert "barcode" in window.barcode_view.message.lower()


def test_nothing_can_be_analyzed_before_a_video_is_open(app):
    window = MainWindow()
    assert not window.analyze_button.isEnabled()
    assert not window.segment_picker.isEnabled()
    window.on_analyze_clicked()
    assert window.barcode_thread is None and not window.barcode_running


def test_clicking_the_barcode_moves_the_slider(app):
    window = MainWindow()
    window.frame_slider.setRange(0, 999)
    window.frame_spin.setRange(0, 999)
    window.info = object()                       # something is open
    window.barcode_view.begin(first=0, last=999, total=4)
    window.barcode_view.resize(400, BARCODE_HEIGHT)
    window.barcode_view.seek_requested.emit(250)
    assert window.frame_slider.value() == 250 and window.frame_spin.value() == 250


def test_a_late_result_for_another_run_is_ignored(app):
    window = MainWindow()
    window.barcode_generation = 5
    window.barcode_view.begin(first=0, last=99, total=2)
    window.on_barcode_updated(4, np.array([RED, BLUE], np.uint8))
    window.on_barcode_done(4, np.array([RED, BLUE], np.uint8))
    assert window.barcode_view.colors is None
    window.on_barcode_updated(5, np.array([RED], np.uint8))
    assert len(window.barcode_view.colors) == 1


def test_stopping_makes_every_earlier_result_late(app):
    window = MainWindow()
    generation = window.barcode_generation
    window.stop_barcode()
    assert window.barcode_generation == generation + 1


def test_a_failure_shows_a_message_and_keeps_the_marker_working(app):
    window = MainWindow()
    window.barcode_view.begin(first=0, last=99, total=2)
    window.on_barcode_failed(window.barcode_generation, "boom")
    assert "not available" in window.barcode_view.message
    window.barcode_view.set_position(5)
    assert window.barcode_view.marker_x() is not None


def test_a_failure_for_another_run_is_ignored(app):
    window = MainWindow()
    window.barcode_view.begin(first=0, last=99, total=2, message="Building")
    window.on_barcode_failed(window.barcode_generation - 1, "boom")
    assert window.barcode_view.message == "Building"


def test_clearing_the_preview_clears_the_barcode(app):
    window = MainWindow()
    window.barcode_view.begin(first=0, last=99, total=2)
    window.barcode_view.set_colors([RED])
    generation = window.barcode_generation
    window.clear_preview()
    assert window.barcode_view.colors is None and window.barcode_view.span == 0
    assert window.barcode_generation > generation


def test_the_slider_moves_the_marker(app):
    window = MainWindow()
    window.info = type("Info", (), {"fps": 30.0})()
    window.frame_slider.setRange(0, 99)
    window.frame_spin.setRange(0, 99)
    window.barcode_view.begin(first=0, last=99, total=4)
    window.barcode_view.resize(400, BARCODE_HEIGHT)
    window.frame_spin.setValue(60)
    assert window.barcode_view.position == 60
    assert window.segment_band.position == 60


def wait_for(window, app, seconds=20):
    """Let the background threads finish and deliver their results."""
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
def red_then_blue(tmp_path_factory):
    path = tmp_path_factory.mktemp("clips") / "red_blue.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=160x120:d=2:r=30",
         "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=2:r=30",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_opening_a_video_does_not_analyze_anything(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    wait_for(window, app)
    assert window.barcode_thread is None and not window.barcode_running
    assert window.barcode_view.colors is None
    assert "segment" in window.barcode_view.message.lower()
    assert window.analyze_button.isEnabled()
    assert window.analyze_button.text() == "Analyze color barcode"


@needs_ffmpeg
def test_opening_a_video_selects_all_of_it(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    assert window.segment_picker.segment() == (0, 119)
    assert window.segment_picker.isEnabled()
    assert (window.segment_band.first, window.segment_band.last) == (0, 119)
    assert window.segment_band.total == 120
    wait_for(window, app)


@needs_ffmpeg
def test_pressing_analyze_builds_the_barcode_of_the_whole_video(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.analyze_button.click()
    assert window.barcode_running and window.analyze_button.text() == "Cancel"
    wait_for(window, app)
    colors = window.barcode_view.colors
    assert len(colors) == barcode_columns(0, 119)
    assert colors[0][2] > 200 and colors[-1][0] > 200            # red, then blue
    image = window.barcode_view.grab().toImage()                 # as drawn on screen
    left = image.pixelColor(int(image.width() * 0.1), 5)
    right = image.pixelColor(int(image.width() * 0.9), 5)
    assert left.red() > 200 and right.blue() > 200


@needs_ffmpeg
def test_pressing_analyze_uses_only_the_chosen_segment(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.segment_picker.start_spin.setValue(65)        # inside the blue part
    window.segment_picker.end_spin.setValue(110)
    window.analyze_button.click()
    wait_for(window, app)
    colors = window.barcode_view.colors
    assert len(colors) == barcode_columns(65, 110)
    assert all(stripe[0] > 200 and stripe[2] < 60 for stripe in colors)   # all blue
    assert (window.barcode_view.first_frame, window.barcode_view.last_frame) == (65, 110)


@needs_ffmpeg
def test_the_segment_can_be_set_from_the_current_frame(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.frame_spin.setValue(30)
    window.segment_picker.start_button.click()
    window.frame_spin.setValue(90)
    window.segment_picker.end_button.click()
    assert window.segment_picker.segment() == (30, 90)
    assert (window.segment_band.first, window.segment_band.last) == (30, 90)
    wait_for(window, app)


@needs_ffmpeg
def test_setting_the_start_past_the_end_moves_the_end_along(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.frame_spin.setValue(20)
    window.segment_picker.end_button.click()            # end = 20
    window.frame_spin.setValue(50)
    window.segment_picker.start_button.click()          # start = 50, past the end
    assert window.segment_picker.segment() == (50, 50)
    wait_for(window, app)


@needs_ffmpeg
def test_the_controls_are_locked_while_analyzing(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.analyze_button.click()
    assert not window.segment_picker.isEnabled()
    wait_for(window, app)
    assert window.segment_picker.isEnabled()
    assert window.analyze_button.text() == "Analyze color barcode"
    assert not window.barcode_running
    assert "frames 0 to 119" in window.barcode_status.text()


@needs_ffmpeg
def test_cancel_drops_the_half_built_barcode(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.analyze_button.click()
    assert window.analyze_button.text() == "Cancel"
    window.analyze_button.click()                       # cancel
    assert not window.barcode_running
    assert window.analyze_button.text() == "Analyze color barcode"
    assert window.segment_picker.isEnabled()
    assert window.barcode_view.colors is None
    assert "Cancelled" in window.barcode_status.text()
    wait_for(window, app)
    assert window.barcode_view.colors is None           # late results ignored


@needs_ffmpeg
def test_cancel_clears_stripes_that_had_already_arrived(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.analyze_button.click()
    window.barcode_view.set_colors([RED, GREEN])        # some stripes are in
    window.analyze_button.click()                       # cancel
    assert window.barcode_view.colors is None
    assert window.barcode_view.span == 0
    wait_for(window, app)


@needs_ffmpeg
def test_analyzing_again_replaces_the_old_barcode(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.segment_picker.set_segment(0, 50)
    window.analyze_button.click()
    wait_for(window, app)
    assert window.barcode_view.last_frame == 50
    window.segment_picker.set_segment(70, 119)
    window.analyze_button.click()
    wait_for(window, app)
    assert (window.barcode_view.first_frame, window.barcode_view.last_frame) == (70, 119)
    assert all(stripe[0] > 200 for stripe in window.barcode_view.colors)


@needs_ffmpeg
def test_the_barcode_keeps_its_segment_when_the_boxes_change_later(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.segment_picker.set_segment(70, 119)
    window.analyze_button.click()
    wait_for(window, app)
    window.segment_picker.start_spin.setValue(0)         # picks a new segment
    assert (window.barcode_view.first_frame, window.barcode_view.last_frame) == (70, 119)
    assert window.barcode_view.colors is not None


@needs_ffmpeg
def test_opening_another_video_drops_the_old_barcode(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.analyze_button.click()
    window.clear_preview()                      # what choose_file does first
    wait_for(window, app)
    assert window.barcode_view.colors is None
    assert not window.barcode_running
    assert not window.analyze_button.isEnabled()
    assert not window.segment_picker.isEnabled()
    assert window.segment_band.total == 0


@needs_ffmpeg
def test_closing_the_window_does_not_wait_for_a_whole_barcode(app, red_then_blue):
    window = MainWindow()
    open_clip(window, red_then_blue)
    window.analyze_button.click()
    started = time.time()
    window.close()
    assert time.time() - started < 10
    assert window.barcode_thread is None or window.barcode_thread.isFinished()