"""Tests for the waveform chart and the drop-down that chooses it."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.analyze import BlackBars, FrameStats, luma_waveform
from blackboard.ui.main_window import SCOPE_NAMES, MainWindow
from blackboard.ui.waveform_view import (
    FULL_SHARE, GUIDES, TRACE, WaveformView, fit_axis, waveform_intensity,
    waveform_picture,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def counts_with(level, slices=10, pixels=100):
    """Waveform counts where every slice has all its pixels at one level."""
    counts = np.zeros((256, slices), np.int64)
    counts[level] = pixels
    return counts


def render(view, width=200, height=92):
    """Draw the widget and return it as a numpy array of (R, G, B)."""
    view.resize(width, height)
    image = view.grab().toImage()
    out = np.zeros((image.height(), image.width(), 3), np.uint8)
    for y in range(image.height()):
        for x in range(image.width()):
            c = image.pixelColor(x, y)
            out[y, x] = (c.red(), c.green(), c.blue())
    return out


# ---------- waveform_intensity ----------

def test_intensity_has_the_same_shape_and_is_8_bit():
    out = waveform_intensity(counts_with(100))
    assert out.shape == (256, 10) and out.dtype == np.uint8


def test_a_level_holding_all_the_pixels_is_full_strength():
    assert waveform_intensity(counts_with(100))[100].max() == 255


def test_a_level_at_the_full_share_is_full_strength():
    counts = np.zeros((256, 1), np.int64)
    counts[50] = int(1000 * FULL_SHARE)
    counts[200] = 1000 - counts[50]
    assert waveform_intensity(counts)[50, 0] == 255


def test_rarer_levels_are_fainter_but_still_show():
    counts = np.zeros((256, 1), np.int64)
    counts[50] = 1                       # one pixel in a thousand
    counts[200] = 999
    faint = waveform_intensity(counts)[50, 0]
    assert 0 < faint < 255


def test_strength_is_eased_with_a_square_root():
    counts = np.zeros((256, 1), np.int64)
    counts[10] = int(2000 * FULL_SHARE / 4)          # a quarter of the full share
    counts[200] = 2000 - counts[10]
    assert waveform_intensity(counts)[10, 0] == pytest.approx(127.5, abs=2)


def test_every_slice_is_judged_on_its_own_pixel_count():
    # Same share of pixels, very different counts: same brightness
    small = np.zeros((256, 1), np.int64); small[30] = 10
    big = np.zeros((256, 1), np.int64); big[30] = 10000
    assert waveform_intensity(small)[30, 0] == waveform_intensity(big)[30, 0]


def test_an_empty_slice_is_dark_and_does_not_divide_by_zero():
    counts = counts_with(100)
    counts[:, 3] = 0
    out = waveform_intensity(counts)
    assert (out[:, 3] == 0).all()


# ---------- fit_axis ----------

def test_fit_axis_to_the_same_size_changes_nothing():
    values = np.arange(12).reshape(3, 4)
    assert (fit_axis(values, 3, 0) == values).all()


def test_fit_axis_stretching_repeats_entries():
    values = np.array([[1], [2]])
    assert fit_axis(values, 4, 0).ravel().tolist() == [1, 1, 2, 2]


def test_fit_axis_squeezing_keeps_the_brightest():
    values = np.array([0, 0, 255, 0, 0, 0, 0, 7]).reshape(8, 1)
    out = fit_axis(values, 4, 0).ravel()
    assert out.tolist() == [0, 255, 0, 7]


def test_a_thin_trace_survives_being_squeezed():
    column = np.zeros((256, 1), np.uint8)
    column[100] = 255
    assert fit_axis(column, 92, 0).max() == 255


@pytest.mark.parametrize("axis, size", [(0, 92), (0, 300), (1, 200), (1, 405), (1, 320)])
def test_fit_axis_gives_the_requested_size(axis, size):
    values = np.ones((256, 320), np.uint8)
    assert fit_axis(values, size, axis).shape[axis] == size


# ---------- waveform_picture ----------

def test_picture_has_the_requested_size_and_three_colors():
    picture = waveform_picture(counts_with(100), 200, 92)
    assert picture.shape == (92, 200, 3) and picture.dtype == np.uint8


def test_white_is_at_the_top_and_black_at_the_bottom():
    white = waveform_picture(counts_with(255), 40, 92)
    black = waveform_picture(counts_with(0), 40, 92)
    assert white[0].max() > 0 and white[-1].max() == 0
    assert black[-1].max() > 0 and black[0].max() == 0


def test_picture_uses_the_trace_color_at_full_strength():
    picture = waveform_picture(counts_with(255), 40, 92)
    assert tuple(picture[0, 0]) == TRACE


def test_a_mid_gray_trace_sits_half_way_up():
    picture = waveform_picture(counts_with(128), 40, 92)
    rows = np.where(picture.max(axis=(1, 2)) > 0)[0]
    assert abs(rows.mean() - 91 / 2) <= 2


def test_a_thin_trace_stays_bright_in_a_short_picture():
    picture = waveform_picture(counts_with(100), 40, 60)
    assert picture.max() == max(TRACE)


# ---------- the widget ----------

def test_widget_keeps_a_fixed_height_and_stretches_sideways(app):
    view = WaveformView()
    assert view.height() == 92 and view.minimumWidth() >= 100
    view.resize(500, 300)
    assert view.height() == 92


def test_empty_widget_shows_only_the_background_and_guides(app):
    view = WaveformView()
    image = render(view)
    guide = (GUIDES.red(), GUIDES.green(), GUIDES.blue())
    background = (0x11, 0x11, 0x11)
    guide_rows = [round((1 - level) * 91) for level in (0, 0.25, 0.5, 0.75, 1.0)]
    for y in guide_rows:
        assert tuple(image[y, 100]) == guide                 # 0, 25, 50, 75 and 100%
    for y in (5, 15, 30, 55, 75):
        assert tuple(image[y, 100]) == background            # between the guides


def test_widget_draws_a_white_trace_at_the_top(app):
    view = WaveformView()
    view.set_waveform(counts_with(255, slices=50))
    image = render(view)
    assert image[0, 100].max() > 100 and image[60, 100].max() < 60


def test_widget_draws_a_black_trace_at_the_bottom(app):
    view = WaveformView()
    view.set_waveform(counts_with(0, slices=50))
    image = render(view)
    assert image[91, 100].max() > 100 and image[30, 100].max() < 60


def test_a_trace_shows_through_the_guide_lines(app):
    view = WaveformView()
    view.set_waveform(counts_with(255, slices=50))
    image = render(view)
    assert image[0, 100, 1] >= image[3, 100, 1]


def test_clearing_removes_the_trace(app):
    view = WaveformView()
    view.set_waveform(counts_with(255, slices=50))
    view.set_waveform(None)
    image = render(view)
    assert image[1, 100].max() < 60


def test_left_and_right_of_the_picture_stay_on_their_own_sides(app):
    gray = np.zeros((20, 100), np.uint8)
    gray[:, 50:] = 255                      # right half white, left half black
    view = WaveformView()
    view.set_waveform(luma_waveform(gray))
    image = render(view, width=200)
    assert image[0, 150].max() > 100 and image[0, 50].max() < 60       # white: top right
    assert image[91, 50].max() > 100 and image[91, 150].max() < 60     # black: bottom left


# ---------- the drop-down in the window ----------

def stats_with_waveform(waveform):
    return FrameStats(50.0, 50.0, 0.0, 0.0, 30.0, 40.0, 30.0, 60.0, None,
                      BlackBars(), np.ones(256, np.int64), waveform)


def test_the_histogram_is_shown_first(app):
    window = MainWindow()
    assert [window.scope_selector.itemText(i) for i in range(window.scope_selector.count())] == SCOPE_NAMES
    assert window.scope_selector.currentIndex() == 0
    assert window.scope_stack.currentWidget() is window.histogram_view


def test_picking_waveform_shows_the_waveform(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(1)
    assert window.scope_stack.currentWidget() is window.waveform_view


def test_picking_histogram_again_goes_back(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(1)
    window.scope_selector.setCurrentIndex(0)
    assert window.scope_stack.currentWidget() is window.histogram_view


def test_both_charts_get_every_frames_measurements(app):
    window = MainWindow()
    counts = counts_with(200, slices=20)
    window.show_analysis(stats_with_waveform(counts))
    assert (window.waveform_view.counts == counts).all()
    assert window.histogram_view.histogram is not None


def test_the_choice_stays_when_the_next_frame_arrives(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(1)
    window.show_analysis(stats_with_waveform(counts_with(200)))
    window.show_analysis(stats_with_waveform(counts_with(50)))
    assert window.scope_stack.currentWidget() is window.waveform_view


def test_clearing_the_analysis_clears_both_charts(app):
    window = MainWindow()
    window.show_analysis(stats_with_waveform(counts_with(200)))
    window.clear_analysis()
    assert window.waveform_view.counts is None
    assert window.histogram_view.histogram is None


def test_a_failed_measurement_clears_both_charts(app):
    window = MainWindow()
    window.show_analysis(stats_with_waveform(counts_with(200)))
    window.show_analysis(None)
    assert window.waveform_view.counts is None
    assert window.histogram_view.histogram is None


def test_stats_without_a_waveform_are_fine(app):
    window = MainWindow()
    window.show_analysis(stats_with_waveform(None))
    assert window.waveform_view.counts is None


def test_each_chart_has_an_explanation(app):
    window = MainWindow()
    assert "histogram" in window.histogram_view.toolTip().lower()
    assert "left edge" in window.waveform_view.toolTip().lower()
    window.scope_selector.setCurrentIndex(1)
    assert window.scope_selector.toolTip() == window.waveform_view.toolTip()
    window.scope_selector.setCurrentIndex(0)
    assert window.scope_selector.toolTip() == window.histogram_view.toolTip()


def test_the_chart_area_keeps_the_same_height_for_both_charts(app):
    window = MainWindow()
    assert window.scope_stack.height() == window.histogram_view.height() == window.waveform_view.height()