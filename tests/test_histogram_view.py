"""Tests for the histogram widget. They draw it without showing a window."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.ui.histogram_view import HistogramView, bar_heights


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ---------- bar_heights: the numbers behind the picture ----------

def counts(**bins):
    """256 zeros with some bins filled in, e.g. counts(b100=5, b255=9)."""
    arr = np.zeros(256)
    for name, value in bins.items():
        arr[int(name[1:])] = value
    return arr


def test_tallest_middle_bar_fills_the_height():
    heights = bar_heights(counts(b100=40, b150=20), 100)
    assert heights[100] == pytest.approx(100)
    assert heights[150] == pytest.approx(50)
    assert heights[0] == 0


def test_end_bars_are_cut_off_instead_of_squashing_the_rest():
    # Lots of pure black must not flatten a healthy spread of other bars
    heights = bar_heights(counts(b0=1000, b100=400, b150=200, b255=500), 100)
    assert heights[0] == pytest.approx(100)         # cut off at the top
    assert heights[255] == pytest.approx(100)
    assert heights[100] == pytest.approx(100)       # the scale ignores the ends
    assert heights[150] == pytest.approx(50)


def test_a_few_stray_pixels_are_not_stretched_to_full_height():
    # 99.7% of the pixels are pure black or white; the rest are the soft edge
    # between them. That edge must stay small next to the end bars.
    heights = bar_heights(counts(b0=50000, b255=50000, b128=300), 100)
    assert heights[0] == pytest.approx(100)
    assert heights[255] == pytest.approx(100)
    assert 0 < heights[128] < 20                    # 0.3% of 2% is 15%


def test_a_bar_needs_two_percent_of_the_pixels_to_reach_the_top():
    total = 10000
    heights = bar_heights(counts(b100=total * 0.02, b0=total * 0.98 - 1), 100)
    assert heights[100] == pytest.approx(100, abs=1)
    heights = bar_heights(counts(b100=total * 0.01, b0=total * 0.99 - 1), 100)
    assert heights[100] == pytest.approx(50, abs=1)


def test_image_with_only_end_bins_still_draws():
    heights = bar_heights(counts(b0=50, b255=100), 80)
    assert heights[255] == pytest.approx(80)
    assert heights[0] == pytest.approx(80)          # both run to the top


def test_single_bin_fills_the_height():
    assert bar_heights(counts(b128=1000), 100)[128] == pytest.approx(100)


def test_empty_histogram_draws_nothing():
    assert bar_heights(np.zeros(256), 100).sum() == 0


# ---------- the widget itself ----------

def render(view):
    view.resize(256, 92)
    return view.grab().toImage()


def is_bar(color):
    return color.red() > 150 and color.green() > 150 and color.blue() > 150


def is_background(color):
    return color.red() < 80 and color.green() < 80 and color.blue() < 80


def test_empty_widget_is_all_background(app):
    image = render(HistogramView())
    for x in (10, 128, 245):
        assert is_background(image.pixelColor(x, 80))


def test_single_bin_draws_one_bar_in_the_right_place(app):
    view = HistogramView()
    view.set_histogram(counts(b128=1000))
    image = render(view)
    assert is_bar(image.pixelColor(128, 85))        # at the bottom of bin 128
    assert is_bar(image.pixelColor(128, 5))         # and nearly at the top
    assert is_background(image.pixelColor(60, 85))  # other bins are empty
    assert is_background(image.pixelColor(200, 85))


def test_half_height_bar_is_half_as_tall(app):
    view = HistogramView()
    view.set_histogram(counts(b60=100, b200=50))
    image = render(view)
    assert is_bar(image.pixelColor(60, 10))         # full height
    assert is_bar(image.pixelColor(200, 70))        # lower half of the bar
    assert is_background(image.pixelColor(200, 20)) # above half height


def test_black_and_white_spikes_appear_at_the_edges(app):
    view = HistogramView()
    view.set_histogram(counts(b0=500, b255=500))
    image = render(view)
    assert is_bar(image.pixelColor(0, 50))
    assert is_bar(image.pixelColor(255, 50))
    assert is_background(image.pixelColor(128, 50))


def test_clearing_removes_the_bars(app):
    view = HistogramView()
    view.set_histogram(counts(b128=1000))
    view.set_histogram(None)
    assert view.histogram is None
    assert is_background(render(view).pixelColor(128, 85))


def test_widget_keeps_a_fixed_height_and_stretches_sideways(app):
    view = HistogramView()
    assert view.minimumHeight() == view.maximumHeight() == 92
    assert view.minimumWidth() >= 200