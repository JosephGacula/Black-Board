"""Tests for the vectorscope chart and its place in the drop-down."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.analyze import (
    VECTORSCOPE_SIZE, BlackBars, FrameStats, chroma_of_color, chroma_vectorscope,
)
from blackboard.ui.main_window import SCOPE_NAMES, MainWindow
from blackboard.ui.vectorscope_view import (
    BACKGROUND, FULL_SHARE, MARGIN, TARGET_COLORS, TARGET_SIZE, TRACE,
    VectorscopeView, map_point, vectorscope_intensity, vectorscope_picture,
)

BACKGROUND_RGB = (BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue())


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def solid(blue, green, red, size=10):
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = (blue, green, red)
    return image


def render(view, width=300, height=92):
    """Draw the widget and return a function: pixel(x, y) -> (R, G, B)."""
    view.resize(width, height)
    image = view.grab().toImage()

    def pixel(x, y):
        color = image.pixelColor(int(round(x)), int(round(y)))
        return color.red(), color.green(), color.blue()
    return pixel


def colors_near(pixel, x, y, radius=5):
    """Every color seen in a small square around a point."""
    spread = range(-radius, radius + 1)
    return {pixel(x + dx, y + dy) for dx in spread for dy in spread}


def brightest_near(pixel, x, y, radius=1):
    """The strongest green seen within a few pixels of a point."""
    return max(pixel(x + dx, y + dy)[1]
               for dx in range(-radius, radius + 1)
               for dy in range(-radius, radius + 1))


# ---------- intensity ----------

def test_intensity_has_the_same_shape_and_is_8_bit():
    strength = vectorscope_intensity(np.ones((16, 16), np.int64))
    assert strength.shape == (16, 16) and strength.dtype == np.uint8


def test_a_cell_holding_the_full_share_is_full_strength():
    counts = np.zeros((8, 8), np.int64)
    counts[3, 3] = 1000
    counts[0, 0] = 1000 * (1 - FULL_SHARE) / FULL_SHARE   # makes (3, 3) exactly the full share
    strength = vectorscope_intensity(counts)
    assert strength[3, 3] == 255


def test_rarer_cells_are_fainter_but_still_show():
    counts = np.zeros((8, 8), np.int64)
    counts[0, 0] = 1_000_000
    counts[4, 4] = 100                      # a tiny share
    strength = vectorscope_intensity(counts)
    assert 0 < strength[4, 4] < strength[0, 0] == 255


def test_strength_is_eased_with_a_square_root():
    counts = np.zeros((8, 8), np.int64)
    total = 1_000_000
    counts[0, 0] = total // 2
    counts[1, 1] = int(total * FULL_SHARE / 4)      # a quarter of the full share
    counts[2, 2] = total - counts[0, 0] - counts[1, 1]
    strength = vectorscope_intensity(counts)
    assert strength[1, 1] == pytest.approx(255 * 0.5, abs=2)   # sqrt(1/4)


def test_the_same_picture_looks_the_same_at_any_size():
    small = chroma_vectorscope(np.tile(solid(0, 0, 255, size=10), (1, 1, 1)))
    big = chroma_vectorscope(solid(0, 0, 255, size=40))
    assert (vectorscope_intensity(small) == vectorscope_intensity(big)).all()


def test_an_empty_map_is_dark_and_does_not_divide_by_zero():
    strength = vectorscope_intensity(np.zeros((8, 8), np.int64))
    assert (strength == 0).all()


# ---------- the picture ----------

def test_picture_has_the_requested_size_and_three_colors():
    picture = vectorscope_picture(chroma_vectorscope(solid(10, 20, 30)), 80)
    assert picture.shape == (80, 80, 3) and picture.dtype == np.uint8


def test_picture_can_be_bigger_than_the_map():
    picture = vectorscope_picture(chroma_vectorscope(solid(10, 20, 30)), 300)
    assert picture.shape == (300, 300, 3)


def test_picture_uses_the_trace_color_at_full_strength():
    picture = vectorscope_picture(chroma_vectorscope(solid(90, 90, 90)), 128)
    brightest = picture.reshape(-1, 3).max(axis=0)
    assert tuple(brightest) == TRACE


def test_a_gray_picture_lights_the_middle_of_the_picture():
    picture = vectorscope_picture(chroma_vectorscope(solid(90, 90, 90)), 128)
    ys, xs = np.nonzero(picture[..., 1])
    assert len(ys) == 1 and abs(ys[0] - 64) <= 1 and abs(xs[0] - 64) <= 1


def test_red_is_at_the_top_and_blue_at_the_right():
    red = vectorscope_picture(chroma_vectorscope(solid(0, 0, 255)), 128)
    blue = vectorscope_picture(chroma_vectorscope(solid(255, 0, 0)), 128)
    ry, rx = np.unravel_index(np.argmax(red[..., 1]), red.shape[:2])
    by, bx = np.unravel_index(np.argmax(blue[..., 1]), blue.shape[:2])
    assert ry <= 2 and rx < 64          # red: top, a little left
    assert bx >= 125 and abs(by - 69) <= 2


def test_a_thin_trace_survives_being_squeezed():
    picture = vectorscope_picture(chroma_vectorscope(solid(0, 0, 255)), 40)
    assert picture[..., 1].max() == TRACE[1]


# ---------- map_point ----------

def test_the_middle_of_the_map_is_the_middle_of_the_square():
    assert map_point(0, 0, 100) == (pytest.approx(50.0), pytest.approx(50.0))


def test_map_point_puts_blue_right_and_red_up():
    x, y = map_point(127.5, 0, 100)
    assert x > 99
    x, y = map_point(0, 127.5, 100)
    assert y < 1


# ---------- the widget ----------

def test_widget_keeps_a_fixed_height_and_stretches_sideways(app):
    view = VectorscopeView()
    assert view.height() == 92 and view.minimumWidth() >= 100
    view.resize(500, 300)
    assert view.height() == 92


def test_the_map_is_a_centered_square(app):
    view = VectorscopeView()
    view.resize(300, 92)
    left, top, side = view.plot_area()
    assert side == 92 - 2 * MARGIN
    assert left == (300 - side) // 2 and top == MARGIN


def test_empty_widget_shows_the_background_the_guides_and_six_boxes(app):
    view = VectorscopeView()
    pixel = render(view)
    assert pixel(2, 2) == BACKGROUND_RGB and pixel(298, 90) == BACKGROUND_RGB
    centre = view.point_of(0, 0)
    assert pixel(*centre) != BACKGROUND_RGB             # the cross
    for blue, green, red in TARGET_COLORS:
        x, y = view.point_of(*chroma_of_color(blue, green, red))
        assert (red, green, blue) in colors_near(pixel, x, y)


def test_every_box_fits_inside_the_widget(app):
    view = VectorscopeView()
    view.resize(300, 92)
    for blue, green, red in TARGET_COLORS:
        x, y = view.point_of(*chroma_of_color(blue, green, red))
        assert x - TARGET_SIZE / 2 >= 0 and x + TARGET_SIZE / 2 <= view.width()
        assert y - TARGET_SIZE / 2 >= 0 and y + TARGET_SIZE / 2 <= view.height()


def test_boxes_are_drawn_in_their_own_colors(app):
    view = VectorscopeView()
    pixel = render(view)
    for blue, green, red in TARGET_COLORS:
        x, y = view.point_of(*chroma_of_color(blue, green, red))
        assert (red, green, blue) in colors_near(pixel, x, y)
        # hollow: the middle of the box is not filled with its color
        assert pixel(x, y) != (red, green, blue)


def test_a_gray_picture_lights_the_middle(app):
    view = VectorscopeView()
    view.set_vectorscope(chroma_vectorscope(solid(90, 90, 90)))
    pixel = render(view)
    x, y = view.point_of(0, 0)
    assert brightest_near(pixel, x, y) >= TRACE[1] - 10


def test_a_red_picture_lights_the_red_corner_and_not_the_blue_one(app):
    view = VectorscopeView()
    view.set_vectorscope(chroma_vectorscope(solid(0, 0, 255)))
    pixel = render(view)
    rx, ry = view.point_of(*chroma_of_color(0, 0, 255))
    bx, by = view.point_of(*chroma_of_color(255, 0, 0))
    assert brightest_near(pixel, rx, ry, radius=2) > 100
    assert brightest_near(pixel, bx - 6, by, radius=1) < 60


def test_clearing_removes_the_trace(app):
    view = VectorscopeView()
    empty = render(view)
    view.set_vectorscope(chroma_vectorscope(solid(90, 90, 90)))
    view.set_vectorscope(None)
    cleared = render(view)
    x, y = view.point_of(0, 0)
    assert cleared(x + 3, y + 3) == empty(x + 3, y + 3)
    assert cleared(x, y) == empty(x, y)


def test_the_trace_is_added_to_the_guides_and_does_not_hide_them(app):
    view = VectorscopeView()
    plain = render(view)
    view.set_vectorscope(chroma_vectorscope(solid(90, 90, 90)))
    lit = render(view)
    x, y = view.point_of(0, 0)
    assert lit(x, y)[1] >= plain(x, y)[1]


# ---------- the drop-down in the window ----------

def stats_with_vectorscope(counts):
    return FrameStats(50.0, 50.0, 0.0, 0.0, 30.0, 40.0, 30.0, 60.0, None,
                      BlackBars(), np.ones(256, np.int64), None, None, counts)


def test_the_drop_down_offers_the_vectorscope_last(app):
    window = MainWindow()
    names = [window.scope_selector.itemText(i) for i in range(window.scope_selector.count())]
    assert names == SCOPE_NAMES and names[-1] == "Vectorscope"


def test_picking_the_vectorscope_shows_it(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(3)
    assert window.scope_stack.currentWidget() is window.vectorscope_view


def test_every_frame_updates_the_vectorscope_even_when_it_is_hidden(app):
    window = MainWindow()
    counts = chroma_vectorscope(solid(0, 0, 255))
    window.show_analysis(stats_with_vectorscope(counts))
    assert (window.vectorscope_view.counts == counts).all()
    assert window.scope_stack.currentWidget() is window.histogram_view


def test_the_choice_stays_when_the_next_frame_arrives(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(3)
    window.show_analysis(stats_with_vectorscope(chroma_vectorscope(solid(0, 0, 255))))
    second = chroma_vectorscope(solid(255, 0, 0))
    window.show_analysis(stats_with_vectorscope(second))
    assert window.scope_stack.currentWidget() is window.vectorscope_view
    assert (window.vectorscope_view.counts == second).all()


def test_clearing_and_failures_empty_the_vectorscope(app):
    window = MainWindow()
    window.show_analysis(stats_with_vectorscope(chroma_vectorscope(solid(0, 0, 255))))
    window.clear_analysis()
    assert window.vectorscope_view.counts is None
    window.show_analysis(stats_with_vectorscope(chroma_vectorscope(solid(0, 0, 255))))
    window.show_analysis(None)
    assert window.vectorscope_view.counts is None


def test_stats_without_a_vectorscope_are_fine(app):
    window = MainWindow()
    window.show_analysis(stats_with_vectorscope(None))
    assert window.vectorscope_view.counts is None


def test_the_vectorscope_has_an_explanation(app):
    window = MainWindow()
    assert "color" in window.vectorscope_view.toolTip().lower()
    window.scope_selector.setCurrentIndex(3)
    assert window.scope_selector.toolTip() == window.vectorscope_view.toolTip()
    assert window.scope_selector.itemData(3, 3) == window.vectorscope_view.toolTip()


def test_a_real_frame_goes_all_the_way_to_the_chart(app):
    from blackboard.analyze import analyze_frame
    window = MainWindow()
    window.show_analysis(analyze_frame(solid(0, 0, 255, size=30)))
    assert window.vectorscope_view.counts.shape == (VECTORSCOPE_SIZE, VECTORSCOPE_SIZE)
    assert window.vectorscope_view.counts.sum() == 900