"""Tests for the color palette chart and its place in the drop-down."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.analyze import BlackBars, FrameStats, PaletteColor
from blackboard.ui.main_window import SCOPE_NAMES, MainWindow
from blackboard.ui.palette_view import (
    BACKGROUND, STRIP_HEIGHT, PaletteView, palette_tip, readable_text_color,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


RED = PaletteColor(blue=0, green=0, red=220, share=50.0)
GREEN = PaletteColor(blue=0, green=200, red=0, share=30.0)
BLUE = PaletteColor(blue=200, green=0, red=0, share=15.0)
WHITE = PaletteColor(blue=250, green=250, red=250, share=5.0)


def render(view, width=300, height=92):
    """Draw the widget and return it as a function: pixel(x, y) -> (R, G, B)."""
    view.resize(width, height)
    image = view.grab().toImage()

    def pixel(x, y):
        color = image.pixelColor(x, y)
        return color.red(), color.green(), color.blue()
    return pixel


# ---------- text color ----------

@pytest.mark.parametrize("rgb, expected", [
    ((0, 0, 0), "#ffffff"),                        # black tile: white text
    ((255, 255, 255), "#000000"),                    # white tile: black text
    ((255, 255, 0), "#000000"),                      # yellow is bright: black text
    ((0, 0, 160), "#ffffff"),                        # dark blue: white text
    ((200, 30, 30), "#ffffff"),                      # red: white text
])
def test_text_is_black_on_light_tiles_and_white_on_dark_ones(rgb, expected):
    assert readable_text_color(*rgb).name() == expected


# ---------- tooltip text ----------

def test_palette_tip_lists_every_color_with_its_share():
    text = palette_tip([RED, GREEN])
    assert text.splitlines() == ["1. #dc0000  50.0 %", "2. #00c800  30.0 %"]


# ---------- the widget ----------

def test_widget_keeps_a_fixed_height_and_stretches_sideways(app):
    view = PaletteView()
    assert view.height() == 92 and view.minimumWidth() >= 100
    view.resize(500, 300)
    assert view.height() == 92


def test_empty_widget_is_all_background(app):
    pixel = render(PaletteView())
    background = (BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue())
    for point in [(0, 0), (150, 40), (299, 91), (10, 88)]:
        assert pixel(*point) == background


def test_each_color_gets_an_equal_tile_in_order(app):
    view = PaletteView()
    view.set_palette([RED, GREEN, BLUE, WHITE])
    pixel = render(view, width=400)                  # 100 pixels per tile
    assert pixel(3, 3) == (220, 0, 0)
    assert pixel(103, 3) == (0, 200, 0)
    assert pixel(203, 3) == (0, 0, 200)
    assert pixel(303, 3) == (250, 250, 250)


def test_the_strip_is_split_by_share(app):
    view = PaletteView()
    view.set_palette([RED, GREEN, BLUE, WHITE])      # 50, 30, 15, 5 percent
    pixel = render(view, width=400)
    y = 92 - STRIP_HEIGHT // 2
    assert pixel(100, y) == (220, 0, 0)              # red: 0 to 200
    assert pixel(250, y) == (0, 200, 0)              # green: 200 to 320
    assert pixel(350, y) == (0, 0, 200)              # blue: 320 to 380
    assert pixel(395, y) == (250, 250, 250)          # white: 380 to 400


def test_a_single_color_fills_the_whole_widget(app):
    view = PaletteView()
    view.set_palette([PaletteColor(blue=10, green=20, red=30, share=100.0)])
    pixel = render(view)
    assert pixel(3, 3) == (30, 20, 10) and pixel(296, 3) == (30, 20, 10)
    assert pixel(150, 88) == (30, 20, 10)


def test_clearing_removes_the_colors(app):
    view = PaletteView()
    view.set_palette([RED, GREEN])
    view.set_palette(None)
    pixel = render(view)
    assert pixel(3, 3) == (BACKGROUND.red(), BACKGROUND.green(), BACKGROUND.blue())


def test_a_narrow_widget_still_draws_every_tile(app):
    view = PaletteView()
    view.set_palette([RED, GREEN, BLUE, WHITE, RED])
    pixel = render(view, width=200)                  # 40 pixels per tile
    assert pixel(3, 3) == (220, 0, 0) and pixel(43, 3) == (0, 200, 0)


def test_text_appears_on_a_tile(app):
    view = PaletteView()
    view.set_palette([RED, GREEN])
    pixel = render(view, width=300)
    # Somewhere in the middle of the first tile the white text differs from the tile color
    colors = {pixel(x, y) for x in range(40, 110) for y in range(15, 60)}
    assert len(colors) > 1


def test_tooltip_has_the_explanation_and_the_colors(app):
    view = PaletteView()
    view.set_help("What a palette is.")
    assert view.toolTip() == "What a palette is."
    view.set_palette([RED, GREEN])
    assert view.toolTip().startswith("What a palette is.")
    assert "1. #dc0000  50.0 %" in view.toolTip()
    view.set_palette(None)
    assert view.toolTip() == "What a palette is."


# ---------- the drop-down in the window ----------

def stats_with_palette(palette):
    return FrameStats(50.0, 50.0, 0.0, 0.0, 30.0, 40.0, 30.0, 60.0, None,
                      BlackBars(), np.ones(256, np.int64), None, palette)


def test_the_drop_down_offers_the_palette_third(app):
    window = MainWindow()
    names = [window.scope_selector.itemText(i) for i in range(window.scope_selector.count())]
    assert names == SCOPE_NAMES and names[2] == "Color palette"


def test_picking_the_palette_shows_it(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(2)
    assert window.scope_stack.currentWidget() is window.palette_view


def test_every_frame_updates_the_palette_even_when_it_is_hidden(app):
    window = MainWindow()
    window.show_analysis(stats_with_palette([RED, GREEN]))
    assert window.palette_view.colors == [RED, GREEN]
    assert window.scope_stack.currentWidget() is window.histogram_view


def test_the_choice_stays_when_the_next_frame_arrives(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(2)
    window.show_analysis(stats_with_palette([RED]))
    window.show_analysis(stats_with_palette([GREEN]))
    assert window.scope_stack.currentWidget() is window.palette_view
    assert window.palette_view.colors == [GREEN]


def test_clearing_and_failures_empty_the_palette(app):
    window = MainWindow()
    window.show_analysis(stats_with_palette([RED]))
    window.clear_analysis()
    assert window.palette_view.colors is None
    window.show_analysis(stats_with_palette([RED]))
    window.show_analysis(None)
    assert window.palette_view.colors is None


def test_stats_without_a_palette_are_fine(app):
    window = MainWindow()
    window.show_analysis(stats_with_palette(None))
    assert window.palette_view.colors is None


def test_the_drop_down_tooltip_matches_the_chart_picked(app):
    window = MainWindow()
    window.scope_selector.setCurrentIndex(2)
    assert "palette" in window.scope_selector.toolTip().lower() or "main colors" in window.scope_selector.toolTip().lower()
    window.scope_selector.setCurrentIndex(0)
    assert window.scope_selector.toolTip() == window.histogram_view.toolTip()


def test_all_four_charts_are_the_same_height(app):
    window = MainWindow()
    heights = {window.scope_stack.height(), window.histogram_view.height(),
               window.waveform_view.height(), window.palette_view.height(),
               window.vectorscope_view.height()}
    assert len(heights) == 1