"""Tests for the little color square next to "Dominant hue"."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.analyze import BlackBars, FrameStats
from blackboard.ui.main_window import MainWindow, hue_color, hue_text


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def stats_with_hue(degrees):
    return FrameStats(50.0, 50.0, 0.0, 0.0, 30.0, 40.0, 30.0, 60.0, degrees,
                      BlackBars(), np.ones(256, np.int64))


# ---------- hue_color: the color the square is painted ----------

@pytest.mark.parametrize("degrees, color", [
    (0, "#ff0000"),       # red
    (60, "#ffff00"),      # yellow
    (120, "#00ff00"),     # green
    (180, "#00ffff"),     # cyan
    (240, "#0000ff"),     # blue
    (300, "#ff00ff"),     # magenta
])
def test_hue_color_of_the_main_colors(degrees, color):
    assert hue_color(degrees) == color


def test_hue_color_wraps_around_the_circle():
    assert hue_color(360) == hue_color(0)
    assert hue_color(-120) == hue_color(240)


def test_hue_color_is_always_a_valid_hex_color():
    for degrees in range(0, 360, 7):
        color = hue_color(degrees)
        assert len(color) == 7 and color[0] == "#"
        int(color[1:], 16)                    # raises if it isn't hex


# ---------- the square in the window ----------

def test_square_is_hidden_until_there_is_a_hue(app):
    window = MainWindow()
    window.show()
    assert not window.hue_swatch.isVisible()


def test_square_shows_the_dominant_hue(app):
    window = MainWindow()
    window.show()
    window.show_analysis(stats_with_hue(120.0))
    assert window.hue_swatch.isVisible()
    assert "#00ff00" in window.hue_swatch.styleSheet()
    assert window.analysis_fields["Dominant hue"].text() == hue_text(120.0)


def test_square_follows_the_hue_when_it_changes(app):
    window = MainWindow()
    window.show()
    window.show_analysis(stats_with_hue(0.0))
    window.show_analysis(stats_with_hue(240.0))
    assert "#0000ff" in window.hue_swatch.styleSheet()
    assert "#ff0000" not in window.hue_swatch.styleSheet()


def test_square_is_hidden_for_a_neutral_picture(app):
    window = MainWindow()
    window.show()
    window.show_analysis(stats_with_hue(120.0))
    window.show_analysis(stats_with_hue(None))
    assert not window.hue_swatch.isVisible()
    assert window.analysis_fields["Dominant hue"].text() == "None (neutral)"


def test_square_is_hidden_when_measuring_fails(app):
    window = MainWindow()
    window.show()
    window.show_analysis(stats_with_hue(120.0))
    window.show_analysis(None)
    assert not window.hue_swatch.isVisible()


def test_square_is_hidden_when_the_analysis_is_cleared(app):
    window = MainWindow()
    window.show()
    window.show_analysis(stats_with_hue(120.0))
    window.clear_analysis()
    assert not window.hue_swatch.isVisible()


def test_square_sits_right_after_the_hue_text(app):
    window = MainWindow()
    window.show()
    window.show_analysis(stats_with_hue(120.0))
    text = window.analysis_fields["Dominant hue"]
    square = window.hue_swatch
    # Same cell, same line, a few pixels to the right of the text
    assert square.parentWidget() is text.parentWidget()
    assert square.x() >= text.x() + text.width()
    assert square.x() - (text.x() + text.width()) <= 12
    assert abs(square.geometry().center().y() - text.geometry().center().y()) <= 2