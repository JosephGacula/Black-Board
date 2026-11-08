"""Tests for the wording of the "Black bars" row in the Frame analysis panel."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.analyze import BlackBars, FrameStats
from blackboard.ui.main_window import MainWindow, bars_text


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ---------- bars_text: the words shown in the panel ----------

def test_no_bars_says_none():
    assert bars_text(BlackBars()) == "None"


def test_matching_top_and_bottom_bars_are_one_number():
    assert bars_text(BlackBars(top=12.5, bottom=12.5)) == "Top/bottom 12.5 %"


def test_nearly_matching_bars_are_averaged():
    # Off by less than a percent is treated as the same on both sides
    assert bars_text(BlackBars(top=12.0, bottom=12.8)) == "Top/bottom 12.4 %"


def test_different_top_and_bottom_bars_are_listed_separately():
    assert bars_text(BlackBars(top=5.0, bottom=12.0)) == "Top 5.0 %, bottom 12.0 %"


def test_matching_side_bars():
    assert bars_text(BlackBars(left=10.0, right=10.0)) == "Left/right 10.0 %"


def test_one_bar_only():
    assert bars_text(BlackBars(top=25.0)) == "Top 25.0 %"
    assert bars_text(BlackBars(right=7.5)) == "Right 7.5 %"


def test_one_bar_only_does_not_say_the_other_side():
    assert "bottom" not in bars_text(BlackBars(top=25.0))


def test_bars_on_all_sides():
    text = bars_text(BlackBars(top=10, bottom=10, left=5, right=5))
    assert text == "Top/bottom 10.0 %, left/right 5.0 %"


# ---------- the row in the window ----------

def stats_with_bars(bars):
    return FrameStats(50.0, 50.0, 0.0, 0.0, 30.0, 40.0, 30.0, 60.0, None,
                      bars, np.ones(256, np.int64))


def test_panel_shows_the_bars(app):
    window = MainWindow()
    window.show_analysis(stats_with_bars(BlackBars(top=12.5, bottom=12.5)))
    assert window.analysis_fields["Black bars"].text() == "Top/bottom 12.5 %"


def test_panel_says_none_without_bars(app):
    window = MainWindow()
    window.show_analysis(stats_with_bars(BlackBars()))
    assert window.analysis_fields["Black bars"].text() == "None"


def test_panel_row_has_an_explanation(app):
    window = MainWindow()
    assert "letterbox" in window.analysis_fields["Black bars"].toolTip().lower()