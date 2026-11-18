"""Tests for choosing a segment: the Start/End boxes and the band."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from blackboard.ui.segment_picker import (
    BAND_BACKGROUND, BAND_HEIGHT, BAND_MARKER, BAND_SEGMENT, MIN_SEGMENT_WIDTH,
    UNIT_FRAMES, UNIT_TIME, SegmentBand, SegmentPicker, TimeBox, format_time,
    parse_time,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def rgb(color):
    return color.red(), color.green(), color.blue()


def render(band, width=400):
    band.resize(width, BAND_HEIGHT)
    image = band.grab().toImage()

    def pixel(x, y=2):
        color = image.pixelColor(int(x), int(y))
        return color.red(), color.green(), color.blue()
    return pixel


def picker(frames=1000, fps=25.0):
    widget = SegmentPicker()
    widget.set_video(frames, fps)
    return widget


# ---------- the picker ----------

def test_a_new_picker_is_switched_off(app):
    widget = SegmentPicker()
    assert not widget.isEnabled()
    assert widget.length_label.text() == ""


def test_opening_a_video_selects_all_of_it(app):
    widget = picker(1000)
    assert widget.isEnabled()
    assert widget.segment() == (0, 999)
    assert widget.start_spin.maximum() == 999 and widget.end_spin.maximum() == 999


def test_the_length_is_shown_in_frames_and_seconds(app):
    widget = picker(1000, fps=25.0)
    assert widget.length_label.text() == "1000 frames, 40.00 s"
    widget.set_segment(100, 124)
    assert widget.length_label.text() == "25 frames, 1.00 s"


def test_one_frame_is_not_plural(app):
    widget = picker()
    widget.set_segment(5, 5)
    assert widget.length_label.text().startswith("1 frame,")


def test_the_length_works_without_a_frame_rate(app):
    widget = picker(100, fps=0.0)
    assert widget.length_label.text() == "100 frames"


def test_typing_a_start_reports_the_segment(app):
    widget = picker()
    heard = []
    widget.changed.connect(lambda first, last: heard.append((first, last)))
    widget.start_spin.setValue(200)
    assert heard == [(200, 999)]
    assert widget.segment() == (200, 999)


def test_typing_an_end_reports_the_segment(app):
    widget = picker()
    heard = []
    widget.changed.connect(lambda first, last: heard.append((first, last)))
    widget.end_spin.setValue(300)
    assert heard == [(0, 300)]


def test_a_start_past_the_end_pushes_the_end_along(app):
    widget = picker()
    widget.end_spin.setValue(300)
    heard = []
    widget.changed.connect(lambda first, last: heard.append((first, last)))
    widget.start_spin.setValue(500)
    assert widget.segment() == (500, 500)
    assert heard == [(500, 500)]               # one report, with the final numbers


def test_an_end_before_the_start_pulls_the_start_back(app):
    widget = picker()
    widget.start_spin.setValue(600)
    widget.end_spin.setValue(400)
    assert widget.segment() == (400, 400)


def test_the_start_never_passes_the_end(app):
    widget = picker(200)
    for start, end in [(150, 20), (3, 3), (199, 0), (0, 199), (80, 81)]:
        widget.start_spin.setValue(start)
        widget.end_spin.setValue(end)
        first, last = widget.segment()
        assert first <= last
        assert widget.length_label.text().startswith(f"{last - first + 1} frame")


def test_the_current_frame_buttons_ask_the_window(app):
    widget = picker()
    asked = []
    widget.start_from_current.connect(lambda: asked.append("start"))
    widget.end_from_current.connect(lambda: asked.append("end"))
    widget.start_button.click()
    widget.end_button.click()
    assert asked == ["start", "end"]


def test_set_first_and_set_last_move_the_boxes(app):
    widget = picker()
    widget.set_first(100)
    widget.set_last(250)
    assert widget.segment() == (100, 250)


def test_set_segment_does_not_report_a_change(app):
    widget = picker()
    heard = []
    widget.changed.connect(lambda first, last: heard.append((first, last)))
    widget.set_segment(10, 20)
    assert heard == [] and widget.segment() == (10, 20)


def test_the_boxes_cannot_leave_the_video(app):
    widget = picker(100)
    widget.start_spin.setValue(-5)
    widget.end_spin.setValue(5000)
    assert widget.segment() == (0, 99)


def test_a_one_frame_video_works(app):
    widget = picker(1)
    assert widget.segment() == (0, 0)
    assert widget.length_label.text().startswith("1 frame,")


def test_clearing_switches_it_off_again(app):
    widget = picker()
    widget.set_segment(10, 20)
    widget.clear()
    assert not widget.isEnabled() and widget.length_label.text() == ""
    assert widget.segment() == (0, 0)


def test_opening_another_video_selects_all_of_that_one(app):
    widget = picker(1000)
    widget.set_segment(10, 20)
    widget.set_video(50, 30.0)
    assert widget.segment() == (0, 49)
    assert widget.start_spin.maximum() == 49


# ---------- the band ----------

def test_an_empty_band_is_all_background(app):
    pixel = render(SegmentBand())
    assert all(pixel(x) == rgb(BAND_BACKGROUND) for x in (0, 100, 399))


def test_the_whole_video_lights_up_the_whole_band(app):
    band = SegmentBand()
    band.set_total(1000)
    pixel = render(band)
    assert pixel(0) == rgb(BAND_SEGMENT) and pixel(399) == rgb(BAND_SEGMENT)


def test_only_the_chosen_stretch_is_lit(app):
    band = SegmentBand()
    band.set_total(1000)
    band.set_segment(250, 499)                 # a quarter, from 25% to 50%
    pixel = render(band)                       # 400 pixels: 100 to 200
    assert pixel(50) == rgb(BAND_BACKGROUND)
    assert pixel(100) == rgb(BAND_SEGMENT) and pixel(199) == rgb(BAND_SEGMENT)
    assert pixel(200) == rgb(BAND_BACKGROUND) and pixel(350) == rgb(BAND_BACKGROUND)


def test_the_segment_rect_matches_the_frames(app):
    band = SegmentBand()
    band.set_total(1000)
    band.resize(400, BAND_HEIGHT)
    band.set_segment(250, 499)
    assert band.segment_rect() == (100, 100)


def test_a_one_frame_segment_is_still_visible(app):
    band = SegmentBand()
    band.set_total(100000)
    band.resize(400, BAND_HEIGHT)
    band.set_segment(5000, 5000)
    left, width = band.segment_rect()
    assert width == MIN_SEGMENT_WIDTH
    assert render(band)(left) == rgb(BAND_SEGMENT)


def test_a_segment_at_the_very_end_stays_inside_the_band(app):
    band = SegmentBand()
    band.set_total(1000)
    band.resize(400, BAND_HEIGHT)
    band.set_segment(999, 999)
    left, width = band.segment_rect()
    assert left == 399 and left + width >= 400    # last pixel is lit


def test_the_tick_shows_the_current_frame(app):
    band = SegmentBand()
    band.set_total(1000)
    band.set_segment(0, 0)                      # keep the lit part out of the way
    band.set_position(500)
    pixel = render(band)
    assert band.marker_x() == int(500.5 / 1000 * 400)
    assert pixel(band.marker_x()) == rgb(BAND_MARKER)


def test_there_is_no_tick_without_a_position(app):
    band = SegmentBand()
    band.set_total(1000)
    assert band.marker_x() is None
    pixel = render(band)
    assert all(pixel(x) != rgb(BAND_MARKER) for x in range(400))


def test_clearing_empties_the_band(app):
    band = SegmentBand()
    band.set_total(1000)
    band.set_position(10)
    band.clear()
    assert band.segment_rect() is None and band.marker_x() is None
    assert render(band)(100) == rgb(BAND_BACKGROUND)


def test_the_band_keeps_a_small_fixed_height(app):
    band = SegmentBand()
    band.resize(700, 300)
    assert band.height() == BAND_HEIGHT


# ---------- reading and writing times ----------

@pytest.mark.parametrize("seconds,text", [
    (0, "0:00.00"), (1.5, "0:01.50"), (59.99, "0:59.99"), (60, "1:00.00"),
    (259.33, "4:19.33"), (3600, "1:00:00.00"), (3725.5, "1:02:05.50"),
    (-3, "0:00.00"),
])
def test_times_are_written_as_minutes_and_seconds(seconds, text):
    assert format_time(seconds) == text


def test_rounding_can_carry_into_the_next_second():
    assert format_time(59.999) == "1:00.00"


def test_three_decimals_can_be_asked_for():
    assert format_time(1.5, places=3) == "0:01.500"


@pytest.mark.parametrize("text,seconds", [
    ("90", 90.0), ("1:30", 90.0), ("1:30.5", 90.5), ("0:01:30.25", 90.25),
    ("4:19.33", 259.33), ("1:30s", 90.0), ("  .5 ", 0.5), ("0", 0.0),
    ("1:02:05.50", 3725.5), ("12.", 12.0),
])
def test_times_can_be_typed_in_several_ways(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "1:", ":30", "1:2:3:4", "-5", "1,5", "1:30:"])
def test_things_that_are_not_times_are_not_accepted(text):
    assert parse_time(text) is None


def test_a_written_time_reads_back_the_same():
    for seconds in (0, 0.01, 12.34, 59.99, 61.5, 3599.99, 3600.01):
        assert parse_time(format_time(seconds)) == pytest.approx(seconds, abs=0.006)


# ---------- the time boxes ----------

def test_the_boxes_start_in_frames(app):
    widget = picker()
    assert widget.unit() == UNIT_FRAMES
    assert widget.start_spin.isVisibleTo(widget) and not widget.start_time.isVisibleTo(widget)
    assert widget.end_spin.isVisibleTo(widget) and not widget.end_time.isVisibleTo(widget)


def test_switching_to_time_swaps_the_boxes(app):
    widget = picker()
    widget.unit_combo.setCurrentIndex(UNIT_TIME)
    assert widget.unit() == UNIT_TIME
    assert widget.start_time.isVisibleTo(widget) and widget.end_time.isVisibleTo(widget)
    assert not widget.start_spin.isVisibleTo(widget) and not widget.end_spin.isVisibleTo(widget)
    widget.unit_combo.setCurrentIndex(UNIT_FRAMES)
    assert widget.start_spin.isVisibleTo(widget) and not widget.start_time.isVisibleTo(widget)


def test_the_time_boxes_show_the_segment(app):
    widget = picker(1000, fps=25.0)
    assert widget.start_time.text() == "0:00.00"
    assert widget.end_time.text() == "0:39.96"          # frame 999 at 25 fps
    widget.set_segment(250, 500)
    assert widget.start_time.text() == "0:10.00"
    assert widget.end_time.text() == "0:20.00"


def test_typing_a_start_time_sets_the_nearest_frame(app):
    widget = picker(1000, fps=25.0)
    widget.start_time.setValue(12.0)
    assert widget.segment() == (300, 999)
    assert widget.start_time.text() == "0:12.00"


def test_typing_a_time_between_frames_rounds_to_the_nearest(app):
    widget = picker(1000, fps=25.0)
    widget.start_time.setValue(1.03)                    # 25.75 frames
    assert widget.start_spin.value() == 26
    assert widget.start_time.text() == "0:01.04"        # the exact time of frame 26


def test_typing_an_end_time_sets_the_end_frame(app):
    widget = picker(1000, fps=25.0)
    widget.end_time.setValue(20.0)
    assert widget.segment() == (0, 500)


def test_a_typed_time_reports_the_change_once(app):
    widget = picker(1000, fps=25.0)
    heard = []
    widget.changed.connect(lambda first, last: heard.append((first, last)))
    widget.start_time.setValue(4.0)
    assert heard == [(100, 999)]


def test_a_time_that_changes_nothing_reports_nothing(app):
    widget = picker(1000, fps=25.0)
    widget.start_time.setValue(0.01)                    # still frame 0
    heard = []
    widget.changed.connect(lambda first, last: heard.append((first, last)))
    widget.start_time.setValue(0.015)
    assert heard == [] and widget.segment() == (0, 999)


def test_a_time_past_the_video_is_cut_to_the_last_frame(app):
    widget = picker(100, fps=25.0)
    widget.end_time.setMaximum(10000)                   # as if the box allowed it
    widget.end_time.setValue(500.0)
    assert widget.segment()[1] == 99


def test_a_start_time_past_the_end_pushes_the_end_along(app):
    widget = picker(1000, fps=25.0)
    widget.end_time.setValue(10.0)                      # frame 250
    widget.start_time.setValue(20.0)                    # frame 500, past the end
    assert widget.segment() == (500, 500)
    assert widget.end_time.text() == "0:20.00"


def test_an_end_time_before_the_start_pulls_the_start_back(app):
    widget = picker(1000, fps=25.0)
    widget.start_time.setValue(20.0)
    widget.end_time.setValue(8.0)
    assert widget.segment() == (200, 200)
    assert widget.start_time.text() == "0:08.00"


def test_the_frame_boxes_and_the_time_boxes_stay_in_step(app):
    widget = picker(1000, fps=25.0)
    widget.start_spin.setValue(125)
    assert widget.start_time.text() == "0:05.00"
    widget.set_first(150)
    assert widget.start_time.text() == "0:06.00"
    widget.set_last(400)
    assert widget.end_time.text() == "0:16.00"


def test_the_buttons_work_in_time_mode_too(app):
    widget = picker(1000, fps=25.0)
    widget.unit_combo.setCurrentIndex(UNIT_TIME)
    widget.set_first(75)                                # what "Use current" does
    assert widget.start_time.text() == "0:03.00"


def test_the_time_boxes_cannot_leave_the_video(app):
    widget = picker(100, fps=25.0)
    assert widget.start_time.minimum() == 0.0
    assert widget.start_time.maximum() == pytest.approx(99 / 25.0)


def test_the_arrows_step_one_frame(app):
    widget = picker(1000, fps=25.0)
    widget.start_time.stepUp()
    assert widget.segment()[0] == 1
    widget.start_time.stepUp()
    widget.start_time.stepUp()
    assert widget.segment()[0] == 3
    widget.start_time.stepDown()
    assert widget.segment()[0] == 2


def test_stepping_works_at_a_high_frame_rate(app):
    widget = picker(1000, fps=120.0)
    for expected in (1, 2, 3, 4):
        widget.start_time.stepUp()
        assert widget.segment()[0] == expected
    assert widget.start_time.text() == "0:00.033"


def test_a_high_frame_rate_shows_three_decimals(app):
    widget = picker(1000, fps=120.0)
    widget.set_segment(60, 600)
    assert widget.start_time.text() == "0:00.500" and widget.end_time.text() == "0:05.000"


def test_without_a_frame_rate_only_frames_can_be_used(app):
    widget = picker(100, fps=0.0)
    assert not widget.unit_combo.isEnabled()
    assert widget.unit() == UNIT_FRAMES
    widget.start_time.setValue(5.0)                     # does nothing
    assert widget.segment() == (0, 99)


def test_a_video_without_a_frame_rate_forces_frames_back(app):
    widget = picker(100, fps=25.0)
    widget.unit_combo.setCurrentIndex(UNIT_TIME)
    widget.set_video(100, 0.0)
    assert widget.unit() == UNIT_FRAMES and not widget.unit_combo.isEnabled()
    widget.set_video(100, 25.0)
    assert widget.unit_combo.isEnabled()


def test_the_chosen_unit_stays_when_another_video_opens(app):
    widget = picker(100, fps=25.0)
    widget.unit_combo.setCurrentIndex(UNIT_TIME)
    widget.set_video(500, 30.0)
    assert widget.unit() == UNIT_TIME
    assert widget.end_time.text() == format_time(499 / 30.0)


def test_typing_text_into_a_time_box_is_understood(app):
    widget = picker(1000, fps=25.0)
    widget.unit_combo.setCurrentIndex(UNIT_TIME)
    box = widget.start_time
    box.lineEdit().setText("0:12")
    box.interpretText()
    assert widget.segment()[0] == 300


def test_nonsense_typed_into_a_time_box_is_ignored(app):
    widget = picker(1000, fps=25.0)
    widget.set_first(100)
    box = widget.start_time
    box.lineEdit().setText("abc")
    box.interpretText()
    assert widget.segment()[0] == 100


def test_the_time_box_only_lets_time_like_text_through(app):
    from PySide6.QtGui import QValidator
    box = TimeBox()
    assert box.validate("1:30.5", 0)[0] == QValidator.State.Acceptable
    assert box.validate("1:", 0)[0] == QValidator.State.Intermediate
    assert box.validate("x", 0)[0] == QValidator.State.Invalid


def test_the_unit_drop_down_has_a_hint(app):
    assert "time" in picker().unit_combo.toolTip().lower()


def test_a_time_that_lands_on_the_same_frame_is_tidied_up(app):
    widget = picker(1000, fps=25.0)
    widget.set_first(25)                                # 1.00 s
    widget.start_time.setValue(1.01)                    # still frame 25
    assert widget.segment()[0] == 25
    assert widget.start_time.text() == "0:01.00"


def test_half_typed_text_leaves_the_frame_alone(app):
    widget = picker(1000, fps=25.0)
    widget.set_first(100)
    box = widget.start_time
    box.lineEdit().setText("1:")
    box.interpretText()
    assert widget.segment()[0] == 100