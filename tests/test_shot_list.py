"""Tests for finding the shots of a segment, and the shot list in the window."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import time

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from blackboard.decode import DecodeError
from blackboard.explanations import SHOTS_HELP
from blackboard.probe import probe
from blackboard.shots import (
    PROGRESS_EVERY, Shot, ShotSummary, detect_segment_shots, summarize_shots,
)
from blackboard.ui.main_window import MainWindow
from blackboard.ui.segment_picker import BAND_BACKGROUND, BAND_CUT, SegmentBand
from blackboard.ui.shot_list import ShotListView, seconds_text, summary_text

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def three_shots(tmp_path_factory):
    """90 frames: red (0-29), blue (30-59), green (60-89)."""
    path = tmp_path_factory.mktemp("clips") / "three.mp4"
    inputs = []
    for color in ("red", "blue", "lime"):
        inputs += ["-f", "lavfi", "-i", f"color=c={color}:s=160x120:d=1:r=30"]
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", "concat=n=3:v=1:a=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def slow_clip(tmp_path_factory):
    """50 frames at 25 fps: red (0-24), blue (25-49)."""
    path = tmp_path_factory.mktemp("clips") / "slow.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=red:s=160x120:d=1:r=25",
         "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=1:r=25",
         "-filter_complex", "concat=n=2:v=1:a=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def make_shots(lengths, fps=30.0):
    shots, start = [], 0
    for number, frames in enumerate(lengths, 1):
        shots.append(Shot(number, start, start + frames - 1, frames, frames / fps))
        start += frames
    return shots


def wait_for(window, app, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        threads = [window.barcode_thread, window.shots_thread, window.frame_thread]
        if all(t is None or t.isFinished() for t in threads):
            break
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)


def open_clip(window, path):
    window.current_path = str(path)
    window.setup_preview(probe(path))


# ---------- detect_segment_shots ----------

@needs_ffmpeg
def test_the_shots_of_a_whole_clip(three_shots):
    result = detect_segment_shots(three_shots, 0, 89, 30.0)
    assert result.cuts == [30, 60]
    assert [(s.start, s.end) for s in result.shots] == [(0, 29), (30, 59), (60, 89)]
    assert (result.first, result.last) == (0, 89)
    assert result.summary.count == 3 and result.summary.average == pytest.approx(1.0)


@needs_ffmpeg
def test_only_the_chosen_segment_is_looked_at(three_shots):
    result = detect_segment_shots(three_shots, 20, 70, 30.0)
    assert result.cuts == [30, 60]
    assert [(s.start, s.end) for s in result.shots] == [(20, 29), (30, 59), (60, 70)]


@needs_ffmpeg
def test_a_segment_with_no_cut_is_one_shot(three_shots):
    result = detect_segment_shots(three_shots, 35, 55, 30.0)
    assert result.cuts == [] and len(result.shots) == 1
    assert (result.shots[0].start, result.shots[0].end) == (35, 55)
    assert result.summary.cuts_per_minute == 0


@needs_ffmpeg
def test_a_cut_on_the_first_frame_of_the_segment_is_not_found(three_shots):
    # Frame 30 starts a new shot, but nothing in the segment is before it
    result = detect_segment_shots(three_shots, 30, 89, 30.0)
    assert result.cuts == [60]


@needs_ffmpeg
def test_a_one_frame_segment_works(three_shots):
    result = detect_segment_shots(three_shots, 40, 40, 30.0)
    assert len(result.shots) == 1 and result.shots[0].frames == 1


@needs_ffmpeg
def test_the_frame_rate_sets_the_lengths_in_seconds(three_shots):
    result = detect_segment_shots(three_shots, 0, 89, 60.0)
    assert result.shots[0].seconds == pytest.approx(0.5)


@needs_ffmpeg
def test_progress_is_reported_in_percent(three_shots):
    seen = []
    detect_segment_shots(three_shots, 0, 89, 30.0, on_progress=seen.append)
    assert seen == [int(n * PROGRESS_EVERY * 100 / 90) for n in range(1, 90 // PROGRESS_EVERY + 1)]
    assert seen == sorted(seen) and all(0 <= p <= 100 for p in seen)


@needs_ffmpeg
def test_it_stops_when_asked_and_gives_nothing(three_shots):
    assert detect_segment_shots(three_shots, 0, 89, 30.0, should_stop=lambda: True) is None


@needs_ffmpeg
def test_it_stops_part_way(three_shots):
    calls = []

    def stop_after_ten():
        calls.append(1)
        return len(calls) >= 10

    assert detect_segment_shots(three_shots, 0, 89, 30.0, should_stop=stop_after_ten) is None
    assert len(calls) == 10


@needs_ffmpeg
def test_a_segment_past_the_end_raises(three_shots):
    with pytest.raises(DecodeError):
        detect_segment_shots(three_shots, 0, 500, 30.0)


@pytest.mark.parametrize("first,last", [(10, 5), (-1, 5)])
def test_a_bad_segment_is_rejected(tmp_path, first, last):
    with pytest.raises(ValueError):
        detect_segment_shots(tmp_path / "nothing.mp4", first, last, 30.0)


def test_a_file_that_cannot_be_read_raises(tmp_path):
    with pytest.raises(DecodeError):
        detect_segment_shots(tmp_path / "nothing.mp4", 0, 9, 30.0)


# ---------- the list widget ----------

def test_the_list_starts_empty(app):
    view = ShotListView()
    assert view.table.rowCount() == 0 and view.summary.text() == ""


def test_the_table_shows_one_row_per_shot(app):
    shots = make_shots([30, 45, 15])
    view = ShotListView()
    view.set_shots(shots, summarize_shots(shots), 0, 89)
    assert view.table.rowCount() == 3
    row = [view.table.item(1, column).text() for column in range(4)]
    assert row == ["2", "30", "74", "1.50 s"]


def test_a_new_result_replaces_the_old_rows(app):
    view = ShotListView()
    first = make_shots([30, 30, 30])
    view.set_shots(first, summarize_shots(first), 0, 89)
    second = make_shots([60])
    view.set_shots(second, summarize_shots(second), 0, 59)
    assert view.table.rowCount() == 1 and view.starts == [0]


def test_clearing_empties_the_table_and_shows_a_message(app):
    view = ShotListView()
    shots = make_shots([30, 30])
    view.set_shots(shots, summarize_shots(shots), 0, 59)
    view.clear("Nothing here")
    assert view.table.rowCount() == 0 and view.starts == []
    assert view.summary.text() == "Nothing here"


def test_the_summary_has_the_numbers():
    shots = make_shots([30, 60, 90])
    text = summary_text(summarize_shots(shots), 100, 279)
    assert "3 shots in frames 100 to 279" in text
    assert "Average 2.00 s, median 2.00 s" in text
    assert "Shortest 1.00 s, longest 3.00 s" in text
    assert "Cutting pace: 20.0 cuts per minute" in text


def test_one_shot_is_not_plural():
    shots = make_shots([30])
    assert summary_text(summarize_shots(shots), 0, 29).startswith("1 shot in")


def test_seconds_have_two_decimals():
    assert seconds_text(1.0) == "1.00 s" and seconds_text(0.3333) == "0.33 s"


def test_clicking_a_row_asks_to_jump_to_the_start_of_that_shot(app):
    shots = make_shots([30, 45, 15])
    view = ShotListView()
    view.set_shots(shots, summarize_shots(shots), 0, 89)
    asked = []
    view.seek_requested.connect(asked.append)
    view.table.cellClicked.emit(1, 0)
    view.table.cellClicked.emit(2, 3)
    assert asked == [30, 75]


def test_a_real_mouse_click_on_a_row_works(app):
    shots = make_shots([30, 45, 15])
    view = ShotListView()
    view.show()
    view.set_shots(shots, summarize_shots(shots), 0, 89)
    asked = []
    view.seek_requested.connect(asked.append)
    y = view.table.rowViewportPosition(2) + 5
    QTest.mouseClick(view.table.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(20, y))
    assert asked == [75]


def test_the_row_of_a_frame_is_the_shot_that_holds_it(app):
    shots = make_shots([30, 45, 15])                 # starts 0, 30, 75
    view = ShotListView()
    view.set_shots(shots, summarize_shots(shots), 0, 89)
    assert [view.row_of(f) for f in (0, 29, 30, 74, 75, 89)] == [0, 0, 1, 1, 2, 2]


def test_a_frame_outside_the_shots_has_no_row(app):
    shots = make_shots([30, 30])
    view = ShotListView()
    assert view.row_of(5) is None                    # no shots yet
    view.set_shots(shots, summarize_shots(shots), 100, 159)
    view.starts = [100, 130]
    assert view.row_of(50) is None


def test_the_current_shot_is_highlighted(app):
    shots = make_shots([30, 45, 15])
    view = ShotListView()
    view.set_shots(shots, summarize_shots(shots), 0, 89)
    view.set_position(40)
    assert view.table.currentRow() == 1
    assert [i.row() for i in view.table.selectedItems()] == [1, 1, 1, 1]
    view.set_position(80)
    assert view.table.currentRow() == 2


def test_moving_the_highlight_does_not_ask_to_jump(app):
    shots = make_shots([30, 45, 15])
    view = ShotListView()
    view.set_shots(shots, summarize_shots(shots), 0, 89)
    asked = []
    view.seek_requested.connect(asked.append)
    view.set_position(40)
    assert asked == []


def test_a_frame_before_the_first_shot_clears_the_highlight(app):
    shots = make_shots([30, 30])
    view = ShotListView()
    view.set_shots(shots, summarize_shots(shots), 100, 159)
    view.starts = [100, 130]
    view.set_position(120)
    assert view.table.selectedItems()
    view.set_position(10)
    assert not view.table.selectedItems()


# ---------- cut marks on the band ----------

def render(band, width=400):
    band.resize(width, 8)
    image = band.grab().toImage()
    return lambda x, y=2: (image.pixelColor(int(x), y).red(),
                           image.pixelColor(int(x), y).green(),
                           image.pixelColor(int(x), y).blue())


def test_cut_marks_are_drawn_where_the_shots_start(app):
    band = SegmentBand()
    band.set_total(1000)
    band.set_segment(0, 0)
    band.set_cuts([250, 500])
    band.resize(400, 8)
    assert band.cut_xs() == [100, 200]
    pixel = render(band)
    yellow = (BAND_CUT.red(), BAND_CUT.green(), BAND_CUT.blue())
    assert pixel(100) == yellow and pixel(200) == yellow
    assert pixel(150) == (BAND_BACKGROUND.red(), BAND_BACKGROUND.green(), BAND_BACKGROUND.blue())


def test_cut_marks_can_be_removed(app):
    band = SegmentBand()
    band.set_total(1000)
    band.set_cuts([250])
    band.set_cuts([])
    assert band.cut_xs() == []


def test_cuts_outside_the_video_are_not_drawn(app):
    band = SegmentBand()
    band.set_total(1000)
    band.resize(400, 8)
    band.set_cuts([-5, 1000, 5000, 500])
    assert band.cut_xs() == [200]


def test_a_cut_near_the_end_stays_inside_the_band(app):
    band = SegmentBand()
    band.set_total(1000)
    band.resize(400, 8)
    band.set_cuts([999])
    assert band.cut_xs() == [399]


def test_opening_or_clearing_removes_the_cut_marks(app):
    band = SegmentBand()
    band.set_total(1000)
    band.set_cuts([250])
    band.set_total(500)
    assert band.cuts == []
    band.set_cuts([100])
    band.clear()
    assert band.cuts == []


# ---------- in the main window ----------

def test_the_window_has_a_shot_list_under_the_video_details(app):
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    app.processEvents()
    y = lambda widget: widget.mapTo(window, QPoint(0, 0)).y()
    last_field = window.fields["Audio length"]
    assert y(window.shot_list) > y(last_field)
    assert window.shot_list.mapTo(window, QPoint(0, 0)).y() > window.preview.mapTo(window, QPoint(0, 0)).y()
    bottom = y(window.shot_list) + window.shot_list.height()
    assert bottom <= window.height()


def test_the_shot_list_has_an_explanation(app):
    window = MainWindow()
    for widget in (window.shots_button, window.shot_list, window.shots_title):
        assert widget.toolTip().replace("\n", " ") == SHOTS_HELP


def test_nothing_can_be_detected_before_a_video_is_open(app):
    window = MainWindow()
    assert not window.shots_button.isEnabled()
    window.on_shots_clicked()
    assert window.shots_thread is None and not window.shots_running


def test_a_late_shot_result_is_ignored(app):
    window = MainWindow()
    window.shots_generation = 5
    shots = make_shots([30, 30])
    from blackboard.shots import SegmentShots
    late = SegmentShots(0, 59, shots, summarize_shots(shots), [30])
    window.on_shots_done(4, late)
    assert window.shot_list.table.rowCount() == 0
    window.on_shots_done(5, late)
    assert window.shot_list.table.rowCount() == 2
    assert window.segment_band.cuts == [30]


def test_stopping_makes_earlier_shot_results_late(app):
    window = MainWindow()
    generation = window.shots_generation
    window.stop_shots()
    assert window.shots_generation == generation + 1


def test_a_shot_failure_shows_a_message(app):
    window = MainWindow()
    window.on_shots_failed(window.shots_generation, "boom")
    assert "could not be found" in window.shot_list.summary.text()
    before = window.shot_list.summary.text()
    window.on_shots_failed(window.shots_generation - 1, "late")
    assert window.shot_list.summary.text() == before


def test_a_late_progress_message_is_ignored(app):
    window = MainWindow()
    window.shot_list.clear("Hello")
    window.on_shots_progress(window.shots_generation - 1, 50)
    assert window.shot_list.summary.text() == "Hello"
    window.on_shots_progress(window.shots_generation, 50)
    assert "50 %" in window.shot_list.summary.text()


@needs_ffmpeg
def test_opening_a_video_finds_no_shots_yet(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    wait_for(window, app)
    assert window.shots_thread is None and window.shot_list.table.rowCount() == 0
    assert window.shots_button.isEnabled()
    assert window.shots_button.text() == "Detect shots"


@needs_ffmpeg
def test_pressing_detect_shots_fills_the_list_and_marks_the_band(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    assert window.shots_running and window.shots_button.text() == "Cancel"
    wait_for(window, app)
    assert not window.shots_running and window.shots_button.text() == "Detect shots"
    table = window.shot_list.table
    assert table.rowCount() == 3
    assert [table.item(r, 1).text() for r in range(3)] == ["0", "30", "60"]
    assert window.segment_band.cuts == [30, 60]
    assert "3 shots in frames 0 to 89" in window.shot_list.summary.text()


@needs_ffmpeg
def test_it_uses_only_the_chosen_segment(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.segment_picker.set_segment(20, 70)
    window.shots_button.click()
    wait_for(window, app)
    table = window.shot_list.table
    assert [table.item(r, 1).text() for r in range(table.rowCount())] == ["20", "30", "60"]
    assert "frames 20 to 70" in window.shot_list.summary.text()


@needs_ffmpeg
def test_clicking_a_shot_moves_the_slider(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    wait_for(window, app)
    window.shot_list.table.cellClicked.emit(1, 0)
    assert window.frame_spin.value() == 30
    wait_for(window, app)


@needs_ffmpeg
def test_moving_the_slider_highlights_the_shot(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    wait_for(window, app)
    window.frame_spin.setValue(65)
    assert window.shot_list.table.currentRow() == 2
    window.frame_spin.setValue(5)
    assert window.shot_list.table.currentRow() == 0
    wait_for(window, app)


@needs_ffmpeg
def test_only_one_analysis_runs_at_a_time(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    assert not window.analyze_button.isEnabled()
    assert not window.segment_picker.isEnabled()
    wait_for(window, app)
    assert window.analyze_button.isEnabled() and window.segment_picker.isEnabled()
    window.analyze_button.click()
    assert not window.shots_button.isEnabled()
    wait_for(window, app)
    assert window.shots_button.isEnabled()


@needs_ffmpeg
def test_cancel_stops_the_detection_and_clears_the_list(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    window.shots_button.click()                         # cancel
    assert not window.shots_running and window.shots_button.text() == "Detect shots"
    assert "Cancelled" in window.shot_list.summary.text()
    wait_for(window, app)
    assert window.shot_list.table.rowCount() == 0       # late result ignored
    assert window.segment_picker.isEnabled()


@needs_ffmpeg
def test_detecting_again_replaces_the_old_marks(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    wait_for(window, app)
    window.segment_picker.set_segment(0, 40)
    window.shots_button.click()
    wait_for(window, app)
    assert window.segment_band.cuts == [30]
    assert window.shot_list.table.rowCount() == 2


@needs_ffmpeg
def test_opening_another_video_drops_the_shots(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    wait_for(window, app)
    window.clear_preview()
    assert window.shot_list.table.rowCount() == 0
    assert window.segment_band.cuts == []
    assert not window.shots_button.isEnabled()
    wait_for(window, app)


@needs_ffmpeg
def test_closing_the_window_while_detecting_does_not_hang(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    started = time.time()
    window.close()
    assert time.time() - started < 10
    assert window.shots_thread is None or window.shots_thread.isFinished()


@needs_ffmpeg
def test_a_shot_failure_on_a_real_run_unlocks_the_controls(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.segment_picker.start_spin.setMaximum(500)
    window.segment_picker.end_spin.setMaximum(500)
    window.segment_picker.set_segment(0, 400)           # past the end of the video
    window.shots_button.click()
    wait_for(window, app)
    assert not window.shots_running and window.segment_picker.isEnabled()
    assert "could not be found" in window.shot_list.summary.text()


@needs_ffmpeg
def test_shot_lengths_use_the_frame_rate_of_the_video(app, slow_clip):
    window = MainWindow()
    open_clip(window, slow_clip)
    window.shots_button.click()
    wait_for(window, app)
    table = window.shot_list.table
    assert [table.item(r, 3).text() for r in range(2)] == ["1.00 s", "1.00 s"]


@needs_ffmpeg
def test_starting_a_new_detection_removes_the_old_marks_at_once(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.segment_band.set_cuts([30, 60])
    window.shots_button.click()
    assert window.segment_band.cuts == []
    wait_for(window, app)


@needs_ffmpeg
def test_closing_the_window_asks_a_running_detection_to_stop(app, three_shots):
    window = MainWindow()
    open_clip(window, three_shots)
    window.shots_button.click()
    generation = window.shots_generation
    window.close()
    assert window.shots_generation > generation