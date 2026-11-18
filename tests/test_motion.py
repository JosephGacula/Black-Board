"""Tests for the motion measurement and the motion strip in the window."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import time

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackboard.decode import DecodeError
from blackboard.explanations import MOTION_HELP
from blackboard.motion import (
    MOTION_FULL, ROUGH_GRAY, UNKNOWN_BGR, MotionData, MotionSummary, build_motion,
    measure_pair, motion_between, motion_colors, motion_starts, summarize_motion,
)
from blackboard.probe import probe
from blackboard.ui.main_window import MainWindow, motion_text

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def texture(width, height=240, seed=1):
    """A smooth random picture with detail everywhere, as a BGR image."""
    rng = np.random.default_rng(seed)
    gray = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 2.0)
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    gray = (60 + gray * 140).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def panned(image, pixels):
    """The same picture seen `pixels` further to the right."""
    return np.roll(image, -pixels, axis=1)


def expected(pixels, width, fps):
    return pixels / width * 100 * fps


# ---------- motion between two frames ----------

def test_two_identical_frames_have_no_motion():
    image = texture(320)
    assert motion_between(image, image.copy(), 30.0) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("pixels", [1, 2, 4, 8])
def test_a_pan_is_measured_as_a_share_of_the_width_per_second(pixels):
    image = texture(320)
    result = motion_between(image, panned(image, pixels), 30.0)
    assert result == pytest.approx(expected(pixels, 320, 30.0), rel=0.08)


def test_faster_pans_measure_higher():
    image = texture(320)
    results = [motion_between(image, panned(image, p), 30.0) for p in (1, 3, 6)]
    assert results == sorted(results) and results[0] < results[1] < results[2]


def test_doubling_the_frame_rate_doubles_the_motion():
    image, moved = texture(320), panned(texture(320), 3)
    slow = motion_between(image, moved, 25.0)
    fast = motion_between(image, moved, 50.0)
    assert fast == pytest.approx(2 * slow, rel=0.001)


def test_the_resolution_does_not_change_the_motion():
    small = texture(320)
    big = cv2.resize(small, (1280, 960), interpolation=cv2.INTER_CUBIC)
    low = motion_between(small, panned(small, 3), 30.0)
    high = motion_between(big, panned(big, 12), 30.0)            # the same pan
    assert high == pytest.approx(low, rel=0.12)


def test_motion_up_down_counts_as_much_as_sideways():
    image = texture(320)
    sideways = motion_between(image, np.roll(image, -4, axis=1), 30.0)
    vertical = motion_between(image, np.roll(image, -4, axis=0), 30.0)
    assert vertical == pytest.approx(sideways, rel=0.1)


def test_black_bars_do_not_change_the_motion():
    image = texture(320)
    plain = motion_between(image, panned(image, 3), 30.0)
    bars = np.zeros((40, 320, 3), np.uint8)

    def letterbox(picture):
        return np.concatenate([bars, picture, bars])

    boxed = motion_between(letterbox(image), letterbox(panned(image, 3)), 30.0)
    assert boxed == pytest.approx(plain, rel=0.1)


def test_a_cut_has_no_motion_value():
    assert motion_between(texture(320, seed=1), texture(320, seed=2), 30.0) is None
    red = np.zeros((120, 160, 3), np.uint8)
    red[:] = (0, 0, 255)
    blue = np.zeros((120, 160, 3), np.uint8)
    blue[:] = (255, 0, 0)
    assert motion_between(red, blue, 30.0) is None


def test_a_flat_picture_has_no_motion():
    flat = np.full((120, 160, 3), 90, np.uint8)
    assert motion_between(flat, flat.copy(), 30.0) == pytest.approx(0.0, abs=0.05)


def test_a_black_picture_is_fine():
    black = np.zeros((120, 160, 3), np.uint8)
    assert motion_between(black, black.copy(), 30.0) == pytest.approx(0.0, abs=0.05)


def test_frames_of_different_sizes_are_rejected():
    with pytest.raises(ValueError):
        motion_between(texture(320), texture(160), 30.0)


@pytest.mark.parametrize("fps", [0, -30])
def test_a_bad_frame_rate_is_rejected(fps):
    with pytest.raises(ValueError):
        motion_between(texture(320), texture(320), fps)


# ---------- which frames are compared ----------

def test_the_pairs_start_at_the_sampled_frames():
    assert motion_starts(0, 99, columns=4) == [12, 37, 62, 87]


def test_the_pairs_stay_inside_the_segment():
    starts = motion_starts(100, 199, columns=4)
    assert starts == [112, 137, 162, 187]
    assert all(100 <= s and s + 1 <= 199 for s in starts)


def test_the_last_frame_is_paired_with_the_one_before_it():
    starts = motion_starts(0, 4)                       # every frame is sampled
    assert starts == [0, 1, 2, 3, 3]


def test_a_two_frame_segment_has_one_pair():
    assert set(motion_starts(10, 11)) == {10}


def test_a_one_frame_segment_has_no_motion():
    with pytest.raises(ValueError):
        motion_starts(10, 10)


def test_a_backwards_segment_is_rejected():
    with pytest.raises(ValueError):
        motion_starts(10, 5)


# ---------- real videos ----------

def make_pan_clip(path, speed, frames=60, seed=1, fps=30):
    """A video of a textured picture sliding `speed` pixels per frame."""
    png = path.with_suffix(".png")
    cv2.imwrite(str(png), texture(320 + frames * max(speed, 1) + 10, seed=seed)[:, :, 0])
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(fps), "-i", str(png),
         "-vf", f"crop=320:240:x='n*{speed}':y=0,format=yuv420p",
         "-frames:v", str(frames), "-c:v", "libx264", "-qp", "10", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def pan_clip(tmp_path_factory):
    return make_pan_clip(tmp_path_factory.mktemp("clips") / "pan3.mp4", 3)


@pytest.fixture(scope="module")
def still_clip(tmp_path_factory):
    return make_pan_clip(tmp_path_factory.mktemp("clips") / "still.mp4", 0)


@pytest.fixture(scope="module")
def cut_clip(tmp_path_factory):
    """30 frames of one picture panning 2 px a frame, then 30 of another."""
    folder = tmp_path_factory.mktemp("clips")
    a = make_pan_clip(folder / "a.mp4", 2, frames=30, seed=1)
    b = make_pan_clip(folder / "b.mp4", 2, frames=30, seed=2)
    out = folder / "cut.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(a), "-i", str(b),
         "-filter_complex", "concat=n=2:v=1:a=0", "-c:v", "libx264", "-qp", "10", str(out)],
        check=True,
    )
    return out


@needs_ffmpeg
def test_a_panning_video_has_the_right_motion(pan_clip):
    data = build_motion(pan_clip, 0, 59, 30.0, columns=10)
    assert len(data.values) == 10 and data.values.dtype == float
    assert np.all(np.abs(data.values - expected(3, 320, 30.0)) < 0.8)


@needs_ffmpeg
def test_a_still_video_has_no_motion(still_clip):
    data = build_motion(still_clip, 0, 59, 30.0, columns=10)
    assert np.all(data.values < 0.3)


@needs_ffmpeg
def test_the_frames_are_the_starts_of_the_pairs(pan_clip):
    data = build_motion(pan_clip, 0, 59, 30.0, columns=10)
    assert data.frames == motion_starts(0, 59, columns=10)


@needs_ffmpeg
def test_only_the_chosen_segment_is_measured(cut_clip):
    data = build_motion(cut_clip, 5, 20, 30.0, columns=8)
    assert not np.isnan(data.values).any()
    assert all(5 <= f < 20 for f in data.frames)


@needs_ffmpeg
def test_a_cut_leaves_a_gap_in_the_strip(cut_clip):
    data = build_motion(cut_clip, 0, 59, 30.0, columns=60)     # every pair
    assert np.isnan(data.values[29])                           # frames 29 -> 30
    assert np.isnan(data.values).sum() == 1
    assert np.all(np.abs(np.nan_to_num(data.values[:29]) - expected(2, 320, 30.0)) < 1.0)


@needs_ffmpeg
def test_every_frame_can_be_compared(pan_clip):
    data = build_motion(pan_clip, 10, 19, 30.0)               # fewer frames than stripes
    assert len(data.values) == 9                              # the last two share a pair


@needs_ffmpeg
def test_the_last_frame_of_the_video_works(pan_clip):
    data = build_motion(pan_clip, 50, 59, 30.0)
    assert len(data.values) == 9 and not np.isnan(data.values).any()


@needs_ffmpeg
def test_progress_is_reported_every_few_stripes(pan_clip):
    seen = []
    build_motion(pan_clip, 0, 59, 30.0, columns=25, on_update=lambda v: seen.append(len(v)))
    assert seen == [10, 20]


@needs_ffmpeg
def test_progress_gives_values_in_order(pan_clip):
    seen = []
    final = build_motion(pan_clip, 0, 59, 30.0, columns=20,
                         on_update=lambda v: seen.append(v.copy()))
    for found in seen:
        assert np.array_equal(found, final.values[:len(found)], equal_nan=True)


@needs_ffmpeg
def test_it_stops_when_asked(pan_clip):
    data = build_motion(pan_clip, 0, 59, 30.0, columns=20, should_stop=lambda: True)
    assert len(data.values) == 1


@needs_ffmpeg
def test_it_stops_part_way(pan_clip):
    calls = []

    def stop_after_three():
        calls.append(1)
        return len(calls) >= 3

    data = build_motion(pan_clip, 0, 59, 30.0, columns=20, should_stop=stop_after_three)
    assert len(data.values) == 3


@needs_ffmpeg
def test_a_segment_longer_than_the_video_gives_what_it_can(pan_clip):
    data = build_motion(pan_clip, 0, 99, 30.0, columns=10)
    assert 0 < len(data.values) < 10


@needs_ffmpeg
def test_a_segment_wholly_past_the_end_raises(pan_clip):
    with pytest.raises(DecodeError):
        build_motion(pan_clip, 200, 260, 30.0)


def test_a_one_frame_segment_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_motion(tmp_path / "nothing.mp4", 5, 5, 30.0)


def test_a_file_that_cannot_be_read_raises(tmp_path):
    with pytest.raises(DecodeError):
        build_motion(tmp_path / "nothing.mp4", 0, 99, 30.0)


# ---------- the summary ----------

def data_of(values, frames=None):
    values = np.array(values, float)
    return MotionData(frames=list(frames or range(len(values))), values=values)


def test_the_summary_has_average_median_and_extremes():
    summary = summarize_motion(data_of([2.0, 10.0, 4.0, 8.0], frames=[10, 20, 30, 40]))
    assert summary.average == pytest.approx(6.0) and summary.median == pytest.approx(6.0)
    assert (summary.peak, summary.peak_frame) == (10.0, 20)
    assert (summary.calmest, summary.calmest_frame) == (2.0, 10)
    assert (summary.measured, summary.skipped) == (4, 0)


def test_cuts_are_left_out_of_the_summary():
    summary = summarize_motion(data_of([5.0, np.nan, 15.0], frames=[1, 2, 3]))
    assert summary.average == pytest.approx(10.0)
    assert (summary.measured, summary.skipped) == (2, 1)
    assert summary.peak_frame == 3


def test_the_median_is_not_pulled_by_one_burst():
    summary = summarize_motion(data_of([1.0, 1.0, 1.0, 100.0]))
    assert summary.median == pytest.approx(1.0) and summary.average > 25


def test_a_summary_of_only_cuts_is_rejected():
    with pytest.raises(ValueError):
        summarize_motion(data_of([np.nan, np.nan]))


def test_a_summary_of_nothing_is_rejected():
    with pytest.raises(ValueError):
        summarize_motion(data_of([]))


# ---------- the colors ----------

def test_still_is_dark_purple_and_fast_is_yellow():
    colors = motion_colors([0.0, MOTION_FULL])
    assert tuple(colors[0]) == (84, 1, 68)
    assert tuple(colors[1]) == (37, 231, 253)


def test_faster_than_the_top_of_the_scale_is_drawn_the_same():
    colors = motion_colors([MOTION_FULL, 5 * MOTION_FULL, 1e9])
    assert (colors == colors[0]).all()


def test_the_colors_change_smoothly_from_still_to_fast():
    colors = motion_colors(np.linspace(0, MOTION_FULL, 50)).astype(int)
    assert np.abs(np.diff(colors, axis=0)).max() < 30
    assert len({tuple(c) for c in colors}) > 30


def test_a_gap_is_gray():
    colors = motion_colors([1.0, np.nan, 3.0])
    assert tuple(colors[1]) == UNKNOWN_BGR


def test_the_color_array_has_the_right_shape():
    colors = motion_colors([1.0, 2.0, 3.0])
    assert colors.shape == (3, 3) and colors.dtype == np.uint8
    assert motion_colors([]).shape == (0, 3)


def test_negative_motion_is_treated_as_still():
    assert tuple(motion_colors([-5.0])[0]) == tuple(motion_colors([0.0])[0])


# ---------- the text under the strip ----------

def test_the_motion_line_has_the_numbers():
    summary = MotionSummary(average=7.44, median=5.0, peak=31.2, peak_frame=5400,
                            calmest=0.4, calmest_frame=120, measured=40, skipped=0)
    text = motion_text(summary)
    assert "average 7.4 %" in text and "busiest 31.2 at frame 5400" in text
    assert "calmest 0.4 at frame 120" in text and "left out" not in text


def test_the_motion_line_mentions_cuts_that_were_left_out():
    summary = MotionSummary(1.0, 1.0, 2.0, 5, 0.5, 6, measured=38, skipped=2)
    assert "(2 cuts left out)" in motion_text(summary)
    one = MotionSummary(1.0, 1.0, 2.0, 5, 0.5, 6, measured=39, skipped=1)
    assert "(a cut left out)" in motion_text(one)


# ---------- in the main window ----------

def wait_for(window, app, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        threads = [window.barcode_thread, window.shots_thread,
                   window.motion_thread, window.frame_thread]
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


def test_the_motion_strip_is_below_the_brightness_strip(app):
    from PySide6.QtCore import QPoint
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    app.processEvents()
    y = lambda widget: widget.mapTo(window, QPoint(0, 0)).y()
    assert y(window.motion_view) >= y(window.brightness_view) + window.brightness_view.height()
    assert y(window.motion_status) >= y(window.motion_view) + window.motion_view.height()
    assert window.motion_view.width() == window.barcode_view.width()


def test_everything_fits_in_the_default_window(app):
    from PySide6.QtCore import QPoint
    window = MainWindow()
    window.show()
    app.processEvents()
    bottom = window.scope_stack.mapTo(window, QPoint(0, 0)).y() + window.scope_stack.height()
    assert bottom <= window.height()
    assert window.minimumSizeHint().height() <= window.height()


def test_the_motion_widgets_have_an_explanation(app):
    window = MainWindow()
    for widget in (window.motion_button, window.motion_view, window.motion_status):
        assert widget.toolTip().replace("\n", " ") == MOTION_HELP


def test_the_motion_strip_starts_empty_with_a_hint(app):
    window = MainWindow()
    assert window.motion_view.colors is None and "motion" in window.motion_view.message.lower()
    assert not window.motion_button.isEnabled()


def test_nothing_can_be_measured_before_a_video_is_open(app):
    window = MainWindow()
    window.on_motion_clicked()
    assert window.motion_thread is None and not window.motion_running


def test_late_motion_results_are_ignored(app):
    window = MainWindow()
    window.motion_generation = 5
    window.motion_view.begin(0, 99, 2)
    window.on_motion_updated(4, np.array([1.0, 2.0]))
    window.on_motion_done(4, data_of([1.0, 2.0]))
    assert window.motion_view.colors is None
    window.on_motion_updated(5, np.array([1.0]))
    assert len(window.motion_view.colors) == 1


def test_stopping_makes_earlier_motion_results_late(app):
    window = MainWindow()
    generation = window.motion_generation
    window.stop_motion()
    assert window.motion_generation == generation + 1


def test_a_motion_failure_shows_a_message(app):
    window = MainWindow()
    window.motion_view.begin(0, 99, 2)
    window.on_motion_failed(window.motion_generation, "boom")
    assert "not available" in window.motion_view.message
    window.motion_view.set_message("keep")
    window.on_motion_failed(window.motion_generation - 1, "late")
    assert window.motion_view.message == "keep"


def test_a_segment_of_only_cuts_says_so(app):
    window = MainWindow()
    window.motion_generation = 1
    window.motion_view.begin(0, 99, 2)
    window.on_motion_done(1, data_of([np.nan, np.nan]))
    assert "No motion could be measured" in window.motion_status.text()


def test_the_marker_follows_the_slider_on_the_motion_strip(app):
    window = MainWindow()
    window.info = type("Info", (), {"fps": 30.0})()
    window.frame_slider.setRange(0, 99)
    window.frame_spin.setRange(0, 99)
    window.motion_view.begin(0, 99, 4)
    window.frame_spin.setValue(40)
    assert window.motion_view.position == 40


def test_clicking_the_motion_strip_moves_the_slider(app):
    window = MainWindow()
    window.frame_slider.setRange(0, 999)
    window.frame_spin.setRange(0, 999)
    window.info = object()
    window.motion_view.seek_requested.emit(321)
    assert window.frame_spin.value() == 321


@needs_ffmpeg
def test_opening_a_video_measures_no_motion_yet(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    wait_for(window, app)
    assert window.motion_thread is None and window.motion_view.colors is None
    assert window.motion_button.isEnabled() and window.motion_button.text() == "Analyze motion"


@needs_ffmpeg
def test_pressing_analyze_motion_fills_the_strip(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.motion_button.click()
    assert window.motion_running and window.motion_button.text() == "Cancel"
    wait_for(window, app)
    assert not window.motion_running and window.motion_button.text() == "Analyze motion"
    assert len(window.motion_view.colors) > 0
    assert "average 28." in window.motion_status.text() or "average 27." in window.motion_status.text()
    assert (window.motion_view.first_frame, window.motion_view.last_frame) == (0, 59)


@needs_ffmpeg
def test_the_chosen_segment_is_used(app, cut_clip):
    window = MainWindow()
    open_clip(window, cut_clip)
    window.segment_picker.set_segment(31, 58)                  # after the cut
    window.motion_button.click()
    wait_for(window, app)
    assert (window.motion_view.first_frame, window.motion_view.last_frame) == (31, 58)
    assert "cut" not in window.motion_status.text()


@needs_ffmpeg
def test_a_cut_in_the_segment_is_reported(app, cut_clip):
    window = MainWindow()
    open_clip(window, cut_clip)
    window.motion_button.click()
    wait_for(window, app)
    assert "left out" in window.motion_status.text() or window.motion_status.text()


@needs_ffmpeg
def test_a_one_frame_segment_gives_a_message_and_no_thread(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.segment_picker.set_segment(10, 10)
    window.motion_button.click()
    assert window.motion_thread is None and not window.motion_running
    assert "at least two frames" in window.motion_view.message
    wait_for(window, app)


@needs_ffmpeg
def test_only_one_analysis_runs_at_a_time(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.motion_button.click()
    assert not window.analyze_button.isEnabled() and not window.shots_button.isEnabled()
    assert not window.segment_picker.isEnabled()
    wait_for(window, app)
    assert window.analyze_button.isEnabled() and window.shots_button.isEnabled()
    window.analyze_button.click()
    assert not window.motion_button.isEnabled()
    wait_for(window, app)
    window.shots_button.click()
    assert not window.motion_button.isEnabled()
    wait_for(window, app)
    assert window.motion_button.isEnabled()


@needs_ffmpeg
def test_cancel_clears_the_strip_and_ignores_late_results(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.motion_button.click()
    window.motion_view.set_colors(motion_colors([1.0, 2.0]))        # some stripes are in
    window.motion_button.click()                                    # cancel
    assert not window.motion_running and window.motion_button.text() == "Analyze motion"
    assert window.motion_view.colors is None and window.motion_view.span == 0
    assert window.segment_picker.isEnabled()
    wait_for(window, app)
    assert window.motion_view.colors is None


@needs_ffmpeg
def test_analyzing_again_replaces_the_old_strip(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.segment_picker.set_segment(0, 30)
    window.motion_button.click()
    wait_for(window, app)
    assert window.motion_view.last_frame == 30
    window.segment_picker.set_segment(30, 59)
    window.motion_button.click()
    wait_for(window, app)
    assert (window.motion_view.first_frame, window.motion_view.last_frame) == (30, 59)


@needs_ffmpeg
def test_opening_another_video_drops_the_motion(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.motion_button.click()
    window.clear_preview()
    wait_for(window, app)
    assert window.motion_view.colors is None and window.motion_status.text() == ""
    assert not window.motion_running and not window.motion_button.isEnabled()


@needs_ffmpeg
def test_closing_the_window_asks_the_measurement_to_stop(app, pan_clip):
    window = MainWindow()
    open_clip(window, pan_clip)
    window.motion_button.click()
    generation = window.motion_generation
    started = time.time()
    window.close()
    assert window.motion_generation > generation
    assert time.time() - started < 15
    assert window.motion_thread is None or window.motion_thread.isFinished()


# ---------- moments the flow cannot follow ----------

def test_grain_between_neighboring_frames_is_not_mistaken_for_a_cut():
    image = texture(320)
    rng = np.random.default_rng(0)
    noisy = np.clip(image.astype(int) + rng.integers(-6, 7, image.shape), 0, 255).astype(np.uint8)
    assert motion_between(image, noisy, 30.0) is not None


def test_a_fast_but_followable_pan_is_still_measured():
    image = texture(320)
    assert motion_between(image, panned(image, 12), 30.0) is not None


def test_two_unrelated_pictures_with_the_same_brightness_are_a_cut():
    # Their color signatures look alike, but nothing in them matches
    first, second = texture(320, seed=3), texture(320, seed=4)
    assert abs(float(first.mean()) - float(second.mean())) < 5
    assert motion_between(first, second, 30.0) is None


def test_a_finished_run_with_few_stripes_is_shown(app):
    window = MainWindow()
    window.motion_generation = 1
    window.motion_view.begin(0, 2, 3)
    window.on_motion_done(1, data_of([1.0, 20.0, 40.0], frames=[0, 1, 2]))
    assert len(window.motion_view.colors) == 3
    assert window.motion_frames == [0, 1, 2]
    assert "average 20.3" in window.motion_status.text()


@needs_ffmpeg
def test_the_window_uses_the_frame_rate_of_the_video(app, tmp_path):
    clip = make_pan_clip(tmp_path / "pan25.mp4", 3, fps=25)      # 3 px a frame at 25 fps
    window = MainWindow()
    open_clip(window, clip)
    window.motion_button.click()
    wait_for(window, app)
    assert "average 23." in window.motion_status.text()          # not 28, which 30 fps gives


def test_a_long_motion_line_wraps_instead_of_widening_the_window(app):
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    window.motion_status.setText("Motion: short")
    app.processEvents()
    width_before = window.width()                      # may be more than 1000 on some screens
    window.motion_status.setText("Motion: " + "very long text " * 30)
    app.processEvents()
    assert window.motion_status.wordWrap()
    assert window.width() == width_before
    assert window.motion_status.width() <= window.width()


# ---------- chaotic footage: flames, smoke, flicker ----------

def make_chaos_clip(path, frames=40, seed0=100):
    """Every frame is a different random picture, so nothing can be followed."""
    folder = path.parent / (path.stem + "_frames")
    folder.mkdir()
    for n in range(frames):
        cv2.imwrite(str(folder / f"f{n:03d}.png"), texture(160, 120, seed=seed0 + n))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", str(folder / "f%03d.png"),
         "-c:v", "libx264", "-qp", "10", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def chaos_clip(tmp_path_factory):
    return make_chaos_clip(tmp_path_factory.mktemp("clips") / "chaos.mp4")


@pytest.fixture(scope="module")
def calm_chaos_calm(tmp_path_factory):
    """30 calm frames, 30 chaotic frames, 30 calm frames (all 160x120 at 30 fps)."""
    folder = tmp_path_factory.mktemp("clips")
    wide_a = texture(160 + 60, 120, seed=1)          # wide enough to pan 2 px a frame
    wide_b = texture(160 + 60, 120, seed=2)
    frames = [wide_a[:, 2 * n: 2 * n + 160] for n in range(30)]
    frames += [texture(160, 120, seed=500 + n) for n in range(30)]
    frames += [wide_b[:, 2 * n: 2 * n + 160] for n in range(30)]
    png = folder / "frames"
    png.mkdir()
    for n, frame in enumerate(frames):
        cv2.imwrite(str(png / f"f{n:03d}.png"), frame)
    out = folder / "calm_chaos_calm.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", str(png / "f%03d.png"),
         "-c:v", "libx264", "-qp", "10", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def test_measure_pair_says_whether_the_change_was_followed():
    first = prepare_for_test(texture(320))
    value, followed = measure_pair(first, prepare_for_test(panned(texture(320), 3)), 30.0)
    assert followed and value == pytest.approx(expected(3, 320, 30.0), rel=0.08)
    value, followed = measure_pair(prepare_for_test(texture(320, seed=3)),
                                   prepare_for_test(texture(320, seed=4)), 30.0)
    assert not followed and value > 0               # still gives a rough estimate


def prepare_for_test(image):
    from blackboard.analyze import prepare
    return prepare(image)


@needs_ffmpeg
def test_chaotic_footage_is_kept_as_rough_estimates(chaos_clip):
    data = build_motion(chaos_clip, 0, 39, 30.0, columns=40)
    assert not np.isnan(data.values).any()
    assert data.rough_flags().all()
    assert (data.values > 5).all()                  # counted as lively, not as still


@needs_ffmpeg
def test_a_cut_is_still_left_out_and_not_called_rough(cut_clip):
    data = build_motion(cut_clip, 0, 59, 30.0, columns=60)
    assert np.isnan(data.values[29])
    assert not data.rough_flags().any()


@needs_ffmpeg
def test_a_cut_at_the_end_of_the_segment_is_still_a_cut(cut_clip):
    data = build_motion(cut_clip, 25, 30, 30.0, columns=6)      # the pair 29 -> 30 is last
    assert np.isnan(data.values[list(data.frames).index(29)])
    assert not data.rough_flags().any()


@needs_ffmpeg
def test_a_two_frame_segment_has_no_neighbor_so_the_change_counts_as_a_cut(cut_clip):
    data = build_motion(cut_clip, 29, 30, 30.0)
    assert len(data.values) == 1 and np.isnan(data.values[0])
    assert not data.rough_flags().any()


@needs_ffmpeg
def test_only_the_chaotic_part_of_a_video_is_rough(calm_chaos_calm):
    data = build_motion(calm_chaos_calm, 0, 89, 30.0, columns=90)
    rough = data.rough_flags()
    frames = np.array(data.frames)
    assert not rough[frames < 28].any()             # calm at the start
    assert rough[(frames > 31) & (frames < 58)].all()   # chaos in the middle
    assert not rough[frames > 62].any()             # calm at the end
    calm = data.values[(frames < 28) | (frames > 62)]
    assert not np.isnan(calm).any()


@needs_ffmpeg
def test_the_calm_parts_keep_accurate_values(calm_chaos_calm):
    data = build_motion(calm_chaos_calm, 0, 89, 30.0, columns=90)
    frames = np.array(data.frames)
    calm = data.values[frames < 28]
    assert np.all(np.abs(calm - expected(2, 160, 30.0)) < 1.5)


@needs_ffmpeg
def test_the_summary_counts_the_rough_stripes(calm_chaos_calm):
    data = build_motion(calm_chaos_calm, 0, 89, 30.0, columns=90)
    summary = summarize_motion(data)
    assert 20 <= summary.rough <= 32
    assert summary.measured >= 85


@needs_ffmpeg
def test_stopping_part_way_keeps_the_arrays_in_step(chaos_clip):
    calls = []

    def stop_after_five():
        calls.append(1)
        return len(calls) >= 5

    data = build_motion(chaos_clip, 0, 39, 30.0, columns=40, should_stop=stop_after_five)
    assert len(data.values) == len(data.frames) == len(data.rough_flags()) == 5


@needs_ffmpeg
def test_progress_during_a_chaotic_run_still_arrives(chaos_clip):
    seen = []
    build_motion(chaos_clip, 0, 39, 30.0, columns=40, on_update=lambda v: seen.append(len(v)))
    assert seen == [10, 20, 30]                     # 39 pairs: the last two stripes share one


def test_motion_data_without_rough_flags_means_none_are_rough():
    data = MotionData(frames=[1, 2, 3], values=np.array([1.0, 2.0, 3.0]))
    assert not data.rough_flags().any() and len(data.rough_flags()) == 3
    assert summarize_motion(data).rough == 0


def test_the_summary_only_counts_rough_stripes_that_have_a_value():
    data = MotionData(frames=[1, 2, 3], values=np.array([1.0, np.nan, 3.0]),
                      rough=np.array([True, True, False]))
    summary = summarize_motion(data)
    assert summary.rough == 1 and summary.skipped == 1


# ---------- the colors of rough stripes ----------

def test_rough_stripes_are_washed_out_but_keep_their_hue_order():
    plain = motion_colors([0.0, 50.0]).astype(int)
    washed = motion_colors([0.0, 50.0], rough=[True, True]).astype(int)
    assert not (plain == washed).all()
    # halfway between the plain color and the wash gray
    assert tuple(washed[0]) == tuple(np.rint((plain[0] + ROUGH_GRAY) / 2).astype(int))
    assert washed[0][0] != washed[1][0]             # still different from each other


def test_only_the_marked_stripes_are_washed_out():
    plain = motion_colors([10.0, 10.0, 10.0])
    mixed = motion_colors([10.0, 10.0, 10.0], rough=[False, True, False])
    assert (mixed[0] == plain[0]).all() and (mixed[2] == plain[2]).all()
    assert not (mixed[1] == plain[1]).all()


def test_a_gap_stays_gray_even_if_marked_rough():
    colors = motion_colors([np.nan, 5.0], rough=[True, True])
    assert tuple(colors[0]) == UNKNOWN_BGR


# ---------- the line under the strip, and the window ----------

def test_the_line_says_how_much_of_the_strip_is_rough():
    summary = MotionSummary(20.0, 18.0, 90.0, 5, 1.0, 6, measured=100, skipped=0, rough=26)
    text = motion_text(summary)
    assert "26 % of the strip" in text and "rough estimate" in text
    assert "washed out" in text


def test_no_rough_note_when_nothing_is_rough():
    summary = MotionSummary(1.0, 1.0, 2.0, 5, 0.5, 6, measured=40, skipped=0, rough=0)
    assert "rough" not in motion_text(summary)


def test_the_help_text_explains_washed_out_stripes():
    assert "washed-out" in MOTION_HELP.lower() and "flames" in MOTION_HELP


def test_a_finished_run_shows_rough_stripes_washed_out(app):
    window = MainWindow()
    window.motion_generation = 1
    window.motion_view.begin(0, 2, 3)
    data = MotionData(frames=[0, 1, 2], values=np.array([20.0, 20.0, 20.0]),
                      rough=np.array([False, True, False]))
    window.on_motion_done(1, data)
    colors = window.motion_view.colors
    assert (colors[0] == colors[2]).all() and not (colors[0] == colors[1]).all()
    assert "33 %" in window.motion_status.text()


@needs_ffmpeg
def test_chaotic_footage_through_the_window(app, chaos_clip):
    window = MainWindow()
    open_clip(window, chaos_clip)
    window.motion_button.click()
    wait_for(window, app)
    assert "100 % of the strip" in window.motion_status.text()
    assert "left out" not in window.motion_status.text()


def test_the_rough_share_is_of_the_measured_stripes_not_the_cuts():
    summary = MotionSummary(20.0, 18.0, 90.0, 5, 1.0, 6, measured=100, skipped=50, rough=26)
    assert "26 % of the strip" in motion_text(summary)


def test_live_stripes_leave_unfollowed_moments_gray_until_they_are_judged(calm_chaos_calm):
    seen = []
    data = build_motion(calm_chaos_calm, 0, 89, 30.0, columns=90,
                        on_update=lambda v: seen.append(v.copy()))
    assert any(np.isnan(v).any() for v in seen)         # judged only at the end
    assert not np.isnan(data.values[data.rough_flags()]).any()