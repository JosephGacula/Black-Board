"""Tests for the platform window and the button that opens it."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import time

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from blackboard.explanations import PLATFORM_HELP
from blackboard.platforms import (
    DEFAULT_KEYS, MB, PLACEMENTS, SPECS_CHECKED, Clip, check_places,
)
from blackboard.probe import probe
from blackboard.ui.main_window import PLATFORMS_TEXT, MainWindow
from blackboard.ui.platform_dialog import NOTHING_PICKED, PlatformDialog

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def vertical():
    return Clip(1080, 1920, 30.0, 30.0, 20 * MB, "mp4")


def card_texts(dialog):
    texts = []
    for n in range(dialog.cards.count()):
        widget = dialog.cards.itemAt(n).widget()
        texts.append(" ".join(label.text() for label in widget.findChildren(QLabel)))
    return texts


# ---------- the window ----------

def test_it_starts_with_the_default_places_ticked(app):
    dialog = PlatformDialog(vertical())
    assert sorted(dialog.selected_keys()) == sorted(DEFAULT_KEYS)
    assert len(dialog.reports) == len(DEFAULT_KEYS)
    assert dialog.cards.count() == len(DEFAULT_KEYS)


def test_every_place_has_a_check_box(app):
    dialog = PlatformDialog(vertical())
    assert list(dialog.boxes) == [p.key for p in PLACEMENTS]
    assert dialog.boxes["ig_reels"].text() == "Instagram Reels"


def test_the_first_line_describes_the_video(app):
    dialog = PlatformDialog(vertical())
    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert any("1080x1920 (9:16), 30.00 fps, 30 s, 20 MB" in t for t in labels)


def test_a_given_selection_is_used(app):
    dialog = PlatformDialog(vertical(), selected=["youtube"])
    assert dialog.selected_keys() == ["youtube"]
    assert dialog.cards.count() == 1


def test_ticking_a_box_adds_a_card(app):
    dialog = PlatformDialog(vertical(), selected=["youtube"])
    dialog.boxes["tiktok"].setChecked(True)
    assert sorted(dialog.selected_keys()) == ["tiktok", "youtube"]
    assert dialog.cards.count() == 2
    dialog.boxes["youtube"].setChecked(False)
    assert dialog.cards.count() == 1 and "TikTok" in card_texts(dialog)[0]


def test_the_best_fit_comes_first(app):
    dialog = PlatformDialog(vertical(), selected=["youtube", "ig_reels"])
    first, second = card_texts(dialog)
    assert "Instagram Reels" in first and "Great fit" in first
    assert "YouTube video" in second and "Poor fit" in second


def test_a_card_lists_every_check_with_its_message(app):
    text = card_texts(PlatformDialog(vertical(), selected=["ig_reels"]))[0]
    for name in ("Picture shape", "Resolution", "Length", "Frame rate", "File size", "File type"):
        assert name in text
    text = card_texts(PlatformDialog(vertical(), selected=["youtube"]))[0]
    assert "fills 32 % of the screen" in text


def test_status_is_shown_in_words_and_symbols_not_only_color(app):
    dialog = PlatformDialog(vertical(), selected=["youtube", "ig_reels"])
    text = " ".join(card_texts(dialog))
    assert "✓" in text and "✗" in text and "Great fit" in text and "Poor fit" in text


def test_none_and_all_buttons(app):
    dialog = PlatformDialog(vertical())
    dialog.pick_all(False)
    assert dialog.selected_keys() == [] and dialog.reports == []
    assert dialog.cards.itemAt(0).widget().text() == NOTHING_PICKED
    dialog.pick_all(True)
    assert len(dialog.selected_keys()) == len(PLACEMENTS)
    assert dialog.cards.count() == len(PLACEMENTS)


def test_the_all_and_none_buttons_are_wired(app):
    from PySide6.QtWidgets import QPushButton
    dialog = PlatformDialog(vertical())
    button = {b.text(): b for b in dialog.findChildren(QPushButton)}
    button["None"].click()
    assert dialog.selected_keys() == []
    button["All"].click()
    assert len(dialog.selected_keys()) == len(PLACEMENTS)


def test_the_window_says_when_the_specs_were_checked(app):
    dialog = PlatformDialog(vertical())
    assert any(SPECS_CHECKED in label.text() for label in dialog.findChildren(QLabel))


def test_the_cards_match_the_checker(app):
    dialog = PlatformDialog(vertical(), selected=["ig_reels", "tiktok"])
    expected = check_places(vertical(), ["ig_reels", "tiktok"])
    assert [r.placement.key for r in dialog.reports] == [r.placement.key for r in expected]


def test_a_long_message_wraps(app):
    dialog = PlatformDialog(Clip(1920, 1080, 30.0, 30.0, 20 * MB, "mp4"), selected=["ig_reels"])
    messages = [label for label in dialog.findChildren(QLabel) if "Picture shape" in label.text()]
    assert messages and messages[0].wordWrap()


def test_the_help_text_is_a_tooltip_on_the_first_line(app):
    dialog = PlatformDialog(vertical(), help_text="HELP")
    first = [l for l in dialog.findChildren(QLabel) if l.text().startswith("Checking")][0]
    assert first.toolTip() == "HELP"


# ---------- the button in the main window ----------

def make_clip(path, size="320x180", seconds=1):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size={size}:rate=30:duration={seconds}", "-pix_fmt", "yuv420p",
                    str(path)], check=True)


def settle(window, app, seconds=30):
    """Let the background threads finish, so no window is destroyed while one runs."""
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


def open_clip(window, path, app):
    window.current_path = str(path)
    window.setup_preview(probe(path))
    settle(window, app)


def test_the_button_is_off_until_a_video_is_open(app):
    window = MainWindow()
    assert window.platforms_button.text() == PLATFORMS_TEXT
    assert not window.platforms_button.isEnabled()


def test_the_button_has_the_help_tooltip(app):
    window = MainWindow()
    assert "aspect ratio" in window.platforms_button.toolTip()
    assert PLATFORM_HELP.split()[0] in window.platforms_button.toolTip()


def test_the_help_text_mentions_what_is_checked():
    for word in ("shape", "resolution", "length", "frame rate", "file size", "file type", "black bars"):
        assert word in PLATFORM_HELP


@needs_ffmpeg
def test_the_button_works_once_a_video_is_open(app, tmp_path, monkeypatch):
    path = tmp_path / "wide.mp4"
    make_clip(path)
    window = MainWindow()
    open_clip(window, path, app)
    assert window.platforms_button.isEnabled()

    seen = {}

    def fake_exec(dialog):
        seen["clip"] = dialog.clip
        seen["selected"] = dialog.selected_keys()
        dialog.boxes["x"].setChecked(True)
        return 0

    monkeypatch.setattr(PlatformDialog, "exec", fake_exec)
    window.platforms_button.click()
    assert (seen["clip"].width, seen["clip"].height) == (320, 180)
    assert seen["clip"].extension == "mp4" and seen["clip"].duration == pytest.approx(1.0, abs=0.1)
    assert sorted(seen["selected"]) == sorted(DEFAULT_KEYS)
    # what was ticked is remembered for the next time
    assert "x" in window.platform_choices


@needs_ffmpeg
def test_the_choices_are_remembered_between_openings(app, tmp_path, monkeypatch):
    path = tmp_path / "wide.mp4"
    make_clip(path)
    window = MainWindow()
    open_clip(window, path, app)
    window.platform_choices = ["youtube"]
    starts = []
    monkeypatch.setattr(PlatformDialog, "exec",
                        lambda dialog: starts.append(dialog.selected_keys()) or 0)
    window.platforms_button.click()
    window.platforms_button.click()
    assert starts == [["youtube"], ["youtube"]]


@needs_ffmpeg
def test_clicking_without_a_video_does_nothing(app, monkeypatch):
    window = MainWindow()
    opened = []
    monkeypatch.setattr(PlatformDialog, "exec", lambda dialog: opened.append(1) or 0)
    window.on_platforms_clicked()
    assert opened == []


@needs_ffmpeg
def test_the_button_stays_usable_while_an_analysis_runs(app, tmp_path):
    path = tmp_path / "wide.mp4"
    make_clip(path)
    window = MainWindow()
    open_clip(window, path, app)
    window.motion_running = True
    window.refresh_controls()
    assert window.platforms_button.isEnabled()
    assert not window.analyze_button.isEnabled()
    window.motion_running = False


def test_the_button_is_in_the_row_with_the_other_buttons(app):
    window = MainWindow()
    window.resize(1000, 900)
    window.show()
    app.processEvents()
    assert abs(window.platforms_button.y() - window.motion_button.y()) < 4
    assert window.platforms_button.x() > window.motion_button.x()


# ---------- black bars found while the window is open ----------

from blackboard.platforms import Bars
from blackboard.ui.platform_dialog import SEARCH_FAILED, SEARCHING
import blackboard.ui.main_window as main_window_module


def barred_clip():
    return Clip(3840, 2160, 60.0, 42.2, 387 * MB, "mp4", bars=Bars(0, 0, 0.342, 0.342, 16 / 9))


def test_the_window_says_it_is_looking_for_bars(app):
    assert PlatformDialog(vertical(), searching=True).status.text() == SEARCHING
    assert PlatformDialog(vertical()).status.text() == ""


def test_finding_the_bars_updates_the_cards_and_the_first_line(app):
    dialog = PlatformDialog(Clip(3840, 2160, 60.0, 42.2, 387 * MB, "mp4"),
                            selected=["ig_reels", "youtube"], searching=True)
    first = card_texts(dialog)
    assert "Black bars" not in " ".join(first) and "Poor fit" in first[-1]
    dialog.set_clip(barred_clip())
    assert dialog.status.text() == ""
    assert "picture inside black bars 1213x2160 (9:16)" in dialog.intro.text()
    cards = card_texts(dialog)
    assert "Instagram Reels" in cards[0] and "Black bars" in cards[0]
    assert "Works, with caveats" in cards[0]
    assert dialog.cards.count() == 2
    assert sorted(dialog.selected_keys()) == ["ig_reels", "youtube"]


def test_a_failed_search_is_said_plainly(app):
    dialog = PlatformDialog(vertical(), searching=True)
    dialog.search_failed()
    assert dialog.status.text() == SEARCH_FAILED


def pillar_file(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=0x808080:size=202x360:rate=30:duration=2,noise=alls=25:allf=t",
                    "-vf", "pad=640:360:219:0:black", "-pix_fmt", "yuv420p", str(path)],
                   check=True)


def wait_until(app, condition, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline and not condition():
        app.processEvents()
        time.sleep(0.02)
    return condition()


@needs_ffmpeg
def test_the_window_finds_the_bars_by_itself(app, tmp_path, monkeypatch):
    path = tmp_path / "pillar.mp4"
    pillar_file(path)
    window = MainWindow()
    open_clip(window, path, app)
    seen = {}

    def fake_exec(dialog):
        seen["start"] = dialog.status.text()
        seen["found"] = wait_until(app, lambda: dialog.clip.has_bars)
        seen["clip"] = dialog.clip
        seen["end"] = dialog.status.text()
        return 0

    monkeypatch.setattr(PlatformDialog, "exec", fake_exec)
    window.platforms_button.click()
    assert seen["start"] == SEARCHING and seen["found"] and seen["end"] == ""
    assert abs(seen["clip"].picture_width - 202) <= 14 and seen["clip"].picture_height == 360
    assert window.bars_found[str(path)].matter
    settle(window, app)


@needs_ffmpeg
def test_the_bars_are_remembered_for_the_next_opening(app, tmp_path, monkeypatch):
    path = tmp_path / "pillar.mp4"
    pillar_file(path)
    window = MainWindow()
    open_clip(window, path, app)
    starts = []
    monkeypatch.setattr(PlatformDialog, "exec", lambda dialog: wait_until(app, lambda: dialog.clip.has_bars) and 0)
    window.platforms_button.click()

    def second(dialog):
        starts.append((dialog.status.text(), dialog.clip.has_bars))
        return 0

    monkeypatch.setattr(PlatformDialog, "exec", second)
    monkeypatch.setattr(main_window_module, "BarsThread", None)       # must not be needed
    window.platforms_button.click()
    assert starts == [("", True)]
    settle(window, app)


@needs_ffmpeg
def test_a_search_that_cannot_tell_says_so(app, tmp_path, monkeypatch):
    path = tmp_path / "plain.mp4"
    make_clip(path)
    window = MainWindow()
    open_clip(window, path, app)
    monkeypatch.setattr(main_window_module, "find_bars", lambda *a, **k: None)
    seen = {}

    def fake_exec(dialog):
        seen["failed"] = wait_until(app, lambda: dialog.status.text() == SEARCH_FAILED)
        return 0

    monkeypatch.setattr(PlatformDialog, "exec", fake_exec)
    window.platforms_button.click()
    assert seen["failed"] and window.bars_found[str(path)] is None
    settle(window, app)


@needs_ffmpeg
def test_closing_the_window_stops_the_search(app, tmp_path, monkeypatch):
    path = tmp_path / "plain.mp4"
    make_clip(path)
    window = MainWindow()
    open_clip(window, path, app)
    started = []

    def slow(path_, count, should_stop=None, **kwargs):
        started.append(1)
        while not should_stop():
            time.sleep(0.01)
        return None

    monkeypatch.setattr(main_window_module, "find_bars", slow)
    monkeypatch.setattr(PlatformDialog, "exec", lambda dialog: 0)     # closed at once
    window.platforms_button.click()                                    # returns only if stopped
    assert started == [1]
    assert not any(t.isRunning() for t in window.findChildren(main_window_module.BarsThread))
    assert str(path) not in window.bars_found                          # nothing was learned
    settle(window, app)