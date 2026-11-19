"""Tests for the arranged main window: the cards, the arrow buttons, the dark
theme, dropping a video on the window, and the pieces they are made of."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import time

import pytest
from PySide6.QtCore import QMimeData, QPoint, QSize, QUrl, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from blackboard.probe import probe
from blackboard.ui import theme
from blackboard.ui.main_window import (
    LABEL_UNITS, ONE_VIEW_HEIGHT, SCOPE_NAMES, SHOT_TABLE_HEIGHT, UNIT_FRAMES, UNIT_TIME,
    EMPTY_PREVIEW_TEXT, LEFT_W, WINDOW_H, WINDOW_W, MainWindow,
)
from blackboard.ui.segment_picker import SegmentPicker
from blackboard.ui.shot_list import LIST_WIDTH, SUMMARY_WIDTH, ShotListView
from blackboard.ui.widgets import (
    POINT, Card, ChevronButton, ClapperProgressBar, ElidedLabel, PreviewLabel, StepButton,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


KEEP = []        # windows stay alive during the file, so none is collected mid-test


@pytest.fixture(scope="module", autouse=True)
def release_windows(app):
    """After the last test of this file, close and delete every window made here, while
    Qt is still running. Leaving them for Python's exit crashes Windows at shutdown."""
    yield
    for window in KEEP:
        try:
            window.close()                 # waits for any background thread
            window.deleteLater()
        except RuntimeError:               # already deleted
            pass
    KEEP.clear()
    for _ in range(5):
        app.processEvents()
    import gc
    gc.collect()


def shown(app):
    window = MainWindow()
    KEEP.append(window)
    window.show()
    app.processEvents()
    return window


def where(widget, window):
    return widget.mapTo(window, QPoint(0, 0))


def settle(window, app, seconds=30):
    """Wait until the video is read and every background thread is done."""
    deadline = time.time() + seconds
    while window.info is None and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    while time.time() < deadline:
        threads = [window.probe_thread, window.barcode_thread, window.shots_thread,
                   window.motion_thread, window.frame_thread]
        if all(t is None or t.isFinished() for t in threads):
            break
        app.processEvents()
        time.sleep(0.02)
    for _ in range(10):
        app.processEvents()
        time.sleep(0.01)
    window.close()                  # waits for any thread still running


def make_clip(path, size="320x180", seconds=1):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size={size}:rate=30:duration={seconds}", "-pix_fmt", "yuv420p",
                    str(path)], check=True)


# ---------- where things are, following the sketch ----------

def test_the_clapper_bar_runs_from_the_open_button_to_the_window_edge(app):
    window = shown(app)
    open_, bar = window.open_button, window.progress_bar
    assert abs((where(open_, window).y() + open_.height() // 2) - (where(bar, window).y() + bar.height() // 2)) <= 3
    assert where(bar, window).x() - (where(open_, window).x() + open_.width()) <= 16   # right after Open
    assert window.width() - (where(bar, window).x() + bar.width()) <= 14               # to the window's edge
    assert where(bar, window).y() + bar.height() < where(window.details_card, window).y()
    assert isinstance(bar, ClapperProgressBar) and bar.isVisible()


def test_the_open_button_is_as_tall_as_the_loading_bar(app):
    window = shown(app)
    assert window.open_button.height() == window.progress_bar.height()
    assert window.open_button.sizeHint().height() == window.open_button.height()


def test_the_video_path_is_just_below_the_loading_bar(app):
    window = shown(app)
    bar, label = window.progress_bar, window.file_label
    gap = where(label, window).y() - (where(bar, window).y() + bar.height())
    assert 0 <= gap <= 6
    assert where(label, window).x() == where(bar, window).x()
    assert label.width() == bar.width()
    assert where(label, window).y() + label.height() <= where(window.details_card, window).y()
    assert not window.details_card.isAncestorOf(label)
    window.file_label.setText("C:/Videos/clip.mp4")
    assert window.file_label.full_text == "C:/Videos/clip.mp4"


def test_the_other_buttons_sit_with_their_feature(app):
    window = shown(app)
    top = where(window.open_button, window).y()
    homes = {window.platforms_button: window.details_card,
             window.shots_button: window.shots_card,
             window.analyze_button: window.strips_card,
             window.motion_button: window.strips_card}
    for button, card in homes.items():
        assert isinstance(button, StepButton) and not isinstance(button, ChevronButton)
        assert card.isAncestorOf(button)
        assert where(button, window).y() > top + 20       # none is on the top row


def test_each_analysis_button_is_beside_its_own_strip(app):
    window = shown(app)
    color = where(window.analyze_button, window).y() + window.analyze_button.height() // 2
    motion = where(window.motion_button, window).y() + window.motion_button.height() // 2
    strip = lambda view: where(view, window).y() + view.height() // 2
    assert abs(color - strip(window.barcode_view)) <= 20
    assert abs(motion - strip(window.motion_view)) <= 20
    assert where(window.analyze_button, window).x() > where(window.barcode_view, window).x()
    assert where(window.motion_button, window).x() > where(window.motion_view, window).x()


def test_the_chart_fills_the_rest_of_the_left_column(app):
    window = shown(app)
    bottom = lambda w: where(w, window).y() + w.height()
    assert bottom(window.views_card) >= window.height() - 12 - 4
    assert bottom(window.analysis_card) < where(window.views_card, window).y()
    assert window.views_card.isAncestorOf(window.histogram_view)
    assert window.histogram_view.height() > 3 * ONE_VIEW_HEIGHT


def test_the_left_column_is_metadata_then_frame_analysis_then_views(app):
    window = shown(app)
    cards = [window.details_card, window.analysis_card, window.views_card]
    ys = [where(c, window).y() for c in cards]
    assert ys == sorted(ys) and len(set(ys)) == 3
    assert len({where(c, window).x() for c in cards}) == 1
    assert all(c.width() == LEFT_W for c in cards)


def test_the_right_column_is_preview_then_segment_analysis_then_shots(app):
    window = shown(app)
    cards = [window.preview_card, window.strips_card, window.shots_card]
    ys = [where(c, window).y() for c in cards]
    assert ys == sorted(ys) and len(set(ys)) == 3
    assert len({where(c, window).x() for c in cards}) == 1
    assert where(cards[0], window).x() >= where(window.details_card, window).x() + LEFT_W
    assert where(cards[2], window).y() + cards[2].height() <= window.height()


def test_no_card_is_squeezed_into_the_next_one(app):
    window = shown(app)
    for column in ([window.details_card, window.analysis_card, window.views_card],
                   [window.preview_card, window.strips_card, window.shots_card]):
        for upper, lower in zip(column, column[1:]):
            assert upper.geometry().bottom() < lower.geometry().top()
            assert upper.height() >= upper.minimumSizeHint().height()
    inner = window.preview_card
    assert where(window.frame_slider, window).y() > where(window.preview, window).y() + window.preview.height() - 2
    assert inner.isAncestorOf(window.frame_info)


def test_the_pieces_are_in_their_cards(app):
    window = shown(app)
    inside = lambda widget, card: card.isAncestorOf(widget)
    assert inside(window.preview, window.preview_card)
    assert inside(window.frame_slider, window.preview_card)
    assert inside(window.segment_picker, window.preview_card)
    assert inside(window.barcode_view, window.strips_card)
    assert inside(window.brightness_view, window.strips_card)
    assert inside(window.motion_view, window.strips_card)
    assert inside(window.motion_status, window.strips_card)
    assert inside(window.scope_stack, window.views_card)
    assert inside(window.scope_selector, window.views_card)
    assert inside(window.shot_list, window.shots_card)
    assert inside(window.scope_stack, window.views_card)
    assert inside(window.exposure_check, window.analysis_card)
    assert inside(window.fields["Codec"], window.details_card)


def test_video_facts_are_in_one_column_and_audio_facts_in_the_other(app):
    window = shown(app)
    video = [where(window.fields[n], window).x() for n in ("Codec", "Resolution", "Duration")]
    audio = [where(window.fields[n], window).x() for n in ("Audio", "Audio codec", "Channels")]
    assert len(set(video)) == 1 and len(set(audio)) == 1
    assert video[0] < audio[0]
    top = lambda name: where(window.fields[name], window).y()
    assert top("Codec") == top("Audio")                 # both columns start together
    assert top("Codec") < top("Profile") < top("Resolution")
    assert top("Audio") < top("Audio tracks") < top("Audio codec")


def test_the_card_titles(app):
    window = shown(app)
    assert window.details_card.title_label.text() == "Metadata"
    assert window.analysis_title.text() == "Frame analysis"
    assert window.shots_title.text() == "Shot list"
    assert window.strips_card.title_label.text() == "Segment analysis"
    assert window.views_card.title_label.text() == "Views"
    assert window.shots_title.toolTip()


def test_the_three_strips_are_named_and_line_up(app):
    window = shown(app)
    rows = {where(v, window).y(): v for v in (window.barcode_view, window.brightness_view,
                                              window.motion_view)}
    assert len(rows) == 3
    assert window.barcode_view.width() == window.brightness_view.width() == window.motion_view.width()
    names = [l.text() for l in window.strips_card.findChildren(QLabel)]
    for name in ("Color", "Brightness", "Motion"):
        assert name in names
    assert where(window.motion_status, window).y() > where(window.motion_view, window).y()


def test_the_window_fits_when_it_opens(app):
    window = shown(app)
    assert window.width() >= window.minimumSizeHint().width()
    assert window.minimumSizeHint().height() <= WINDOW_H
    assert window.minimumSizeHint().width() <= 1500          # fits a 1536-wide screen
    assert window.width() >= max(WINDOW_W, window.minimumSizeHint().width())   # opens big enough
    bottom = where(window.views_card, window).y() + window.views_card.height()
    assert bottom <= window.height()
    assert where(window.shots_card, window).y() + window.shots_card.height() <= window.height()


def test_the_window_accepts_drops_and_starts_empty(app):
    window = shown(app)
    assert window.acceptDrops()
    assert window.preview.text() == EMPTY_PREVIEW_TEXT
    assert "drop" in EMPTY_PREVIEW_TEXT
    assert window.file_label.full_text == "No video selected"


def test_a_running_step_shows_as_busy(app):
    window = shown(app)
    window.info = object()
    window.motion_running = True
    window.refresh_controls()
    assert window.motion_button.busy and not window.analyze_button.busy
    window.motion_running = False
    window.barcode_running = True
    window.refresh_controls()
    assert window.analyze_button.busy and not window.motion_button.busy
    window.barcode_running = False
    window.shots_running = True
    window.refresh_controls()
    assert window.shots_button.busy
    window.shots_running = False
    window.info = None
    window.refresh_controls()
    assert not any(b.busy for b in (window.analyze_button, window.shots_button, window.motion_button))


def test_the_shot_list_sits_under_the_segment_analysis_with_its_numbers_beside_it(app):
    window = shown(app)
    table, summary = window.shot_list.table, window.shot_list.summary
    assert table.height() == SHOT_TABLE_HEIGHT
    assert table.width() >= window.shots_card.width() - SUMMARY_WIDTH - 60
    assert where(summary, window).x() >= where(table, window).x() + table.width()
    assert abs(where(summary, window).y() - where(table, window).y()) <= 4          # level with the table's top
    assert summary.width() == SUMMARY_WIDTH


def test_the_shot_list_has_three_shapes(app):
    assert ShotListView().table.width() == LIST_WIDTH                  # the default is unchanged
    assert ShotListView().table.minimumHeight() == ShotListView().table.maximumHeight()
    assert ShotListView(grow=True).table.maximumHeight() > 190
    assert ShotListView(width=300).table.width() == 300
    assert ShotListView(width=300).summary.width() == 300
    beside = ShotListView(width=None, beside=True, height=111)
    assert beside.table.height() == 111 and beside.summary.width() == SUMMARY_WIDTH


# ---------- dropping a video on the window ----------

def mime(*paths):
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(p) for p in paths])
    return data


def test_the_first_video_in_a_drop_is_chosen():
    assert MainWindow.dropped_video(mime("/a/notes.txt", "/a/Clip.MP4", "/a/b.mov")) == "/a/Clip.MP4"
    for name in ("a.mp4", "a.mov", "a.m4v", "a.mkv", "a.avi"):
        assert MainWindow.dropped_video(mime("/x/" + name)) == "/x/" + name
    assert MainWindow.dropped_video(mime("/a/notes.txt", "/a/pic.png")) is None
    assert MainWindow.dropped_video(QMimeData()) is None
    text = QMimeData()
    text.setText("hello.mp4")
    assert MainWindow.dropped_video(text) is None


class FakeEvent:
    def __init__(self, data):
        self.data = data
        self.accepted = False

    def mimeData(self):
        return self.data

    def acceptProposedAction(self):
        self.accepted = True


def test_dragging_a_video_over_the_window_is_accepted(app):
    window = shown(app)
    video, other = FakeEvent(mime("/a/clip.mp4")), FakeEvent(mime("/a/notes.txt"))
    window.dragEnterEvent(video)
    window.dragEnterEvent(other)
    assert video.accepted and not other.accepted


def test_dropping_a_video_opens_it(app, monkeypatch):
    window = shown(app)
    opened = []
    monkeypatch.setattr(window, "open_path", opened.append)
    event = FakeEvent(mime("/a/notes.txt", "/a/clip.mp4"))
    window.dropEvent(event)
    expected = os.path.normpath("/a/clip.mp4")        # backslashes on Windows
    assert opened == [expected] and event.accepted
    window.dropEvent(FakeEvent(mime("/a/notes.txt")))
    assert opened == [expected]


def test_drops_are_ignored_while_a_video_is_being_read(app, monkeypatch):
    window = shown(app)
    window.open_button.setEnabled(False)
    opened = []
    monkeypatch.setattr(window, "open_path", opened.append)
    entering, dropping = FakeEvent(mime("/a/clip.mp4")), FakeEvent(mime("/a/clip.mp4"))
    window.dragEnterEvent(entering)
    window.dropEvent(dropping)
    assert opened == [] and not entering.accepted and not dropping.accepted


@needs_ffmpeg
def test_a_dropped_file_is_read_and_shown(app, tmp_path):
    path = tmp_path / "dropped.mp4"
    make_clip(path)
    window = shown(app)
    window.dropEvent(FakeEvent(mime(str(path))))
    settle(window, app)
    assert window.info is not None and window.current_path == str(path)
    assert window.file_label.full_text == str(path)
    assert window.fields["Resolution"].text() == "320\u00d7180" or "320" in window.fields["Resolution"].text()
    assert window.analyze_button.isEnabled() and window.platforms_button.isEnabled()
    assert window.open_button.isEnabled()


@needs_ffmpeg
def test_the_file_dialog_and_dropping_share_the_same_path(app, tmp_path, monkeypatch):
    path = tmp_path / "chosen.mp4"
    make_clip(path)
    import blackboard.ui.main_window as module
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    window = shown(app)
    window.open_button.click()
    settle(window, app)
    assert window.current_path == str(path) and window.info is not None


def test_cancelling_the_file_dialog_does_nothing(app, monkeypatch):
    import blackboard.ui.main_window as module
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    window = shown(app)
    window.open_button.click()
    assert window.info is None and window.file_label.full_text == "No video selected"


# ---------- the segment boxes with frames and time side by side ----------

def test_both_kinds_of_box_are_shown_without_the_drop_down(app):
    picker = SegmentPicker(show_both=True)
    picker.show()
    picker.set_video(300, 30.0)
    app.processEvents()
    for box in (picker.start_spin, picker.end_spin, picker.start_time, picker.end_time):
        assert box.isVisible() and box.isEnabled()
    assert not picker.unit_combo.isVisible()


def test_the_default_picker_still_has_the_drop_down(app):
    picker = SegmentPicker()
    picker.show()
    picker.set_video(300, 30.0)
    app.processEvents()
    assert picker.unit_combo.isVisible()
    assert picker.start_spin.isVisible() and not picker.start_time.isVisible()


def test_the_boxes_follow_each_other(app):
    picker = SegmentPicker(show_both=True)
    picker.set_video(300, 30.0)
    picker.start_spin.setValue(60)
    assert picker.start_time.value() == pytest.approx(2.0)
    picker.end_time.setValue(5.0)
    assert picker.end_spin.value() == 150
    assert picker.segment() == (60, 150)


def test_without_a_frame_rate_the_time_boxes_are_off(app):
    picker = SegmentPicker(show_both=True)
    picker.set_video(300, 0.0)
    assert picker.start_spin.isEnabled() and not picker.start_time.isEnabled()
    assert not picker.end_time.isEnabled()
    picker.set_video(300, 30.0)
    assert picker.start_time.isEnabled() and picker.end_time.isEnabled()


def test_the_window_uses_both_kinds_of_box(app):
    window = shown(app)
    assert window.segment_picker.show_both


# ---------- the pieces ----------

def test_a_card_has_a_title_and_room_for_more_in_its_header(app):
    card = Card("Things")
    assert card.title_label.text() == "Things" and not card.title_label.isHidden()
    extra = QLabel("extra")
    assert card.add_to_header(extra) is extra
    card.body.addWidget(QLabel("inside"))
    card.show()
    app.processEvents()
    assert extra.y() < card.body.itemAt(0).widget().y()
    assert Card().title_label.isHidden()


def test_buttons_inside_cards_are_shorter_than_a_full_size_arrow(app):
    window = shown(app)
    assert window.analyze_button.height() == window.open_button.height()
    assert ChevronButton("Go", compact=True).height() < ChevronButton("Go").height()
    assert ChevronButton("Go", compact=True).sizeHint().height() == ChevronButton("Go", compact=True).height()


def test_an_arrow_button_is_wider_for_longer_text(app):
    short, long = ChevronButton("Go"), ChevronButton("Analyze color barcode")
    assert long.sizeHint().width() > short.sizeHint().width()
    assert ChevronButton("Go", first=True).sizeHint().width() < short.sizeHint().width()
    assert short.sizeHint().height() == short.height()


def test_an_arrow_button_text_fits_in_its_arrow(app):
    from PySide6.QtGui import QFontMetrics
    button = ChevronButton("Analyze color barcode")
    needed = QFontMetrics(button.font()).horizontalAdvance(button.text())
    assert button.sizeHint().width() >= needed + 2 * POINT


def test_clicking_the_middle_of_an_arrow_button_works(app):
    button = ChevronButton("Go")
    button.resize(button.sizeHint())
    button.show()
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=button.rect().center())
    assert clicks == [1]


def test_the_corners_of_an_arrow_button_do_not_click(app):
    button = ChevronButton("Go")
    button.resize(button.sizeHint())
    assert button.hitButton(button.rect().center())
    assert not button.hitButton(QPoint(button.width() - 2, 1))            # outside the tip
    assert not button.hitButton(QPoint(1, button.height() // 2))          # inside the notch
    first = ChevronButton("Go", first=True)
    first.resize(first.sizeHint())
    assert first.hitButton(QPoint(1, first.height() // 2))                # the first one is flat


def test_a_disabled_arrow_button_does_not_click(app):
    button = ChevronButton("Go")
    button.resize(button.sizeHint())
    button.show()
    button.setEnabled(False)
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=button.rect().center())
    assert clicks == []


def test_arrow_button_colors_tell_the_state(app):
    plain, primary = ChevronButton("Go"), ChevronButton("Open", primary=True)
    assert plain.fill_color() != primary.fill_color()
    assert primary.fill_color() == QColor(theme.ACCENT)
    plain.set_busy(True)
    assert plain.fill_color() == QColor(theme.BUSY)
    plain.set_busy(False)
    assert plain.fill_color() == QColor(theme.BUTTON)
    plain.setEnabled(False)
    assert plain.fill_color() == QColor(theme.CARD)


def test_an_arrow_button_can_be_set_busy_twice(app):
    button = ChevronButton("Go")
    button.set_busy(True)
    button.set_busy(True)
    assert button.busy
    button.set_busy(False)
    assert not button.busy


def test_an_arrow_button_keeps_its_tooltip_and_text(app):
    button = ChevronButton("Analyze motion")
    button.setToolTip("hello")
    button.setText("Cancel")
    assert button.text() == "Cancel" and button.toolTip() == "hello"


def test_the_arrow_shape_has_a_notch_unless_it_is_first(app):
    button, first = ChevronButton("Go"), ChevronButton("Go", first=True)
    for b in (button, first):
        b.resize(120, 36)
    assert button.shape().count() == 6 and first.shape().count() == 5


# ---------- long file names ----------

def test_a_long_path_is_shortened_in_the_middle(app):
    label = ElidedLabel()
    label.resize(200, 20)
    path = "C:/Users/josep/Videos/a/very/long/folder/name/Makers Fair Reel Final.mp4"
    label.setText(path)
    assert label.full_text == path and label.toolTip() == path
    assert label.text() != path and "\u2026" in label.text()
    assert label.text().startswith("C:") and label.text().endswith("mp4")


def test_a_short_path_is_not_changed(app):
    label = ElidedLabel()
    label.resize(400, 20)
    label.setText("clip.mp4")
    assert label.text() == "clip.mp4"


def test_a_label_gets_shorter_text_as_it_gets_narrower(app):
    label = ElidedLabel()
    label.show()
    label.resize(500, 20)
    label.setText("C:/Users/josep/Videos/Makers Fair Reel Final.mp4")
    app.processEvents()
    wide = label.text()
    label.resize(150, 20)
    app.processEvents()
    assert len(label.text()) < len(wide)
    label.resize(500, 20)
    app.processEvents()
    assert label.text() == wide


def test_a_long_path_does_not_make_the_window_wider(app):
    window = shown(app)
    before = window.minimumSizeHint().width()
    window.file_label.setText("C:/" + "very-long-folder-name/" * 30 + "clip.mp4")
    app.processEvents()
    assert window.minimumSizeHint().width() == before


# ---------- the dark theme ----------

def test_the_theme_sets_a_dark_look_and_can_be_undone(app):
    old_style, old_palette, old_sheet = app.style().objectName(), app.palette(), app.styleSheet()
    try:
        theme.apply_theme(app)
        assert app.palette().color(app.palette().ColorRole.Window) == QColor(theme.WINDOW)
        assert app.palette().color(app.palette().ColorRole.Highlight) == QColor(theme.ACCENT)
        assert "QPushButton" in app.styleSheet() and "QToolTip" in app.styleSheet()
    finally:
        if old_style:
            app.setStyle(old_style)
        app.setPalette(old_palette)
        app.setStyleSheet(old_sheet)


def test_the_theme_is_nearly_black():
    for color in (theme.WINDOW, theme.CARD, theme.FIELD, theme.BUTTON):
        assert QColor(color).lightness() < 40
    assert QColor(theme.WINDOW).lightness() < QColor(theme.CARD).lightness()


def test_the_palette_is_dark_with_readable_text():
    from PySide6.QtGui import QPalette
    palette = theme.make_palette()
    window = palette.color(QPalette.ColorRole.Window)
    text = palette.color(QPalette.ColorRole.WindowText)
    assert window.lightness() < 60 and text.lightness() > 190
    disabled = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText)
    assert disabled != palette.color(QPalette.ColorGroup.Active, QPalette.ColorRole.ButtonText)
    assert QColor(theme.MUTED).lightness() < text.lightness()


# ---------- the clapper board progress bar ----------

def test_the_bar_is_empty_until_something_runs(app):
    bar = ClapperProgressBar()
    assert bar.value() == 0 and bar.fraction() == 0.0 and not bar.is_busy()
    bar.setValue(40)
    assert bar.fraction() == pytest.approx(0.4)
    bar.setValue(100)
    assert bar.fraction() == 1.0
    assert not bar.timer.isActive()


def test_an_unknown_length_makes_the_stripes_slide(app):
    bar = ClapperProgressBar()
    bar.show()
    bar.setRange(0, 0)
    assert bar.is_busy() and bar.fraction() == 1.0 and bar.timer.isActive()
    before = bar.offset
    bar.slide()
    assert bar.offset != before
    bar.setRange(0, 100)
    assert not bar.is_busy() and not bar.timer.isActive()
    bar.hide()


def test_a_hidden_bar_does_not_keep_a_timer_running(app):
    bar = ClapperProgressBar()
    bar.setRange(0, 0)
    bar.show()
    assert bar.timer.isActive()
    bar.hide()
    assert not bar.timer.isActive()
    bar.show()
    assert bar.timer.isActive()
    bar.setRange(0, 100)
    bar.hide()


def test_the_stripes_wrap_around(app):
    bar = ClapperProgressBar()
    for _ in range(100):
        bar.slide()
    assert 0 <= bar.offset < 2 * 14


def paint_columns(bar):
    from PySide6.QtGui import QImage
    bar.resize(300, bar.height())
    image = QImage(bar.size(), QImage.Format.Format_RGB32)
    image.fill(QColor("#ff00ff"))
    bar.render(image)
    return image


def test_the_finished_part_is_black_and_white_and_the_rest_is_dark(app):
    bar = ClapperProgressBar()
    bar.setValue(50)
    image = paint_columns(bar)
    y = bar.height() // 2
    done = {image.pixelColor(x, y).lightness() for x in range(6, 140)}
    rest = {image.pixelColor(x, y).lightness() for x in range(165, 290)}
    assert max(done) > 200 and min(done) < 20          # white stripes and black stripes
    assert max(rest) < 60                              # dark, nothing white
    assert image.pixelColor(150, y).lightness() < 60 or image.pixelColor(150, y).lightness() > 200


def test_an_idle_bar_has_no_white(app):
    bar = ClapperProgressBar()
    image = paint_columns(bar)
    y = bar.height() // 2
    assert max(image.pixelColor(x, y).lightness() for x in range(6, 290)) < 60


def test_the_window_uses_the_bar_and_resets_it_after_reading(app, tmp_path):
    window = shown(app)
    window.on_progress(-1)
    assert window.progress_bar.is_busy()
    window.on_progress(30)
    assert window.progress_bar.value() == 30 and not window.progress_bar.is_busy()
    window.end_probe()
    assert window.progress_bar.value() == 0 and window.progress_bar.isVisible()
    assert window.open_button.isEnabled()


# ---------- all the views at once ----------

def views_shown(window):
    return [v for v in window.chart_views if v.isVisible()]


def pick(window, app, position):
    window.scope_selector.setCurrentIndex(position)
    app.processEvents()


def test_one_chart_is_shown_at_a_time_and_it_grows_with_the_card(app):
    window = shown(app)
    assert window.scope_selector.currentIndex() == 0
    assert views_shown(window) == [window.histogram_view]
    assert window.histogram_view.height() >= ONE_VIEW_HEIGHT
    for position in range(len(SCOPE_NAMES)):
        pick(window, app, position)
        assert views_shown(window) == [window.chart_views[position]]
        assert window.chart_views[position].height() == window.scope_stack.height()
    assert len(SCOPE_NAMES) == 4 and "All views" not in SCOPE_NAMES


def test_the_window_still_fits_with_any_choice(app):
    window = shown(app)
    for position in range(len(SCOPE_NAMES)):
        pick(window, app, position)
        assert window.minimumSizeHint().height() <= WINDOW_H


def test_the_drop_down_describes_each_choice(app):
    window = shown(app)
    assert len(window.scope_tips) == len(SCOPE_NAMES)
    for position in range(len(SCOPE_NAMES)):
        window.scope_selector.setCurrentIndex(position)
        assert window.scope_selector.toolTip() == window.scope_tips[position]


# ---------- plain buttons and the gray highlight ----------

def test_a_step_button_looks_busy_only_while_its_step_runs(app):
    button = StepButton("Analyze motion")
    assert not button.busy and button.property("busy") is False
    button.set_busy(True)
    button.set_busy(True)
    assert button.busy and button.property("busy") is True
    button.set_busy(False)
    assert not button.busy and button.property("busy") is False


def test_a_step_button_is_a_normal_rectangle_and_short(app):
    button = StepButton("Detect shots")
    assert button.height() < ChevronButton("Open", first=True).height()
    assert not hasattr(button, "shape")            # no arrow outline, no notch
    assert button.minimumWidth() >= 100


def test_a_step_button_clicks_anywhere_and_not_when_disabled(app):
    button = StepButton("Go")
    button.resize(120, button.height())
    button.show()
    clicks = []
    button.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=QPoint(button.width() - 3, 3))
    assert clicks == [1, 1]
    button.setEnabled(False)
    QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=button.rect().center())
    assert clicks == [1, 1]


def test_the_busy_style_is_in_the_stylesheet():
    assert 'QPushButton[busy="true"]' in theme.STYLE_SHEET and theme.BUSY in theme.STYLE_SHEET


def is_gray(color):
    c = QColor(color)
    return abs(c.red() - c.green()) < 12 and abs(c.green() - c.blue()) < 12


def test_highlights_are_a_light_gray_not_blue(app):
    assert is_gray(theme.ACCENT) and QColor(theme.ACCENT).lightness() > 150
    assert QColor(theme.ON_ACCENT).lightness() < 40            # dark text on it
    from blackboard.ui.segment_picker import BAND_SEGMENT
    assert is_gray(BAND_SEGMENT.name()) and BAND_SEGMENT.lightness() > 120
    palette = theme.make_palette()
    from PySide6.QtGui import QPalette
    assert palette.color(QPalette.ColorRole.Highlight) == QColor(theme.ACCENT)
    assert palette.color(QPalette.ColorRole.HighlightedText) == QColor(theme.ON_ACCENT)
    assert "#3d8bfd" not in theme.STYLE_SHEET


def test_the_open_button_is_light_gray_with_dark_text(app):
    window = shown(app)
    assert window.open_button.fill_color() == QColor(theme.ACCENT)
    assert window.open_button.primary


def test_a_selected_shot_is_gray_with_dark_text(app):
    from PySide6.QtGui import QPalette
    old_palette, old_sheet = app.palette(), app.styleSheet()
    try:
        theme.apply_theme(app)
        window = shown(app)
        palette = window.shot_list.table.palette()
        highlight = palette.color(QPalette.ColorRole.Highlight)
        assert is_gray(highlight.name()) and highlight.lightness() > 150
        assert palette.color(QPalette.ColorRole.HighlightedText).lightness() < 40
    finally:
        app.setPalette(old_palette)
        app.setStyleSheet(old_sheet)


# ---------- the picture grows with its card ----------

def pixmap_of(width, height, color="#ff0000"):
    from PySide6.QtGui import QPixmap
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(color))
    return pixmap


def test_the_preview_keeps_the_shape_of_the_picture(app):
    label = PreviewLabel()
    label.resize(600, 400)
    label.setPixmap(pixmap_of(1600, 900))
    shown_size = label.pixmap().size()
    assert shown_size.width() == 600 and abs(shown_size.height() - 338) <= 1
    label.setPixmap(pixmap_of(608, 1080))                          # a tall video
    shown_size = label.pixmap().size()
    assert shown_size.height() == 400 and abs(shown_size.width() - 225) <= 1


def test_the_preview_follows_the_size_of_the_label(app):
    label = PreviewLabel()
    label.show()
    label.resize(500, 300)
    label.setPixmap(pixmap_of(1280, 720))
    small = label.pixmap().width()
    label.resize(900, 600)
    app.processEvents()
    assert label.pixmap().width() > small
    assert label.pixmap().width() == 900
    label.resize(400, 250)
    app.processEvents()
    assert label.pixmap().width() <= 400 and label.pixmap().height() <= 250


def test_a_big_picture_does_not_push_the_label_bigger(app):
    label = PreviewLabel()
    label.setPixmap(pixmap_of(4000, 3000))
    assert label.sizeHint() == QSize(PreviewLabel.WANT_W, PreviewLabel.WANT_H)
    assert label.minimumSizeHint() == QSize(PreviewLabel.MIN_W, PreviewLabel.MIN_H)
    label.resize(500, 400)
    app.processEvents()
    assert label.size() == QSize(500, 400)


def test_text_and_clear_remove_the_picture(app):
    label = PreviewLabel()
    label.resize(400, 300)
    label.setPixmap(pixmap_of(100, 100))
    assert label.source is not None
    label.setText("Loading frame...")
    assert label.source is None and label.text() == "Loading frame..."
    label.setPixmap(pixmap_of(100, 100))
    label.clear()
    assert label.source is None and label.pixmap().isNull()
    label.resize(500, 350)                    # resizing with nothing to show is harmless
    assert label.pixmap().isNull()


def test_a_null_picture_is_ignored(app):
    from PySide6.QtGui import QPixmap
    label = PreviewLabel()
    label.resize(300, 200)
    label.setPixmap(QPixmap())
    label.resize(310, 210)


def test_the_picture_is_bigger_in_a_bigger_window(app):
    import numpy as np
    from blackboard.ui.main_window import frame_to_qimage
    from PySide6.QtGui import QPixmap
    window = shown(app)
    window.normal_picture = QPixmap.fromImage(frame_to_qimage(np.full((720, 1280, 3), 90, np.uint8)))
    window.warning_picture = None
    window.show_current_picture()
    app.processEvents()
    before = window.preview.pixmap().size()
    window.resize(window.width() + 400, window.height() + 160)
    app.processEvents()
    after = window.preview.pixmap().size()
    assert after.width() > before.width() and after.height() > before.height()
    assert abs(after.width() / after.height() - 16 / 9) < 0.02


def test_the_preview_picture_is_made_at_the_larger_size(app):
    import numpy as np
    from blackboard.ui.main_window import PREVIEW_H, PREVIEW_W, shrink_for_preview
    assert PREVIEW_W >= 1280 and PREVIEW_H >= 720
    assert shrink_for_preview(np.zeros((1000, 1500, 3), np.uint8)).shape == (720, 1080, 3)
    assert shrink_for_preview(np.zeros((360, 640, 3), np.uint8)).shape == (360, 640, 3)    # never enlarged


def test_the_preview_card_takes_the_spare_height(app):
    window = shown(app)
    small = window.preview.height()
    window.resize(window.width(), window.height() + 150)
    app.processEvents()
    assert window.preview.height() >= small + 120
    assert window.shots_card.height() <= window.shots_card.minimumSizeHint().height() + 4
    assert window.strips_card.height() <= window.strips_card.minimumSizeHint().height() + 4


# ---------- naming the position in frames or in time ----------

def test_the_position_is_named_in_frames_at_first(app):
    window = shown(app)
    assert [window.label_unit.itemText(i) for i in range(window.label_unit.count())] == LABEL_UNITS
    assert window.label_unit.currentIndex() == UNIT_FRAMES
    assert window.preview_card.isAncestorOf(window.label_unit)
    assert not window.label_unit.isEnabled()                    # no video yet
    assert where(window.label_unit, window).y() >= where(window.frame_info, window).y() - 4


def fake_video(window, frames=360, fps=30.0):
    from types import SimpleNamespace
    window.info = SimpleNamespace(fps=fps)
    window.frame_slider.blockSignals(True)
    window.frame_slider.setRange(0, frames - 1)
    window.frame_slider.blockSignals(False)


def test_time_mode_names_the_position_as_a_time(app):
    window = shown(app)
    fake_video(window)
    window.frame_slider.blockSignals(True)
    window.frame_slider.setValue(150)
    window.frame_slider.blockSignals(False)
    window.update_frame_label(150)
    assert window.frame_info.text() == "Frame 150 of 359   |   5.00 s"
    window.label_unit.setCurrentIndex(UNIT_TIME)
    assert window.frame_info.text() == "0:05.00 of 0:11.97   |   frame 150"
    window.update_frame_label(0)
    assert window.frame_info.text() == "0:00.00 of 0:11.97   |   frame 0"
    window.label_unit.setCurrentIndex(UNIT_FRAMES)       # the slider is still at frame 150
    assert window.frame_info.text() == "Frame 150 of 359   |   5.00 s"


def test_the_label_changes_as_soon_as_the_toggle_is_flipped(app):
    window = shown(app)
    fake_video(window)
    window.frame_slider.blockSignals(True)
    window.frame_slider.setValue(90)
    window.frame_slider.blockSignals(False)
    window.update_frame_label(90)
    window.label_unit.setCurrentIndex(UNIT_TIME)
    assert "0:03.00" in window.frame_info.text()
    window.label_unit.setCurrentIndex(UNIT_FRAMES)
    assert "Frame 90 of 359" in window.frame_info.text()


def test_the_toggle_does_nothing_without_a_video(app):
    window = shown(app)
    window.label_unit.setEnabled(True)
    window.label_unit.setCurrentIndex(UNIT_TIME)          # nothing to label, and nothing breaks
    assert window.frame_info.text() == ""


def test_long_videos_use_hours_in_time_mode(app):
    window = shown(app)
    fake_video(window, frames=108000, fps=30.0)            # one hour
    window.label_unit.setCurrentIndex(UNIT_TIME)
    window.update_frame_label(54000)
    assert window.frame_info.text().startswith("30:00.00 of 59:59.97")
    window.update_frame_label(107999)
    assert "of 59:59.97" in window.frame_info.text()
    fake_video(window, frames=216000, fps=30.0)
    window.update_frame_label(108000)
    assert window.frame_info.text().startswith("1:00:00.00 of 1:59:59.97")


@needs_ffmpeg
def test_a_real_video_enables_the_toggle_and_the_choice_stays_for_the_next_one(app, tmp_path):
    first, second = tmp_path / "a.mp4", tmp_path / "b.mp4"
    make_clip(first, seconds=2)
    make_clip(second, seconds=1)
    window = shown(app)
    window.open_path(str(first))
    settle(window, app)
    assert window.label_unit.isEnabled()
    window.label_unit.setCurrentIndex(UNIT_TIME)
    assert "of 0:01.97" in window.frame_info.text() or "of 0:02" in window.frame_info.text()
    window.open_path(str(second))
    settle(window, app)
    assert window.label_unit.currentIndex() == UNIT_TIME
    assert "of 0:00.97" in window.frame_info.text()
    window.clear_preview()
    assert not window.label_unit.isEnabled()


# ---------- check boxes and small text ----------

def test_small_text_is_lighter_than_before_and_still_below_the_main_text():
    assert QColor(theme.MUTED).lightness() >= 165
    assert QColor(theme.MUTED).lightness() < QColor(theme.TEXT).lightness()


def contrast(a, b):
    def lum(color):
        c = QColor(color)
        parts = []
        for v in (c.redF(), c.greenF(), c.blueF()):
            parts.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_small_text_is_easy_to_read_on_the_cards():
    assert contrast(theme.MUTED, theme.CARD) >= 7.0          # WCAG "AAA" for normal text
    assert contrast(theme.TEXT, theme.CARD) >= 12.0
    assert contrast(theme.ON_ACCENT, theme.ACCENT) >= 7.0     # the Open button and selected rows


def test_check_boxes_have_a_visible_box(app):
    assert "QCheckBox::indicator" in theme.STYLE_SHEET
    assert "QCheckBox::indicator:checked" in theme.STYLE_SHEET
    assert contrast(theme.MUTED, theme.WINDOW) >= 7.0          # the outline against its fill


def test_a_check_box_is_drawn_with_an_outline_and_a_fill_when_checked(app):
    from PySide6.QtWidgets import QCheckBox
    old_palette, old_sheet = app.palette(), app.styleSheet()
    try:
        theme.apply_theme(app)
        box = QCheckBox("Show")
        box.show()
        app.processEvents()

        def indicator_pixels():
            image = box.grab().toImage()
            return {image.pixelColor(x, image.height() // 2).name() for x in range(0, 24)}

        empty = indicator_pixels()
        box.setChecked(True)
        app.processEvents()
        checked = indicator_pixels()
        assert theme.ACCENT.lower() in checked and theme.ACCENT.lower() not in empty
        assert any(QColor(c).lightness() > 120 for c in empty)       # the outline is visible
        box.close()
    finally:
        app.setPalette(old_palette)
        app.setStyleSheet(old_sheet)


def test_a_video_without_a_frame_rate_can_only_be_labeled_in_frames(app, monkeypatch):
    from types import SimpleNamespace
    window = shown(app)
    monkeypatch.setattr(window, "request_frame", lambda: None)
    monkeypatch.setattr(window, "refresh_controls", lambda: None)
    window.label_unit.setCurrentIndex(UNIT_TIME)
    window.setup_preview(SimpleNamespace(fps=0.0, frame_count=100, duration=0.0))
    assert window.label_unit.currentIndex() == UNIT_FRAMES
    assert window.label_unit.isEnabled()
    assert not window.label_unit.model().item(UNIT_TIME).isEnabled()
    assert window.frame_info.text().startswith("Frame 0 of 99")
    window.setup_preview(SimpleNamespace(fps=25.0, frame_count=100, duration=4.0))
    assert window.label_unit.model().item(UNIT_TIME).isEnabled()