"""Choosing a segment of the video: where it starts and where it ends."""
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QValidator
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QSpinBox, QWidget,
)

BAND_BACKGROUND = QColor("#2a2a2a")
BAND_SEGMENT = QColor("#3d8bfd")
BAND_MARKER = QColor("#ffffff")
BAND_CUT = QColor("#ffcc00")
BAND_HEIGHT = 8
MIN_SEGMENT_WIDTH = 2        # so even a one-frame segment can be seen

UNIT_FRAMES, UNIT_TIME = 0, 1        # the two ways to enter the segment
HIGH_FRAME_RATE = 60                 # above this, times show 3 decimals


def format_time(seconds, places=2):
    """Seconds as m:ss.cc, or h:mm:ss.cc from one hour on."""
    scale = 10 ** places
    total = round(max(seconds, 0.0) * scale)
    whole, fraction = divmod(total, scale)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    tail = f"{secs:02d}.{fraction:0{places}d}"
    if hours:
        return f"{hours}:{minutes:02d}:{tail}"
    return f"{minutes}:{tail}"


_TIME_PATTERN = re.compile(
    r"^\s*(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d*)?|\.\d+)\s*s?\s*$")


def parse_time(text):
    """Seconds from text like '90', '1:30', '1:30.5' or '0:01:30.25'.

    Returns None when the text is not a time.
    """
    match = _TIME_PATTERN.match(text)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return (int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds))


class TimeBox(QDoubleSpinBox):
    """A box for a time in seconds that shows and accepts m:ss.cc."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.places = 2
        self.setDecimals(3)
        self.setKeyboardTracking(False)       # wait until the user is done typing

    def textFromValue(self, value):
        return format_time(value, self.places)

    def valueFromText(self, text):
        parsed = parse_time(text)
        return self.value() if parsed is None else parsed

    def validate(self, text, pos):
        if text.strip() == "" or parse_time(text) is not None:
            return QValidator.State.Acceptable, text, pos
        if re.fullmatch(r"[\d:.\s]*s?", text):        # still being typed
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos


class SegmentBand(QWidget):
    """A thin band under the slider. The chosen segment is lit up, and a
    white tick shows the frame being looked at."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.total = 0               # frames in the video; 0 = no video
        self.first = 0
        self.last = -1
        self.position = None
        self.cuts = []               # frames where a new shot starts
        self.setFixedHeight(BAND_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_total(self, total):
        self.total = total
        self.first, self.last, self.position = 0, total - 1, None
        self.cuts = []
        self.update()

    def clear(self):
        self.total = 0
        self.first, self.last, self.position = 0, -1, None
        self.cuts = []
        self.update()

    def set_segment(self, first, last):
        self.first, self.last = first, last
        self.update()

    def set_position(self, index):
        self.position = index
        self.update()

    def set_cuts(self, cuts):
        """Mark the frames where a new shot starts (an empty list removes them)."""
        self.cuts = list(cuts)
        self.update()

    def cut_xs(self):
        """The pixel of every cut mark."""
        if self.total < 1:
            return []
        return [min(self.x_of(frame), max(self.width() - 1, 0))
                for frame in self.cuts if 0 <= frame < self.total]

    def x_of(self, frame):
        """The pixel where this frame starts."""
        return int(frame / self.total * self.width())

    def segment_rect(self):
        """(x, width) of the lit-up part, or None when there is nothing."""
        if self.total < 1 or self.last < self.first:
            return None
        left = self.x_of(self.first)
        right = self.x_of(self.last + 1)
        return left, max(right - left, MIN_SEGMENT_WIDTH)

    def marker_x(self):
        if self.total < 1 or self.position is None:
            return None
        return int((self.position + 0.5) / self.total * self.width())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), BAND_BACKGROUND)
        lit = self.segment_rect()
        if lit is not None:
            painter.fillRect(lit[0], 0, lit[1], self.height(), BAND_SEGMENT)
        for x in self.cut_xs():
            painter.fillRect(x, 0, 1, self.height(), BAND_CUT)
        x = self.marker_x()
        if x is not None:
            painter.fillRect(x, 0, 1, self.height(), BAND_MARKER)


class SegmentPicker(QWidget):
    """Start and end boxes, each with a button that takes the frame being
    looked at. The start never goes past the end. The boxes can be read and
    typed in frames or in time (m:ss.cc); the frames are what is kept."""

    changed = Signal(int, object)           # first, last (a person edited them)
    start_from_current = Signal()           # the "Use current frame" buttons
    end_from_current = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps = 0.0

        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        self.start_time = TimeBox()
        self.end_time = TimeBox()
        self.start_button = QPushButton("Use current")
        self.end_button = QPushButton("Use current")
        for button in (self.start_button, self.end_button):
            button.setToolTip("Use the frame you are looking at")
        self.length_label = QLabel("")
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Frames", "Time"])
        self.unit_combo.setToolTip("Enter the start and end as frame numbers or as times")
        self.high = 0                  # the last frame the boxes allow

        self.start_spin.valueChanged.connect(self.on_start_edited)
        self.end_spin.valueChanged.connect(self.on_end_edited)
        self.start_button.clicked.connect(self.start_from_current.emit)
        self.end_button.clicked.connect(self.end_from_current.emit)
        self.start_time.valueChanged.connect(lambda seconds: self.on_time_edited(True, seconds))
        self.end_time.valueChanged.connect(lambda seconds: self.on_time_edited(False, seconds))
        self.unit_combo.currentIndexChanged.connect(self.show_unit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Start:"))
        layout.addWidget(self.start_spin)
        layout.addWidget(self.start_time)
        layout.addWidget(self.start_button)
        layout.addSpacing(8)
        layout.addWidget(QLabel("End:"))
        layout.addWidget(self.end_spin)
        layout.addWidget(self.end_time)
        layout.addWidget(self.end_button)
        layout.addSpacing(8)
        layout.addWidget(self.length_label)
        layout.addStretch()
        layout.addWidget(self.unit_combo)

        self.clear()
        self.show_unit()

    # ---------- setting it up ----------

    def clear(self):
        """No video: everything is off."""
        self.fps = 0.0
        self.set_range(0, 0)
        self.length_label.setText("")
        self.update_unit_choice()
        self.setEnabled(False)

    def set_video(self, frame_count, fps):
        """A video with this many frames is open; select all of it."""
        self.fps = fps
        self.set_range(0, frame_count - 1)
        self.set_segment(0, frame_count - 1)
        self.update_unit_choice()
        self.setEnabled(True)

    def set_range(self, low, high):
        self.high = high
        for spin in (self.start_spin, self.end_spin):
            spin.blockSignals(True)
            spin.setRange(low, high)
            spin.setValue(low)
            spin.blockSignals(False)
        self.sync_time_boxes()

    def set_segment(self, first, last):
        """Show this segment, without telling anyone it changed."""
        for spin, value in ((self.start_spin, first), (self.end_spin, last)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.update_length()
        self.sync_time_boxes()

    def segment(self):
        """(first, last) of the chosen segment."""
        return self.start_spin.value(), self.end_spin.value()

    # ---------- editing ----------

    def set_first(self, frame):
        """Move the start; the end follows if it would be passed."""
        self.start_spin.setValue(frame)

    def set_last(self, frame):
        """Move the end; the start follows if it would be passed."""
        self.end_spin.setValue(frame)

    def on_start_edited(self, value):
        if value > self.end_spin.value():
            self.end_spin.blockSignals(True)
            self.end_spin.setValue(value)
            self.end_spin.blockSignals(False)
        self.segment_edited()

    def on_end_edited(self, value):
        if value < self.start_spin.value():
            self.start_spin.blockSignals(True)
            self.start_spin.setValue(value)
            self.start_spin.blockSignals(False)
        self.segment_edited()

    def segment_edited(self):
        self.update_length()
        self.sync_time_boxes()
        first, last = self.segment()
        self.changed.emit(first, last)

    def update_length(self):
        first, last = self.segment()
        frames = last - first + 1
        text = f"{frames} frame" + ("" if frames == 1 else "s")
        if self.fps > 0:
            text += f", {frames / self.fps:.2f} s"
        self.length_label.setText(text)

    # ---------- frames or time ----------

    def unit(self):
        """UNIT_FRAMES or UNIT_TIME: what the boxes show now."""
        return self.unit_combo.currentIndex()

    def show_unit(self, _index=None):
        in_time = self.unit() == UNIT_TIME
        self.start_spin.setVisible(not in_time)
        self.end_spin.setVisible(not in_time)
        self.start_time.setVisible(in_time)
        self.end_time.setVisible(in_time)

    def update_unit_choice(self):
        """Times need a frame rate. Without one only frames can be used."""
        has_rate = self.fps > 0
        self.unit_combo.setEnabled(has_rate)
        if not has_rate:
            self.unit_combo.setCurrentIndex(UNIT_FRAMES)

    def sync_time_boxes(self):
        """Make the time boxes show the frames the other boxes hold."""
        fps = self.fps
        places = 3 if fps > HIGH_FRAME_RATE else 2
        step = 1 / fps if fps > 0 else 1.0
        for box, spin in ((self.start_time, self.start_spin),
                          (self.end_time, self.end_spin)):
            box.blockSignals(True)
            box.places = places
            box.setRange(0.0, self.high / fps if fps > 0 else 0.0)
            box.setSingleStep(step)
            box.setValue(spin.value() / fps if fps > 0 else 0.0)
            box.blockSignals(False)
            # Redraw the text, which does not change by itself with the value
            box.lineEdit().setText(box.textFromValue(box.value()))

    def on_time_edited(self, is_start, seconds):
        """A time was typed or stepped: turn it into the nearest frame."""
        if self.fps <= 0:
            return
        frame = max(0, min(round(seconds * self.fps), self.high))
        spin = self.start_spin if is_start else self.end_spin
        if spin.value() != frame:
            spin.setValue(frame)          # this keeps start <= end and reports it
        self.sync_time_boxes()            # show the exact time of that frame