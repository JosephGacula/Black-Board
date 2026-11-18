"""The color barcode strip: the colors of the segment that was analyzed."""
import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

BACKGROUND = QColor("#111111")
MESSAGE_COLOR = QColor("#888888")
MARKER = QColor("#ffffff")
MARKER_EDGE = QColor("#000000")
BARCODE_HEIGHT = 26


class BarcodeView(QWidget):
    """Draws the barcode of one segment, a white marker at the current frame
    (when it is inside the segment), and lets the user click or drag to jump
    to a moment of the segment."""

    seek_requested = Signal(int)        # a frame number

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = None              # stripes so far: (n x 3) BGR, or None
        self.total = 0                  # stripes the finished barcode will have
        self.first_frame = 0            # the segment the barcode is of
        self.last_frame = -1            # (last < first means: no segment)
        self.position = None            # the frame being looked at
        self.message = ""
        self.setMinimumWidth(100)
        self.setFixedHeight(BARCODE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # ---------- what to show ----------

    @property
    def span(self):
        """How many frames the segment holds; 0 when there is none."""
        return max(self.last_frame - self.first_frame + 1, 0)

    def clear(self, message=""):
        """Forget the segment, and show this message instead."""
        self.colors = None
        self.total = 0
        self.first_frame = 0
        self.last_frame = -1
        self.position = None
        self.message = message
        self.unsetCursor()
        self.update()

    def begin(self, first, last, total, message=""):
        """A barcode of frames first..last is starting: it will have `total`
        stripes. Nothing is measured yet."""
        self.colors = None
        self.total = total
        self.first_frame = first
        self.last_frame = last
        self.message = message
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def set_colors(self, colors):
        """Show the stripes measured so far (the rest stays empty)."""
        self.colors = None if colors is None else np.asarray(colors, np.uint8)
        self.update()

    def set_position(self, index):
        """Move the marker to this frame number (None hides it)."""
        self.position = index
        self.update()

    def set_message(self, message):
        self.message = message
        self.update()

    # ---------- geometry ----------

    def stripe_row(self, width):
        """One row of `width` pixels: the stripe behind every pixel, or the
        background where that stripe isn't measured yet. B, G, R order."""
        row = np.empty((width, 3), np.uint8)
        row[:] = (BACKGROUND.blue(), BACKGROUND.green(), BACKGROUND.red())
        if self.colors is None or len(self.colors) == 0 or self.total < 1:
            return row
        # The stripe under the middle of each pixel
        stripe = ((np.arange(width) + 0.5) * self.total / width).astype(np.int64)
        known = stripe < len(self.colors)
        row[known] = self.colors[stripe[known]]
        return row

    def marker_x(self):
        """The pixel of the marker, or None when there is nothing to mark."""
        if self.position is None or self.span < 1:
            return None
        if not self.first_frame <= self.position <= self.last_frame:
            return None
        offset = self.position - self.first_frame
        return int((offset + 0.5) / self.span * self.width())

    def frame_at(self, x):
        """The frame number under pixel x of the widget."""
        if self.span < 1:
            return None
        offset = int(x / max(self.width(), 1) * self.span)
        return self.first_frame + min(max(offset, 0), self.span - 1)

    # ---------- drawing ----------

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), BACKGROUND)

        row = np.ascontiguousarray(self.stripe_row(width).reshape(1, width, 3))
        image = QImage(row.data, width, 1, row.strides[0], QImage.Format.Format_BGR888)
        painter.drawImage(QRect(0, 0, width, height), image)

        if self.message and (self.colors is None or len(self.colors) == 0):
            painter.setPen(MESSAGE_COLOR)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)

        x = self.marker_x()
        if x is not None:
            painter.fillRect(x - 1, 0, 3, height, MARKER_EDGE)
            painter.fillRect(x, 0, 1, height, MARKER)

    # ---------- the mouse ----------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.seek_to(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.seek_to(event.position().x())

    def seek_to(self, x):
        frame = self.frame_at(x)
        if frame is not None:
            self.seek_requested.emit(frame)