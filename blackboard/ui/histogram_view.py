"""A small widget that draws a brightness histogram."""
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from blackboard.analyze import ZONE_DARK_MAX, ZONE_MID_MAX

BINS = 256
BACKGROUND = QColor("#111111")
BARS = QColor("#d0d0d0")
ZONE_LINES = QColor("#3a3a3a")


# The chart is never scaled so tall that a bar holding less than this share
# of all the pixels reaches the top. Without it, a few stray pixels (the soft
# edge between a black area and a white one, say) would be stretched to full
# height and look like a big part of the picture.
MIN_SCALE_SHARE = 0.02


def bar_heights(counts, height):
    """Turn 256 pixel counts into bar heights from 0 to `height`.

    The bars are scaled so the tallest one fits, but the two end bars (pure
    black and pure white) are left out when picking that scale. A frame with
    lots of crushed blacks would otherwise squash every other bar flat. The
    end bars are cut off at the top instead, which is how video scopes show
    clipping: a bar running all the way up at either end.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.zeros(counts.size)      # nothing to draw
    tallest_middle = counts[1:-1].max() if counts.size > 2 else 0
    scale = max(tallest_middle, total * MIN_SCALE_SHARE)
    return np.minimum(counts / scale, 1.0) * height


class HistogramView(QWidget):
    """Draws 256 bars: black on the left, white on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.histogram = None             # 256 counts, or None for "nothing"
        self.setMinimumWidth(200)
        self.setFixedHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def set_histogram(self, counts):
        """Show these 256 counts, or pass None to clear the chart."""
        self.histogram = None if counts is None else np.asarray(counts)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), BACKGROUND)

        # Faint lines mark the dark / mid / bright thirds
        painter.setPen(QPen(ZONE_LINES, 1))
        for edge in (ZONE_DARK_MAX + 1, ZONE_MID_MAX + 1):
            x = round(edge / BINS * width)
            painter.drawLine(x, 0, x, height)

        if self.histogram is None:
            return

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(BARS)
        for i, bar in enumerate(bar_heights(self.histogram, height)):
            if bar <= 0:
                continue
            left = i * width / BINS
            right = (i + 1) * width / BINS
            painter.drawRect(QRectF(left, height - bar, max(right - left, 1), bar))