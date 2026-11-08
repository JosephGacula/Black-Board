"""A small widget that draws a luma waveform."""
import numpy as np
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

BACKGROUND = QColor("#111111")
GUIDES = QColor("#3a3a3a")
TRACE = (160, 230, 170)            # pale green, as (red, green, blue)

# A brightness level that holds at least this share of the pixels in a slice
# of the picture is drawn at full strength. Rarer levels are fainter.
FULL_SHARE = 0.05

# Guide lines are drawn at these brightness levels, as a share of the height
GUIDE_LEVELS = (0, 0.25, 0.5, 0.75, 1.0)


def waveform_intensity(counts):
    """Turn waveform counts (256 x slices) into brightnesses from 0 to 255.

    Every slice is judged on its own: what matters is what share of its
    pixels sit at each level, not the raw count. The share is eased with a
    square root, so the thin traces from a few pixels still show up.
    Row 0 of the result is black, like the counts.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=0)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    strength = np.sqrt(np.clip(shares / FULL_SHARE, 0.0, 1.0))
    return np.round(strength * 255).astype(np.uint8)


def fit_axis(values, size, axis):
    """Stretch or squeeze an array to `size` entries along one axis.

    Squeezing keeps the brightest entry of each group instead of averaging,
    so a thin trace stays as bright as it was and doesn't fade out when the
    waveform is drawn smaller than its 256 levels.
    """
    count = values.shape[axis]
    if size == count:
        return values
    if size > count:                              # stretch: repeat entries
        return np.take(values, np.arange(size) * count // size, axis=axis)
    starts = np.arange(size) * count // size      # squeeze: keep the brightest
    return np.maximum.reduceat(values, starts, axis=axis)


def waveform_picture(counts, width, height):
    """The waveform as a (height, width, 3) array of colors.

    White is at the top and black at the bottom, so the picture matches how
    the scope is read.
    """
    strength = waveform_intensity(counts)
    strength = fit_axis(fit_axis(strength, height, 0), width, 1)[::-1]
    picture = np.empty((height, width, 3), np.uint8)
    for channel, top in enumerate(TRACE):
        picture[..., channel] = (strength.astype(np.uint16) * top // 255).astype(np.uint8)
    return picture


class WaveformView(QWidget):
    """Draws the brightness of the picture across its width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = None                # 256 x slices counts, or None
        self.setMinimumWidth(200)
        self.setFixedHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def set_waveform(self, counts):
        """Show these counts, or pass None to clear the waveform."""
        self.counts = None if counts is None else np.asarray(counts)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), BACKGROUND)
        if width < 1 or height < 1:
            return

        # Guide lines at 0, 25, 50, 75 and 100 percent brightness
        painter.setPen(QPen(GUIDES, 1))
        for level in GUIDE_LEVELS:
            y = round((1 - level) * (height - 1))
            painter.drawLine(0, y, width, y)

        if self.counts is None:
            return
        picture = np.ascontiguousarray(waveform_picture(self.counts, width, height))
        image = QImage(picture.data, width, height, picture.strides[0],
                       QImage.Format.Format_RGB888)
        # Add the trace to what is already there, so the guides show through
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        painter.drawImage(0, 0, image)