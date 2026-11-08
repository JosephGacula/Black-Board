"""A small widget that draws a vectorscope: the picture's colors on a map."""
import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from blackboard.analyze import chroma_of_color
from blackboard.ui.waveform_view import fit_axis

BACKGROUND = QColor("#111111")
GUIDES = QColor("#3a3a3a")
TRACE = (160, 230, 170)            # pale green, as (red, green, blue)

# A cell holding at least this share of all the pixels is drawn at full
# strength. Rarer cells are fainter. Most of a picture's colors are spread
# over many cells, so this is a small share.
FULL_SHARE = 0.001

# Space left around the map, in pixels
MARGIN = 5

# The six boxes: where pure red, yellow, green, cyan, blue and magenta land.
# Each is (blue, green, red), like the frames.
TARGET_COLORS = ((0, 0, 255), (0, 255, 255), (0, 255, 0),
                 (255, 255, 0), (255, 0, 0), (255, 0, 255))
TARGET_SIZE = 7


def vectorscope_intensity(counts):
    """Turn vectorscope counts into brightnesses from 0 to 255.

    What matters is what share of all the pixels sit in a cell, not the raw
    count, so a big picture and a small one look the same. The share is eased
    with a square root, so thin traces from a few pixels still show up.
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return np.zeros(counts.shape, np.uint8)
    strength = np.sqrt(np.clip(counts / total / FULL_SHARE, 0.0, 1.0))
    return np.round(strength * 255).astype(np.uint8)


def vectorscope_picture(counts, side):
    """The vectorscope as a (side, side, 3) array of colors.

    Blue is to the right and red toward the top, so the picture matches how
    the scope is read.
    """
    strength = vectorscope_intensity(counts)
    strength = fit_axis(fit_axis(strength, side, 0), side, 1)[::-1]
    picture = np.empty((side, side, 3), np.uint8)
    for channel, top in enumerate(TRACE):
        picture[..., channel] = (strength.astype(np.uint16) * top // 255).astype(np.uint8)
    return picture


def map_point(cb, cr, side):
    """Where a color sits in a map `side` pixels wide, as (x, y) from its
    top-left corner. Cb runs left to right and Cr bottom to top."""
    return (cb + 128) / 256 * side, (128 - cr) / 256 * side


class VectorscopeView(QWidget):
    """Draws every color in the picture on a flat color map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = None                # size x size counts, or None
        self.setMinimumWidth(200)
        self.setFixedHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def set_vectorscope(self, counts):
        """Show these counts, or pass None to clear the vectorscope."""
        self.counts = None if counts is None else np.asarray(counts)
        self.update()

    def plot_area(self):
        """The square the map is drawn in, as (left, top, side)."""
        side = max(min(self.width(), self.height()) - 2 * MARGIN, 1)
        return (self.width() - side) // 2, (self.height() - side) // 2, side

    def point_of(self, cb, cr):
        """The pixel of this widget where a color is drawn."""
        left, top, side = self.plot_area()
        x, y = map_point(cb, cr, side)
        return left + x, top + y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        left, top, side = self.plot_area()

        # Guides: the circle that holds the colors, and the cross through
        # the middle where gray sits
        painter.setPen(QPen(GUIDES, 1))
        painter.drawEllipse(QRectF(left, top, side, side))
        middle_x, middle_y = left + side / 2, top + side / 2
        painter.drawLine(int(left), int(middle_y), int(left + side), int(middle_y))
        painter.drawLine(int(middle_x), int(top), int(middle_x), int(top + side))

        if self.counts is not None:
            picture = np.ascontiguousarray(vectorscope_picture(self.counts, side))
            image = QImage(picture.data, side, side, picture.strides[0],
                           QImage.Format.Format_RGB888)
            # Add the trace to what is already there, so the guides show through
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            painter.drawImage(left, top, image)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # The six boxes, hollow so the trace shows inside them
        half = TARGET_SIZE / 2
        for blue, green, red in TARGET_COLORS:
            x, y = self.point_of(*chroma_of_color(blue, green, red))
            painter.setPen(QPen(QColor(red, green, blue), 1))
            painter.drawRect(QRectF(x - half, y - half, TARGET_SIZE, TARGET_SIZE))