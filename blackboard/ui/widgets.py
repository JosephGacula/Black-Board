"""Small building blocks of the main window: a titled card, and the arrow
shaped buttons that run along the top."""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QRegion
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
)

from blackboard.ui import theme

CARD_MARGIN = 12
CARD_SPACING = 6

POINT = 14            # how far the arrow's tip sticks out, in pixels
OVERLAP = 4           # arrows are placed this much closer than their tips, leaving a thin gap
BUTTON_HEIGHT = 36
COMPACT_HEIGHT = 28     # the buttons that sit inside a card
PAD = 14


class Card(QFrame):
    """A panel with a title. Put the contents in `body`; anything that belongs
    next to the title (a drop-down, a status line) goes in `header`."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(
            f"QFrame#card {{ background-color: {theme.CARD}; border: 1px solid {theme.CARD_BORDER};"
            f" border-radius: 8px; }}")
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"font-weight: bold; color: {theme.MUTED}; background: transparent; border: none;")
        self.header = QHBoxLayout()
        self.header.addWidget(self.title_label)
        self.header.addStretch(1)
        self.body = QVBoxLayout()
        self.body.setSpacing(CARD_SPACING)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(CARD_MARGIN, CARD_MARGIN - 2, CARD_MARGIN, CARD_MARGIN)
        outer.setSpacing(CARD_SPACING)
        outer.addLayout(self.header)
        outer.addLayout(self.body)
        if not title:
            self.title_label.hide()

    def add_to_header(self, widget):
        """Place a widget at the right end of the title row."""
        self.header.addWidget(widget)
        return widget


class StepButton(QPushButton):
    """An ordinary rounded button that starts one of the analysis steps. While
    its step runs it is shown as busy (amber), and the same click can cancel it."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.busy = False
        self.setFixedHeight(COMPACT_HEIGHT)
        self.setMinimumWidth(110)
        self.setProperty("busy", False)

    def set_busy(self, busy):
        if busy != self.busy:
            self.busy = busy
            self.setProperty("busy", busy)
            self.style().unpolish(self)         # so the stylesheet notices the change
            self.style().polish(self)
            self.update()


class ChevronButton(QPushButton):
    """A button shaped like an arrow pointing right, so a row of them reads as
    steps. It behaves like any QPushButton. `first` gives it a flat left edge.
    `set_busy(True)` colors it for something that is running and can be cancelled."""

    def __init__(self, text="", first=False, primary=False, compact=False, parent=None):
        super().__init__(text, parent)
        self.first = first
        self.primary = primary
        self.busy = False
        self.height_wanted = COMPACT_HEIGHT if compact else BUTTON_HEIGHT
        self.setFixedHeight(self.height_wanted)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    # ---------- size ----------

    def sizeHint(self):
        width = QFontMetrics(self.font()).horizontalAdvance(self.text())
        return QSize(width + 2 * PAD + POINT + (0 if self.first else POINT // 2),
                     self.height_wanted)

    def minimumSizeHint(self):
        return self.sizeHint()

    def set_busy(self, busy):
        if busy != self.busy:
            self.busy = busy
            self.update()

    # ---------- drawing ----------

    def fill_color(self):
        if not self.isEnabled():
            return QColor(theme.CARD)
        if self.busy:
            base = QColor(theme.BUSY)
        elif self.primary:
            base = QColor(theme.ACCENT)
        else:
            base = QColor(theme.BUTTON)
        if self.isDown():
            return base.darker(125)
        if self.underMouse():
            return base.lighter(120)
        return base

    def shape(self):
        """The arrow's outline, with the tip on the right and a notch on the left."""
        width, height = self.width() - 1.0, self.height() - 1.0
        points = [QPointF(0, 0), QPointF(width - POINT, 0), QPointF(width, height / 2),
                  QPointF(width - POINT, height), QPointF(0, height)]
        if not self.first:
            points.append(QPointF(POINT, height / 2))
        return QPolygonF(points)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(0.5, 0.5)
        outline = QColor(theme.ACCENT) if self.hasFocus() else QColor(theme.CARD_BORDER)
        painter.setPen(QPen(outline, 1))
        painter.setBrush(self.fill_color())
        painter.drawPolygon(self.shape())
        if not self.isEnabled():
            text_color = theme.DISABLED_TEXT
        elif self.primary or self.busy:
            text_color = theme.ON_ACCENT
        else:
            text_color = theme.TEXT
        painter.setPen(QColor(text_color))
        left = 0 if self.first else POINT / 2
        painter.drawText(QRectF(left, 0, self.width() - POINT - left, self.height()),
                         Qt.AlignmentFlag.AlignCenter, self.text())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Clicks on the notch of an arrow go to the arrow behind it, because
        # the notch is part of that arrow's tip
        self.setMask(QRegion(self.shape().toPolygon()))

    def hitButton(self, pos):
        """Only the arrow itself reacts to clicks, not the corners around it."""
        return self.shape().containsPoint(QPointF(pos), Qt.FillRule.OddEvenFill)


class ElidedLabel(QLabel):
    """A label for text of unknown length, such as a file path. It never makes
    the window wider: text that does not fit is shortened in the middle
    ("C:/Users/.../Reel.mp4"), and the whole text is the tooltip."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text):
        self.full_text = text
        self.setToolTip(text)
        self.show_fitted()

    def show_fitted(self):
        fitted = QFontMetrics(self.font()).elidedText(
            self.full_text, Qt.TextElideMode.ElideMiddle, max(self.width() - 2, 0))
        super().setText(fitted if self.width() > 0 else self.full_text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.show_fitted()


STRIPE = 14           # width of one black or white stripe on the clapper arm
BAR_HEIGHT = COMPACT_HEIGHT       # the same height as the Open button


class ClapperProgressBar(QProgressBar):
    """A progress bar made like the striped arm of a film clapperboard: black
    and white diagonal stripes. Idle, the arm is dark; while a step runs, the
    finished part is black and white. When the length is unknown (range 0 to 0)
    the stripes slide along the whole bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(False)
        self.setFixedHeight(BAR_HEIGHT)
        self.offset = 0
        self.timer = QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self.slide)
        self.setRange(0, 100)
        self.setValue(0)

    def is_busy(self):
        """True when the length of the work is not known."""
        return self.minimum() == self.maximum()

    def fraction(self):
        if self.is_busy():
            return 1.0
        span = self.maximum() - self.minimum()
        return max(0.0, min(1.0, (self.value() - self.minimum()) / span))

    def slide(self):
        self.offset = (self.offset + 3) % (2 * STRIPE)
        self.update()

    def setRange(self, low, high):
        super().setRange(low, high)
        if hasattr(self, "timer"):
            if low == high and not self.timer.isActive():
                self.timer.start()
            elif low != high:
                self.timer.stop()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self.is_busy():
            self.timer.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        outline = QPainterPath()
        outline.addRoundedRect(rect, 3, 3)
        painter.setClipPath(outline)
        painter.fillRect(rect, QColor("#000000"))
        painter.save()
        filled = QRectF(rect.left(), rect.top(), rect.width() * self.fraction(), rect.height())
        for area, light, dark in ((rect, "#242424", "#000000"), (filled, "#f4f4f4", "#000000")):
            if area.width() <= 0:
                continue
            painter.setClipRect(area)
            painter.setClipPath(outline, Qt.ClipOperation.IntersectClip)
            painter.fillRect(area, QColor(dark))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(light))
            height = rect.height()
            x = rect.left() - height - 2 * STRIPE + self.offset
            while x < area.right() + height:
                painter.drawPolygon(QPolygonF([QPointF(x, rect.bottom()), QPointF(x + STRIPE, rect.bottom()),
                                               QPointF(x + STRIPE + height, rect.top()),
                                               QPointF(x + height, rect.top())]))
                x += 2 * STRIPE
        painter.restore()
        painter.setClipping(False)
        painter.setPen(QPen(QColor("#f4f4f4" if self.fraction() > 0 else theme.CARD_BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 3, 3)



class PreviewLabel(QLabel):
    """Shows a picture as large as the space allows, keeping its shape. It
    never makes the window bigger: the picture follows the label, not the
    other way round."""

    MIN_W, MIN_H = 320, 180
    WANT_W, WANT_H = 640, 360

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None                       # the picture as it was given
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(self.MIN_W, self.MIN_H)

    def sizeHint(self):
        return QSize(self.WANT_W, self.WANT_H)

    def minimumSizeHint(self):
        return QSize(self.MIN_W, self.MIN_H)

    def setPixmap(self, pixmap):
        self.source = pixmap
        self.fit()

    def setText(self, text):
        self.source = None
        super().setText(text)

    def clear(self):
        self.source = None
        super().clear()

    def fit(self):
        if self.source is None or self.source.isNull() or self.width() < 2 or self.height() < 2:
            if self.source is not None and not self.source.isNull():
                super().setPixmap(self.source)
            return
        super().setPixmap(self.source.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()