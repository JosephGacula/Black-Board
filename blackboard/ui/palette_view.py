"""A small widget that shows the main colors of a picture."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

BACKGROUND = QColor("#111111")

# Height of the strip along the bottom that shows each color's real share
STRIP_HEIGHT = 10

# Narrower tiles than this show only a whole-number percent, and narrower
# than the second limit show no color code
HEX_MIN_WIDTH = 64
PERCENT_DECIMAL_MIN_WIDTH = 52


def readable_text_color(red, green, blue):
    """Black or white, whichever is easier to read on this background color."""
    brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return QColor("#000000") if brightness > 140 else QColor("#ffffff")


def palette_tip(colors):
    """The palette as lines of text: '1. #7a1f2b  42.1 %'."""
    return "\n".join(f"{number}. {color.hex}  {color.share:.1f} %"
                     for number, color in enumerate(colors, start=1))


class PaletteView(QWidget):
    """Draws one tile for each color, and a strip showing their shares.

    The tiles are all the same size so every color is easy to see and read.
    The strip along the bottom is split by share, so you can see how much of
    the picture each color really covers.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = None                # a list of PaletteColor, or None
        self.help_text = ""
        self.setMinimumWidth(200)
        self.setFixedHeight(92)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def set_help(self, text):
        """The explanation shown when hovering, above the list of colors."""
        self.help_text = text
        self.refresh_tip()

    def set_palette(self, colors):
        """Show these colors, or pass None to clear the palette."""
        self.colors = None if colors is None else list(colors)
        self.refresh_tip()
        self.update()

    def refresh_tip(self):
        lines = [self.help_text] if self.help_text else []
        if self.colors:
            lines.append(palette_tip(self.colors))
        self.setToolTip("\n\n".join(lines))

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), BACKGROUND)
        if not self.colors:
            return

        count = len(self.colors)
        tile_height = height - STRIP_HEIGHT
        tile_width = width / count
        bold = QFont(painter.font())
        bold.setBold(True)
        small = QFont(painter.font())
        small.setPointSizeF(max(bold.pointSizeF() - 1.5, 6.0))

        painter.setPen(Qt.PenStyle.NoPen)
        for position, color in enumerate(self.colors):
            left = position * tile_width
            tile = QRectF(left, 0, tile_width, tile_height)
            painter.setBrush(QColor(color.red, color.green, color.blue))
            painter.drawRect(tile)

            painter.setPen(readable_text_color(color.red, color.green, color.blue))
            percent = (f"{color.share:.1f} %" if tile_width >= PERCENT_DECIMAL_MIN_WIDTH
                       else f"{color.share:.0f}%")
            painter.setFont(bold)
            if tile_width >= HEX_MIN_WIDTH:
                painter.drawText(QRectF(left, tile_height * 0.2, tile_width, tile_height * 0.3),
                                 Qt.AlignmentFlag.AlignCenter, percent)
                painter.setFont(small)
                painter.drawText(QRectF(left, tile_height * 0.5, tile_width, tile_height * 0.3),
                                 Qt.AlignmentFlag.AlignCenter, color.hex)
            else:
                painter.drawText(tile, Qt.AlignmentFlag.AlignCenter, percent)
            painter.setPen(Qt.PenStyle.NoPen)

        # The strip: each color as wide as its share of the picture
        total = sum(color.share for color in self.colors) or 1.0
        left = 0.0
        for color in self.colors:
            segment = color.share / total * width
            painter.setBrush(QColor(color.red, color.green, color.blue))
            painter.drawRect(QRectF(left, tile_height, segment, STRIP_HEIGHT))
            left += segment