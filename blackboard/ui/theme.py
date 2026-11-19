"""The look of the whole app: one dark theme, set once when the app starts."""
from PySide6.QtGui import QColor, QPalette

WINDOW = "#080808"        # behind everything
CARD = "#121213"          # the panels
CARD_BORDER = "#2a2a2d"
FIELD = "#050505"         # inside boxes you type in
TEXT = "#f0f0f0"
MUTED = "#b4b4ba"         # captions and titles
ACCENT = "#b9b9be"        # highlights: a light gray (text on it is ON_ACCENT)
ON_ACCENT = "#0a0a0a"
BUTTON = "#1d1d1f"
BUTTON_HOVER = "#2a2a2d"
BUTTON_DOWN = "#161617"
BUSY = "#c98a1b"          # a step that is running and can be cancelled
DISABLED_TEXT = "#5a5a5f"

STYLE_SHEET = f"""
QToolTip {{ background-color: {CARD}; color: {TEXT}; border: 1px solid {CARD_BORDER}; padding: 4px; }}
QPushButton {{ background-color: {BUTTON}; color: {TEXT}; border: 1px solid {CARD_BORDER};
               border-radius: 6px; padding: 4px 12px; }}
QPushButton:hover {{ background-color: {BUTTON_HOVER}; }}
QPushButton:pressed {{ background-color: {BUTTON_DOWN}; }}
QPushButton[busy="true"] {{ background-color: {BUSY}; color: {ON_ACCENT}; border-color: {BUSY}; }}
QPushButton:disabled {{ color: {DISABLED_TEXT}; background-color: {CARD}; }}
QSlider::groove:horizontal {{ height: 4px; background: {CARD_BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 12px; height: 12px; margin: -4px 0;
                              border-radius: 6px; }}
QSlider::handle:horizontal:disabled {{ background: {DISABLED_TEXT}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {MUTED}; border-radius: 3px;
                        background: {WINDOW}; }}
QCheckBox::indicator:hover {{ border-color: {TEXT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:disabled {{ border-color: {DISABLED_TEXT}; }}
QProgressBar {{ background: {CARD}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QScrollBar:vertical {{ background: {WINDOW}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {CARD_BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""


def make_palette():
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: WINDOW,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: FIELD,
        QPalette.ColorRole.AlternateBase: CARD,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: BUTTON,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.ToolTipBase: CARD,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: ON_ACCENT,
        QPalette.ColorRole.PlaceholderText: MUTED,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(DISABLED_TEXT))
    return palette


def apply_theme(app):
    """Give the whole application the dark look. Call it once, right after
    creating the QApplication."""
    app.setStyle("Fusion")
    app.setPalette(make_palette())
    app.setStyleSheet(STYLE_SHEET)