from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from blackboard.platforms import (
    DEFAULT_KEYS, FAIL, OK, PLACEMENTS, SPECS_CHECKED, WARN, check_places,
    clip_summary,
)

SYMBOLS = {OK: "✓", WARN: "!", FAIL: "✗"}
COLORS = {OK: "#5cb85c", WARN: "#e0a030", FAIL: "#e0605a"}

NOTHING_PICKED = "Pick at least one place on the left."
SEARCHING = "Looking for black bars in the picture..."
SEARCH_FAILED = "Could not look for black bars in this video."


class PlatformDialog(QDialog):
    """Pick where the video will be posted and see how well it suits each place."""

    def __init__(self, clip, selected=DEFAULT_KEYS, parent=None, help_text="",
                 searching=False):
        super().__init__(parent)
        self.setWindowTitle("Where will you post it?")
        self.clip = clip
        self.reports = []
        self.boxes = {}                       # placement key -> its check box

        self.intro = QLabel()
        self.intro.setWordWrap(True)
        if help_text:
            self.intro.setToolTip(help_text)
        self.status = QLabel(SEARCHING if searching else "")
        self.status.setStyleSheet("color: #aaa;")
        self.show_clip()

        # --- left: the places to choose from, grouped by platform ---
        choices = QVBoxLayout()
        title = QLabel("Where will you post it?")
        title.setStyleSheet("font-weight: bold;")
        choices.addWidget(title)
        platform = None
        for place in PLACEMENTS:
            if place.platform != platform:
                platform = place.platform
                heading = QLabel(platform)
                heading.setStyleSheet("color: #aaa; margin-top: 8px;")
                choices.addWidget(heading)
            box = QCheckBox(place.name)
            box.setChecked(place.key in selected)
            box.setToolTip(place.note)
            box.toggled.connect(self.rebuild)
            self.boxes[place.key] = box
            choices.addWidget(box)
        everything = QPushButton("All")
        nothing = QPushButton("None")
        everything.clicked.connect(lambda: self.pick_all(True))
        nothing.clicked.connect(lambda: self.pick_all(False))
        buttons = QHBoxLayout()
        buttons.addWidget(everything)
        buttons.addWidget(nothing)
        choices.addSpacing(8)
        choices.addLayout(buttons)
        choices.addStretch(1)
        choice_panel = QWidget()
        choice_panel.setLayout(choices)
        choice_panel.setFixedWidth(190)

        # --- right: one card per chosen place, best fit first ---
        self.cards = QVBoxLayout()
        self.cards.setAlignment(Qt.AlignmentFlag.AlignTop)
        holder = QWidget()
        holder.setLayout(self.cards)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(holder)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        body = QHBoxLayout()
        body.addWidget(choice_panel)
        body.addWidget(self.scroll, 1)

        footer = QLabel(f"Specs checked {SPECS_CHECKED} against the platforms' public "
                        "guides. Platforms change them, so treat this as a guide.")
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #888;")
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.addWidget(footer, 1)
        bottom.addWidget(close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.intro)
        layout.addWidget(self.status)
        layout.addLayout(body, 1)
        layout.addLayout(bottom)
        self.resize(860, 640)
        self.rebuild()

    # ---------- what is being checked ----------

    def show_clip(self):
        self.intro.setText(f"Checking your video: {clip_summary(self.clip)}")

    def set_clip(self, clip):
        """The facts changed (the black bars were found): check again."""
        self.clip = clip
        self.status.setText("")
        self.show_clip()
        self.rebuild()

    def search_failed(self):
        self.status.setText(SEARCH_FAILED)

    # ---------- the choices ----------

    def selected_keys(self):
        return [key for key, box in self.boxes.items() if box.isChecked()]

    def pick_all(self, on):
        for box in self.boxes.values():
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
        self.rebuild()

    # ---------- the cards ----------

    def rebuild(self, *_):
        while self.cards.count():
            item = self.cards.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.reports = check_places(self.clip, self.selected_keys())
        if not self.reports:
            self.cards.addWidget(QLabel(NOTHING_PICKED))
        for report in self.reports:
            self.cards.addWidget(self.make_card(report))

    def make_card(self, report):
        color = COLORS[report.verdict]
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(f"QFrame#card {{ border: 1px solid {color}; border-radius: 6px; }}")
        layout = QVBoxLayout(card)

        head = QLabel(f"<b>{report.placement.name}</b> &nbsp; "
                      f"<span style='color:{color}'>{SYMBOLS[report.verdict]} "
                      f"{report.verdict_text}</span>")
        head.setStyleSheet("border: none;")
        layout.addWidget(head)
        for check in report.checks:
            line = QLabel(f"<span style='color:{COLORS[check.status]}'>"
                          f"{SYMBOLS[check.status]}</span> <b>{check.name}</b>: "
                          f"{check.message}")
            line.setWordWrap(True)
            line.setStyleSheet("border: none;")
            layout.addWidget(line)
        return card