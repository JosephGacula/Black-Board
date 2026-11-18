"""The shot list: a table of the shots found, and numbers describing them."""
from bisect import bisect_right

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

COLUMNS = ["Shot", "Start", "End", "Length"]
LIST_WIDTH = 250
LIST_HEIGHT = 190


def seconds_text(seconds):
    return f"{seconds:.2f} s"


def summary_text(summary, first, last):
    """The lines of numbers under the table."""
    return "\n".join([
        f"{summary.count} shot" + ("" if summary.count == 1 else "s")
        + f" in frames {first} to {last}",
        f"Average {seconds_text(summary.average)}, median {seconds_text(summary.median)}",
        f"Shortest {seconds_text(summary.shortest)}, longest {seconds_text(summary.longest)}",
        f"Cutting pace: {summary.cuts_per_minute:.1f} cuts per minute",
    ])


class ShotListView(QWidget):
    """A table with one row per shot. Clicking a row asks to jump to the
    first frame of that shot. The row of the frame being looked at is
    highlighted."""

    seek_requested = Signal(int)            # a frame number

    def __init__(self, parent=None):
        super().__init__(parent)
        self.starts = []                    # first frame of every shot

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setFixedSize(LIST_WIDTH, LIST_HEIGHT)
        self.table.cellClicked.connect(self.on_row_clicked)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setFixedWidth(LIST_WIDTH)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addWidget(self.summary)

    # ---------- what to show ----------

    def clear(self, message=""):
        """Empty the table, and show this message under it."""
        self.starts = []
        self.table.setRowCount(0)
        self.summary.setText(message)

    def set_message(self, message):
        self.summary.setText(message)

    def set_shots(self, shots, summary, first, last):
        """Show these shots, with the numbers that describe them."""
        self.starts = [shot.start for shot in shots]
        self.table.setRowCount(len(shots))
        for row, shot in enumerate(shots):
            cells = [str(shot.number), str(shot.start), str(shot.end),
                     seconds_text(shot.seconds)]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, column, item)
        self.table.clearSelection()
        self.summary.setText(summary_text(summary, first, last))

    # ---------- following the frame being looked at ----------

    def row_of(self, frame):
        """The row of the shot that holds this frame, or None if it is
        before the first shot or there are no shots."""
        if not self.starts:
            return None
        row = bisect_right(self.starts, frame) - 1
        return row if row >= 0 else None

    def set_position(self, frame):
        row = self.row_of(frame)
        if row is None:
            self.table.clearSelection()
            return
        if row != self.table.currentRow() or not self.table.selectedItems():
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))

    # ---------- the mouse ----------

    def on_row_clicked(self, row, column):
        if 0 <= row < len(self.starts):
            self.seek_requested.emit(self.starts[row])