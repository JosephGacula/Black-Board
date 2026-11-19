import sys

from PySide6.QtWidgets import QApplication

from blackboard.ui.main_window import MainWindow
from blackboard.ui.theme import apply_theme


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.showMaximized()           # the picture and the chart grow with the screen
    sys.exit(app.exec())


if __name__ == "__main__":
    main()