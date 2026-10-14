import sys
from PySide6.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel("Black Board: hello world")
label.resize(300, 100)
label.show()
sys.exit(app.exec())