"""Shared test setup.

Several test files open windows and dialogs and never close them. If they are
still alive when Python exits, Qt has already shut down and the program crashes
(Windows reports "access violation" after the results are printed). This closes
and deletes every leftover window once, after the last test, while Qt is
still running."""
import gc

import pytest


@pytest.fixture(scope="session", autouse=True)
def close_leftover_windows():
    yield
    try:
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:                       # tests that need no Qt
        return
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        try:
            widget.close()                    # a window waits for its background threads
            widget.deleteLater()
        except RuntimeError:                  # already deleted
            pass
    for _ in range(5):
        app.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()