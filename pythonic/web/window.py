"""
The web interface's top-level window: a QWebEngineView showing
``app://ui/index.html`` with the bridge published as ``bridge``.

The page draws a fixed 1600x1000 panel scaled to the window and letterboxed
(``static/js/stage.js``); the window opens at the minimum 1280x800 (scale
0.8) and cannot shrink below it. Closing the window stops the frame timer
and, when it owns the core, closes it.
"""

import sys

import shiboken6
from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .bridge import Bridge
from .scheme import INDEX_URL, install_handler

MIN_SIZE = QSize(1280, 800)
PANEL_BACKGROUND = '#000000'


class PanelPage(QWebEnginePage):
    """Keeps the page's console messages and forwards warnings and errors to
    stderr (``console`` holds ``(level, message, line, source)`` tuples)."""

    LEVELS = {
        QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel: 'info',
        QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel: 'warning',
        QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel: 'error',
    }

    def __init__(self, profile, parent=None, *, echo=True):
        super().__init__(profile, parent)
        self.console = []
        self.echo = echo

    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
        name = self.LEVELS.get(level, 'info')
        self.console.append((name, message, line, source))
        if self.echo and name != 'info':
            print(f'[js {name}] {message} ({source}:{line})', file=sys.stderr, flush=True)


class PanelWindow(QWidget):
    """The window of the web interface over an app core (or a stand-in with
    the same interface). ``closed`` is emitted once the window has closed."""

    closed = Signal()

    def __init__(self, core, *, owns_core=True, url=INDEX_URL, profile=None, echo_console=True):
        super().__init__()
        self.core = core
        self.owns_core = owns_core
        self.setWindowTitle('Pythonic')
        self.setMinimumSize(MIN_SIZE)
        self.setStyleSheet(f'background: {PANEL_BACKGROUND};')

        profile = profile or QWebEngineProfile.defaultProfile()
        install_handler(profile)
        self.view = QWebEngineView(self)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.page = PanelPage(profile, self.view, echo=echo_console)
        self.page.setBackgroundColor(QColor(PANEL_BACKGROUND))
        self.view.setPage(self.page)

        self.bridge = Bridge(core, self, dialog_parent=self)
        self.channel = QWebChannel(self)
        self.channel.registerObject('bridge', self.bridge)
        self.page.setWebChannel(self.channel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.resize(MIN_SIZE)

        self.bridge.start()
        self.view.load(QUrl(url))

    def closeEvent(self, event):  # noqa: N802
        self.shutdown()
        super().closeEvent(event)
        self.closed.emit()

    def shutdown(self):
        """Stop the frames and, when the window owns it, close the core."""
        self.bridge.stop()
        if self.owns_core and self.core is not None:
            core, self.core = self.core, None
            core.close()

    def dispose(self):
        """Shut down and delete the window with its page now: a page that
        outlives its profile (at interpreter exit) crashes QtWebEngine."""
        self.shutdown()
        shiboken6.delete(self)
