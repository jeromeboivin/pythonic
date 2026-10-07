"""
``pythonic --ui web``: the QApplication, the app core and the panel window.
"""

import os
import sys

from .scheme import register_scheme


def prepare_headless(environ=None):
    """With Qt's offscreen platform, Chromium must render in software: with a
    display around it tries the GPU and draws nothing. Call before Qt starts."""
    environ = os.environ if environ is None else environ
    if not environ.get('QT_QPA_PLATFORM', '').startswith('offscreen'):
        return
    flags = environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
    if '--disable-gpu' not in flags.split():
        environ['QTWEBENGINE_CHROMIUM_FLAGS'] = f'{flags} --disable-gpu'.strip()


def frame_report(stats):
    """One line on the frame ticks of a run (``--quit-after`` prints it)."""
    from .bridge import FRAME_BUDGET_MS
    return (f'UI frames: {stats.count} ticks, mean {stats.mean_ms:.2f} ms, max {stats.max_ms:.2f} ms, '
            f'{stats.over_budget} over the {FRAME_BUDGET_MS:g} ms budget')


def run(quit_after=None, devtools_port=None, core=None):
    """Start the web interface and block until its window closes; returns the
    exit code. ``quit_after`` (seconds) closes the window by itself, prints
    the frame timings and exits non-zero when the page did not load (start-up
    checks)."""
    prepare_headless()
    register_scheme()  # before the QApplication exists
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from .window import PanelWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('Pythonic')
    if core is None:
        from pythonic.app import AppCore
        core = AppCore()
    window = PanelWindow(core)
    # The app ends with its window, whatever other top-level windows remain
    window.closed.connect(app.quit)
    loaded = []
    window.view.loadFinished.connect(loaded.append)
    # As tkinter: the last preset loads before the core freezes the start-up
    # heap and opens the stream
    core.act('preset.load_last')
    core.start()
    window.show()
    if devtools_port is not None:
        print(f'DevTools: http://127.0.0.1:{devtools_port}', flush=True)
    ready = []
    if quit_after is not None:
        def check_and_close():
            def done(value):
                ready.append(value is True)
                window.close()
            window.page.runJavaScript('!!(window.pythonic && window.pythonic.ready)', 0, done)
        QTimer.singleShot(int(quit_after * 1000), check_and_close)
    app.exec()
    stats = window.bridge.stats
    window.dispose()
    if quit_after is not None:
        print(frame_report(stats), file=sys.stderr, flush=True)
    if quit_after is not None and not (loaded and loaded[-1] and ready == [True]):
        print(f'The web interface did not start (loaded {loaded}, ready {ready})',
              file=sys.stderr)
        return 1
    return 0
