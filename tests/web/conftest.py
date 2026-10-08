"""
Fixtures of the web front-end tests (pytest-qt, offscreen QtWebEngine).

- ``core_table`` (session): describe() and values of every address of a real core
- ``fake_core``: a FakeCore built from it
- ``panel``: the real page in a shown PanelWindow over the fake core, booted
  (a ``Page``, see page.py); fails on JS errors at teardown
- ``real_core``: a started AppCore on a fake audio backend (pull blocks with
  ``real_core.backend.stream.pull()``), closed at teardown
- ``open_panel(core)``: the page over any core

Headless: QT_QPA_PLATFORM defaults to offscreen; CI also sets
QTWEBENGINE_DISABLE_SANDBOX=1. The app:// scheme is registered here, before
pytest-qt creates the QApplication, and its handler is installed once per
profile by PanelWindow.
"""

import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def pytest_collection_modifyitems(items):
    """The start-up test runs the application's event loop to its end, and
    QtWebEngine shuts down with the application (aboutToQuit): no page loads
    after it in this process, so it runs last."""
    def last(item):
        return item.path.name == 'test_startup.py' and item.path.parent.name == 'web'
    items[:] = [i for i in items if not last(i)] + [i for i in items if last(i)]


try:
    import PySide6.QtWebEngineWidgets  # noqa: F401  (before the QApplication exists)
    import pytestqt  # noqa: F401
    import shiboken6
except ImportError:  # no QtWebEngine or no pytest-qt: nothing here can run
    collect_ignore_glob = ['test_*.py']
else:
    from pythonic.web.app import prepare_rendering
    from pythonic.web.scheme import register_scheme

    prepare_rendering(gpu=False)
    register_scheme()

    from tests.web.fake_core import FakeCore, core_table as build_core_table
    from tests.web.page import Page

    @pytest.fixture(scope='session')
    def core_table():
        return build_core_table()

    @pytest.fixture
    def fake_core(core_table):
        return FakeCore(core_table)

    @pytest.fixture
    def open_panel(qtbot):
        from pythonic.web.window import PanelWindow

        windows = []

        def open_(core, owns_core=True):
            window = PanelWindow(core, owns_core=owns_core, echo_console=False)
            windows.append(window)
            qtbot.addWidget(window)
            with qtbot.waitSignal(window.view.loadFinished, timeout=15000) as loaded:
                window.show()
            assert loaded.args == [True]
            page = Page(qtbot, window)
            page.wait_ready()
            return page

        yield open_
        errors = []
        for window in windows:
            if shiboken6.isValid(window):
                errors += Page(qtbot, window).console_errors()
                window.close()
                window.dispose()
        assert not errors, f'JS errors: {errors}'

    @pytest.fixture
    def panel(open_panel, fake_core):
        return open_panel(fake_core)

    @pytest.fixture
    def real_core(prefs):
        from pythonic.app import AppCore
        from tests.fake_audio import FakeAudioBackend

        backend = FakeAudioBackend()
        core = AppCore(preferences=prefs, audio_backend=backend, midi_backend=None,
                       stall_timeout=None)
        core.backend = backend
        started = core.start()
        assert core.wait(started)['status'] == 'done'
        yield core
        core.close()
