import sys
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot, QTimer, QBuffer, QByteArray, QUrl, QIODevice
from PySide6.QtWebEngineCore import (QWebEngineUrlScheme, QWebEngineUrlSchemeHandler,
                                     QWebEngineUrlRequestJob, QWebEngineProfile)

UI = Path(__file__).parent / 'ui'
MIME = {'.html': b'text/html', '.js': b'text/javascript', '.json': b'application/json'}


def register_scheme():
    s = QWebEngineUrlScheme(b'app')
    s.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    F = QWebEngineUrlScheme.Flag
    s.setFlags(F.SecureScheme | F.LocalAccessAllowed | F.CorsEnabled | F.FetchApiAllowed)
    QWebEngineUrlScheme.registerScheme(s)


class Handler(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        p = (UI / job.requestUrl().path().lstrip('/')).resolve()
        if UI not in p.parents or not p.is_file():
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buf = QBuffer(parent=job)
        buf.setData(QByteArray(p.read_bytes()))
        buf.open(QIODevice.ReadOnly)
        job.reply(MIME.get(p.suffix, b'application/octet-stream'), buf)


class FakeCore(QObject):
    """Stands in for the real bridge object: records set() calls, pushes frames on demand."""
    frame = Signal('QVariantMap')

    def __init__(self):
        super().__init__()
        self.calls = []

    @Slot(str, float)
    def set(self, address, value):
        self.calls.append((address, value))
        print('SET', address, round(value, 3), flush=True)


_HANDLER = None


def install_handler():
    """Once per profile: Qt refuses a second handler for the same scheme, and the
    profile does not own the handler, so keep it alive for the whole process."""
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = Handler()
        QWebEngineProfile.defaultProfile().installUrlSchemeHandler(b'app', _HANDLER)


def build(view):
    from PySide6.QtWebChannel import QWebChannel
    install_handler()
    core = FakeCore()
    ch = QWebChannel()
    ch.registerObject('core', core)
    view.page().setWebChannel(ch)
    view._keep = (core, ch)
    view.load(QUrl('app://ui/index.html'))
    return core


if __name__ == '__main__':
    register_scheme()
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWebEngineWidgets import QWebEngineView
    app = QApplication(sys.argv)
    v = QWebEngineView()
    v.resize(800, 500)
    core = build(v)
    v.show()
    n = [0]

    def tick():
        n[0] += 1
        core.frame.emit({'step': n[0] % 16})

    t = QTimer()
    t.timeout.connect(tick)
    t.start(16)
    sys.exit(app.exec())
