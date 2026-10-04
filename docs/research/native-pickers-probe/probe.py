"""Probe: what QtWebEngine hands chooseFiles() for each web picker, and what JS gets back.

Runs with an overridden chooseFiles() that never shows a dialog: it logs its arguments and
returns a prepared path. Clicks are real input events (QTest) so the page has user activation.
Run: QT_QPA_PLATFORM=offscreen python probe.py   (set NOFSA=1 to leave fileSystemAccessRequested unhandled)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPoint, QTimer, QUrl, Qt
from PySide6.QtWebEngineCore import (QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestJob,
                                     QWebEngineUrlScheme, QWebEngineUrlSchemeHandler)

HERE = Path(__file__).parent
TMP = Path(tempfile.mkdtemp(prefix='pick-'))
(TMP / 'kit.mtpreset').write_text('{"name": "kit"}')
(TMP / 'a.wav').write_bytes(b'RIFF0000WAVE')
(TMP / 'b.wav').write_bytes(b'RIFF0000WAVE')
(TMP / 'folder').mkdir()
(TMP / 'folder' / 'x.json').write_text('{}')
LOG = []


def register_scheme():
    s = QWebEngineUrlScheme(b'app')
    s.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    F = QWebEngineUrlScheme.Flag
    s.setFlags(F.SecureScheme | F.LocalAccessAllowed | F.CorsEnabled | F.FetchApiAllowed)
    QWebEngineUrlScheme.registerScheme(s)


class Handler(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        p = HERE / job.requestUrl().path().lstrip('/')
        buf = QBuffer(parent=job)
        buf.setData(QByteArray(p.read_bytes()))
        buf.open(QIODevice.ReadOnly)
        job.reply(b'text/html', buf)


class Page(QWebEnginePage):
    def chooseFiles(self, mode, oldFiles, acceptedMimeTypes):
        entry = {'mode': mode.name, 'oldFiles': list(oldFiles), 'accept': list(acceptedMimeTypes)}
        M = QWebEnginePage.FileSelectionMode
        if mode == M.FileSelectOpen:
            ret = [str(TMP / 'kit.mtpreset')]
        elif mode == M.FileSelectOpenMultiple:
            ret = [str(TMP / 'a.wav'), str(TMP / 'b.wav')]
        elif mode == M.FileSelectUploadFolder:
            ret = [str(TMP / 'folder')]
        else:
            ret = [str(TMP / 'saved.mtpreset')]
        entry['returned'] = ret
        LOG.append(('chooseFiles', entry))
        print('chooseFiles', json.dumps(entry), flush=True)
        return ret

    def javaScriptConsoleMessage(self, level, msg, line, src):
        print('JS', msg, flush=True)
        if msg.startswith('RESULT '):
            LOG.append(('js', json.loads(msg[7:])))


def main():
    register_scheme()
    from PySide6.QtTest import QTest
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    prof = QWebEngineProfile.defaultProfile()
    h = Handler()
    prof.installUrlSchemeHandler(b'app', h)
    view = QWebEngineView()
    page = Page(prof, view)
    view.setPage(page)

    def fsa(req):
        e = {'origin': req.origin().toString(), 'path': req.filePath().toString(),
             'handleType': req.handleType().name, 'access': str(req.accessFlags())}
        LOG.append(('fileSystemAccessRequested', e))
        print('fileSystemAccessRequested', e, flush=True)
        req.accept()
    if not os.environ.get('NOFSA'):
        page.fileSystemAccessRequested.connect(fsa)

    view.resize(900, 600)
    view.show()
    ids = ['in1', 'in2', 'in3', 'fsOpen', 'fsSave', 'fsDir']
    state = {'i': 0}

    def click_next():
        if state['i'] >= len(ids):
            page.runJavaScript('probeEnv().then(e => window.__env = JSON.stringify(e))', 0, lambda r: QTimer.singleShot(3500, lambda: page.runJavaScript('window.__env', 0, lambda e: finish(json.loads(e)))))
            return
        bid = ids[state['i']]
        state['i'] += 1

        def got(rect):
            print('click', bid, rect, flush=True)
            x, y = rect
            QTest.mouseClick(view.focusProxy(), Qt.LeftButton, Qt.NoModifier, QPoint(int(x), int(y)))
            QTimer.singleShot(1200, click_next)
        page.runJavaScript(f'JSON.stringify(centre("{bid}"))', 0, lambda r: got(json.loads(r)))

    def finish(env):
        print('ENV', json.dumps(env), flush=True)
        saved = TMP / 'saved.mtpreset'
        print('saved file exists:', saved.exists(), repr(saved.read_text()) if saved.exists() else '')
        print('written into folder:', sorted(p.name for p in (TMP / 'folder').iterdir()))
        app.quit()

    page.loadFinished.connect(lambda ok: (print('loaded', ok, flush=True), QTimer.singleShot(500, click_next)))
    view.load(QUrl('app://ui/probe.html'))
    app.exec()


if __name__ == '__main__':
    main()
