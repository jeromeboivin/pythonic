"""Probe: does a 60 Hz main-thread QTimer (the frame push) keep running while a QFileDialog is up?

Each variant opens a dialog over a QWebEngineView, closes it after 1.5 s via
QApplication.activeModalWidget(), and reports timer ticks and page-side frames during that time.
Usage: modal.py <variant>  (static | exec | open | open-nonnative)
"""
import sys
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QFileDialog

variant = sys.argv[1]
app = QApplication(sys.argv)
print('platform theme:', app.platformName(), flush=True)
view = QWebEngineView()
view.setHtml('<script>window.n = 0</script><body>frames</body>')
view.resize(600, 400)
view.show()
ticks = [0]


def tick():
    ticks[0] += 1
    view.page().runJavaScript('window.n++')


timer = QTimer()
timer.setTimerType(Qt.PreciseTimer)
timer.timeout.connect(tick)
state = {}


def closer():
    w = QApplication.activeModalWidget()
    state['modal_widget'] = type(w).__name__ if w else None
    state['window_modality'] = w.windowModality().name if w else None
    state['ticks_at_close'] = ticks[0] - state['t0']
    state['elapsed'] = round(time.perf_counter() - state['start'], 2)
    if w:
        w.reject()


def report():
    def got(n):
        state['page_frames_total'] = n
        print('RESULT', variant, state, flush=True)
        app.quit()
    view.page().runJavaScript('window.n', 0, got)


def start():
    timer.start(16)
    QTimer.singleShot(300, open_dialog)


def open_dialog():
    state['t0'] = ticks[0]
    state['start'] = time.perf_counter()
    QTimer.singleShot(1500, closer)
    if variant == 'static':
        r = QFileDialog.getSaveFileName(view, 'Save preset', '', 'Preset (*.mtpreset *.json)')
        state['returned_after_s'] = round(time.perf_counter() - state['start'], 2)
        state['ticks_while_blocked'] = ticks[0] - state['t0']
        QTimer.singleShot(300, report)
    elif variant == 'exec':
        d = QFileDialog(view, 'Save preset')
        d.setAcceptMode(QFileDialog.AcceptSave)
        d.exec()
        state['ticks_while_blocked'] = ticks[0] - state['t0']
        QTimer.singleShot(300, report)
    else:
        d = QFileDialog(view, 'Save preset')
        d.setAcceptMode(QFileDialog.AcceptSave)
        if variant == 'open-nonnative':
            d.setOption(QFileDialog.DontUseNativeDialog)
        d.finished.connect(lambda r: QTimer.singleShot(300, report))
        d.open()
        state['open_returned_after_s'] = round(time.perf_counter() - state['start'], 3)
        view._d = d


QTimer.singleShot(1500, start)
QTimer.singleShot(15000, lambda: (print('TIMEOUT', variant, state, flush=True), app.quit()))
app.exec()
