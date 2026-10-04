from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWebEngineWidgets import QWebEngineView

HTML = """<body style="margin:0"><div id="b" style="width:200px;height:100px;background:#333"></div>
<script>
window.__log = [];
const b = document.getElementById('b');
for (const t of ['pointerdown','pointermove','pointerup','click','wheel'])
  b.addEventListener(t, e => window.__log.push(t + ':' + Math.round(e.clientY ?? 0)));
</script></body>"""


def js(qtbot, page, src):
    with qtbot.waitCallback() as cb:
        page.runJavaScript(src, 0, cb)
    return cb.args[0]


def test_qtest_mouse_reaches_page(qtbot):
    v = QWebEngineView()
    qtbot.addWidget(v)
    v.resize(400, 300)
    v.show()
    with qtbot.waitSignal(v.loadFinished):
        v.setHtml(HTML)
    qtbot.waitExposed(v)
    qtbot.wait(200)
    w = v.focusProxy()
    print('focusProxy', type(w).__name__)
    QTest.mousePress(w, Qt.LeftButton, Qt.NoModifier, QPoint(100, 80))
    QTest.mouseMove(w, QPoint(100, 30))
    QTest.mouseRelease(w, Qt.LeftButton, Qt.NoModifier, QPoint(100, 30))
    qtbot.wait(200)
    log = js(qtbot, v.page(), 'JSON.stringify(window.__log)')
    print('log', log)
    assert 'pointerdown:80' in log and 'pointerup:30' in log
