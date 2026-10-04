import json

import app
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from test_qt_page import js


def test_component_tests_run_inside_engine(qtbot):
    """ui/test.html imports *.engine-spec.js modules and leaves results on window.__results."""
    v = QWebEngineView()
    qtbot.addWidget(v)
    app.build(v)  # installs the app:// handler
    v.load(QUrl('app://ui/test.html'))
    qtbot.waitUntil(lambda: js(qtbot, v.page(), 'Array.isArray(window.__results)') is True, timeout=10000)
    # runJavaScript only returns scalars in PySide6 6.11: ship structured results as JSON text
    res = json.loads(js(qtbot, v.page(), 'JSON.stringify(window.__results)'))
    print(res)
    # 3 node:test-style specs from model.test.js (via the import map), then 2 DOM specs;
    # the last one fails on purpose to show failures and stacks come back
    assert [r['ok'] for r in res] == [True, True, True, True, False]


def test_grab_offscreen_has_pixels(qtbot):
    v = QWebEngineView()
    qtbot.addWidget(v)
    v.resize(400, 200)
    v.show()
    with qtbot.waitSignal(v.loadFinished, timeout=15000):
        v.setHtml('<body style="margin:0;background:rgb(0,200,0)"></body>')
    qtbot.wait(300)
    img = v.grab().toImage()
    c = img.pixelColor(200, 100)
    print('grab', img.width(), img.height(), c.red(), c.green(), c.blue())
    assert (c.red(), c.green(), c.blue()) == (0, 200, 0)
