import app
from PySide6.QtWebEngineWidgets import QWebEngineView


def js(qtbot, page, src):
    """Evaluate src in the page's main world and return the (scalar) result."""
    with qtbot.waitCallback(timeout=5000) as cb:
        page.runJavaScript(src, 0, cb)
    return cb.args[0]


def test_knob_drag_reaches_core_and_frame_reaches_dom(qtbot):
    v = QWebEngineView()
    qtbot.addWidget(v)
    v.resize(800, 500)
    with qtbot.waitSignal(v.loadFinished, timeout=15000) as blk:
        core = app.build(v)
    assert blk.args == [True]
    qtbot.waitUntil(lambda: js(qtbot, v.page(), 'window.__ready === true'), timeout=10000)
    js(qtbot, v.page(), "document.getElementById('k').drag(-100); 1")
    qtbot.waitUntil(lambda: bool(core.calls), timeout=5000)
    assert core.calls == [('ch1.tune', 24.0)]
    core.frame.emit({'step': 7})
    qtbot.waitUntil(lambda: js(qtbot, v.page(), "document.getElementById('step').textContent") == '7',
                    timeout=5000)
