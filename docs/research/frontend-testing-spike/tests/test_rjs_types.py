from PySide6.QtWebEngineWidgets import QWebEngineView

EXPRS = ['42', '"s"', '[1,2]', '({a:1})', '[{a:1}]', '({x:[{a:1}]})', '[[1],[2]]',
         'Promise.resolve(1)', 'undefined']


def test_types(qtbot):
    v = QWebEngineView()
    qtbot.addWidget(v)
    with qtbot.waitSignal(v.loadFinished):
        v.setHtml('<p>x</p>')
    for e in EXPRS:
        with qtbot.waitCallback() as cb:
            v.page().runJavaScript(e, 0, cb)
        print(f'{e:20} -> {cb.args[0]!r}')
