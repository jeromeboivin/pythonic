"""
The JS specs in pythonic/web/static/test/ (node:test style), run inside
QtWebEngine through runner.html's import map, so CI needs no Node.
One pytest test per spec file; each spec's failures are listed.
"""

import json

import pytest
import shiboken6
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

from pythonic.web import STATIC_DIR
from pythonic.web.scheme import install_handler

SPEC_DIR = STATIC_DIR / 'test'
SPECS = sorted(p.name for p in SPEC_DIR.iterdir()
               if p.name.endswith('.test.js') or p.name.endswith('.engine-spec.js'))


@pytest.fixture
def view(qtbot):
    install_handler()
    view = QWebEngineView()
    qtbot.addWidget(view)
    yield view
    shiboken6.delete(view)  # a page must not outlive its profile


def run_specs(qtbot, view, *specs):
    """Load the runner with these spec files; returns its results."""
    query = '&'.join(f'spec={name}' for name in specs)
    view.load(QUrl(f'app://ui/test/runner.html?{query}'))

    def results():
        with qtbot.waitCallback(timeout=5000) as callback:
            view.page().runJavaScript('JSON.stringify(window.__results ?? null)', 0, callback)
        return json.loads(callback.args[0] or 'null')

    qtbot.waitUntil(lambda: results() is not None, timeout=15000)
    return results()


def test_there_are_specs():
    assert 'store.test.js' in SPECS and 'panel.engine-spec.js' in SPECS


@pytest.mark.parametrize('spec', SPECS)
def test_js_spec(qtbot, view, spec):
    results = run_specs(qtbot, view, spec)
    assert results, f'{spec} registered no tests'
    failures = [f"{r['name']}: {r['error']}" for r in results if not r['ok']]
    assert not failures, '\n\n'.join(failures)


def test_failures_and_import_errors_are_reported(qtbot, view):
    results = run_specs(qtbot, view, 'missing.test.js')
    assert [r['ok'] for r in results] == [False]
    assert results[0]['name'] == '(import)'
