"""
The web interface: the HTML/JS panel in ``static/`` shown in a PySide6
QtWebEngine window, talking to the app core over a QWebChannel bridge.

- ``scheme``: the ``app://`` URL scheme serving ``static/`` (registered
  before the QApplication exists, its handler installed once per profile).
- ``bridge``: the ``Bridge`` QObject: get/set/act/describe slots (JSON text in
  and out) and one coalesced ``frame`` signal per timer tick.
- ``window``: ``PanelWindow``, the top-level window with the web view.
- ``app``: ``run()``, the ``pythonic --ui web`` entry point.

Nothing here is imported by the package itself, so the rest of Pythonic
works without PySide6.
"""

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / 'static'
