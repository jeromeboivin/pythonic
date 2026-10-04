import pathlib
import sys

import pytest

pytest.importorskip('PySide6.QtWebEngineWidgets')  # spike only runs where QtWebEngine is installed

HERE = pathlib.Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import app  # noqa: E402

app.register_scheme()  # must run before pytest-qt creates the QApplication
