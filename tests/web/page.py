"""
Page-test helpers: drive the real app:// page in a PanelWindow from pytest-qt.

``Page(qtbot, window)`` wraps a shown window whose page has booted:

- ``js(expr)``: evaluate an expression in the page and return its value,
  shipped back as JSON text (PySide6's runJavaScript returns scalars only)
- ``run(statements)``: run statements, no result
- ``wait_js(expr, timeout, pump)``: wait until an expression is truthy;
  ``pump()`` runs before every check (pull audio blocks for a real core)
- ``rect(selector)``, ``click(selector)``, ``wheel(selector, steps)``,
  ``drag(selector, dy, modifiers, dx=0)``, ``right_click``, ``double_click``,
  ``press``, ``type_text(text)``: real pointer and key input through QTest
  on the view's focus proxy, at an element's centre
- ``pixel(x, y)``, ``color_at(selector)``: coarse ``grab()`` pixel checks;
  ``close_to(color, expected, tolerance)``; ``wait_pixels(check)`` (painting
  lags the DOM: wait for pixels, never sleep)
- ``answer_dialog(path | None)``: choose a file in the dialog the page
  opened, or cancel it
- ``tick()``: push one bridge frame now; ``bound_addresses()``: the
  ``data-address`` of every control on the page; ``console_errors()``
"""

import json
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit

# Network errors of optional resources (fonts not fetched yet) are not page errors
IGNORED_CONSOLE = ('Failed to load resource',)


def same_path(a, b):
    """Paths equal across separators and case rules (Qt answers with / on Windows)."""
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


def choose_in_dialog(qtbot, dialog, path, timeout=5000):
    """Accept a (Qt widget) file dialog with a path typed into its name field
    (selectFile() leaves the field alone while it has the focus)."""
    path = str(path)
    edit = dialog.findChild(QLineEdit, 'fileNameEdit')
    if edit is not None:
        edit.setText(path)
    else:
        dialog.selectFile(path)
    qtbot.waitUntil(lambda: [same_path(f, path) for f in dialog.selectedFiles()] == [True],
                    timeout=timeout)
    dialog.accept()


class Page:
    def __init__(self, qtbot, window):
        self.qtbot = qtbot
        self.window = window

    # ------------------------------------------------------------------ parts
    @property
    def view(self):
        return self.window.view

    @property
    def page(self):
        return self.window.page

    @property
    def bridge(self):
        return self.window.bridge

    @property
    def core(self):
        return self.window.bridge.core

    # ------------------------------------------------------------------ script
    def _eval(self, source, timeout):
        with self.qtbot.waitCallback(timeout=timeout) as callback:
            self.page.runJavaScript(source, 0, callback)
        return callback.args[0] if callback.args else None

    def js(self, expression, timeout=5000):
        """The value of a JS expression, as JSON (undefined -> None)."""
        text = self._eval(f'JSON.stringify(({expression}) ?? null)', timeout)
        return None if text in (None, '') else json.loads(text)

    def run(self, statements, timeout=5000):
        self._eval(f'(() => {{ {statements}\n}})(); null', timeout)

    def wait_js(self, expression, timeout=5000, pump=None):
        def check():
            if pump is not None:
                pump()
            return bool(self.js(expression))
        self.qtbot.waitUntil(check, timeout=timeout)

    def wait_ready(self, timeout=15000):
        self.wait_js('window.pythonic && (window.pythonic.ready || window.pythonic.error)',
                     timeout)
        error = self.js('window.pythonic.error')
        assert error is None, f'page boot failed: {error}'
        # Input reaches the page once it has been painted (hit testing)
        self.run('window.__painted = false; requestAnimationFrame(() => '
                 'requestAnimationFrame(() => { window.__painted = true; }));')
        self.wait_js('window.__painted', timeout)

    # ------------------------------------------------------------------ input
    def rect(self, selector):
        """(x, y, width, height) of an element in view pixels."""
        box = self.js(f"(() => {{ const r = document.querySelector({json.dumps(selector)})"
                      f".getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }})()")
        return tuple(box)

    def center(self, selector):
        x, y, w, h = self.rect(selector)
        return QPoint(round(x + w / 2), round(y + h / 2))

    def click(self, selector):
        point = self.center(selector)
        target = self.view.focusProxy()
        QTest.mouseMove(target, point)
        QTest.mousePress(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                           point)

    def press(self, selector, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier,
              fx=0.5, fy=0.5):
        """Press a mouse button at a fraction of an element's box; returns the point."""
        x, y, w, h = self.rect(selector)
        point = QPoint(round(x + w * fx), round(y + h * fy))
        target = self.view.focusProxy()
        QTest.mouseMove(target, point)
        QTest.mousePress(target, button, modifiers, point)
        return point

    def drag(self, selector, dy, modifiers=Qt.KeyboardModifier.NoModifier, steps=4, fx=0.5, fy=0.5,
             dx=0):
        """Press on an element, move `dy` view pixels down (negative: up) and `dx`
        right in steps, release."""
        start = self.press(selector, modifiers=modifiers, fx=fx, fy=fy)
        target = self.view.focusProxy()
        point = start
        for i in range(1, steps + 1):
            point = QPoint(round(start.x() + dx * i / steps), round(start.y() + dy * i / steps))
            # A move with the button held and the modifiers (QTest.mouseMove has neither)
            event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point),
                                QPointF(target.mapToGlobal(point)), Qt.MouseButton.NoButton,
                                Qt.MouseButton.LeftButton, modifiers)
            QApplication.sendEvent(target, event)
            self.qtbot.wait(5)
        QTest.mouseRelease(target, Qt.MouseButton.LeftButton, modifiers, point)

    def right_click(self, selector):
        point = self.press(selector, Qt.MouseButton.RightButton)
        QTest.mouseRelease(self.view.focusProxy(), Qt.MouseButton.RightButton,
                           Qt.KeyboardModifier.NoModifier, point)

    def double_click(self, selector):
        """Two presses in quick succession, sent as raw events (QTest spaces its
        clicks by the double-click interval so they never pair up)."""
        point = self.center(selector)
        target = self.view.focusProxy()
        QTest.mouseMove(target, point)
        left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
        local, screen = QPointF(point), QPointF(target.mapToGlobal(point))
        stamp = int(time.monotonic() * 1000)
        for i, (kind, buttons) in enumerate([
                (QEvent.Type.MouseButtonPress, left), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
                (QEvent.Type.MouseButtonDblClick, left), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)]):
            event = QMouseEvent(kind, local, screen, left, buttons, none)
            event.setTimestamp(stamp + 20 * i)
            QApplication.sendEvent(target, event)

    def type_text(self, text, enter=True):
        """Type into the focused element of the page (Enter at the end)."""
        target = self.view.focusProxy()
        QTest.keyClicks(target, text)
        if enter:
            QTest.keyClick(target, Qt.Key.Key_Return)

    def wheel(self, selector, steps=1):
        """Turn the wheel over an element: steps > 0 is away from the user (up)."""
        point = QPointF(self.center(selector))
        target = self.view.focusProxy()
        QTest.mouseMove(target, point.toPoint())  # the page hit-tests the hovered element
        global_point = QPointF(target.mapToGlobal(point.toPoint()))
        event = QWheelEvent(point, global_point, QPoint(), QPoint(0, 120 * steps),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(target, event)

    # ------------------------------------------------------------------ pixels
    def pixel(self, x, y):
        """(r, g, b) of the view at a view pixel, from grab()."""
        image = self.view.grab().toImage()
        color = image.pixelColor(int(x * image.devicePixelRatio()),
                                 int(y * image.devicePixelRatio()))
        return color.red(), color.green(), color.blue()

    def color_at(self, selector, fx=0.5, fy=0.5):
        """(r, g, b) at a fraction of an element's box (default its centre)."""
        x, y, w, h = self.rect(selector)
        return self.pixel(x + w * fx, y + h * fy)

    @staticmethod
    def close_to(color, expected, tolerance=40):
        return all(abs(a - b) <= tolerance for a, b in zip(color, expected))

    def wait_pixels(self, check, timeout=5000):
        """Wait until check() holds on grabbed pixels (painting lags the DOM)."""
        self.qtbot.waitUntil(lambda: bool(check()), timeout=timeout)

    # ------------------------------------------------------------------ app
    def answer_dialog(self, path=None, timeout=5000):
        """Answer the file dialog the page opened: choose path, or cancel."""
        self.qtbot.waitUntil(lambda: len(self.bridge.open_dialogs()) == 1, timeout=timeout)
        (dialog,) = self.bridge.open_dialogs()
        if path is None:
            dialog.reject()
            return
        choose_in_dialog(self.qtbot, dialog, path, timeout)

    def tick(self):
        self.bridge.tick()

    def bound_addresses(self):
        return set(self.js("[...document.querySelectorAll('[data-address]')]"
                           ".map((e) => e.dataset.address)"))

    def console_errors(self):
        return [entry for entry in self.page.console if entry[0] == 'error'
                and not any(text in entry[1] for text in IGNORED_CONSOLE)]
