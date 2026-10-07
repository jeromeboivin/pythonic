"""
The QWebChannel bridge between the app core and the page.

Python -> JS: one coalesced plain signal per tick. A main-thread QTimer
(60 Hz) calls ``core.poll(since)`` and emits the result as JSON text on
``frame``; the audio callback only writes the snapshot that poll reads, it
never touches Qt. A tick stays within the frame budget (at most 4 ms of
Python per frame; ``stats`` keeps the timings).

JS -> Python: slots taking and returning JSON text (PySide6 hands strings
over unchanged, and JSON keeps one format both ways):

- ``get('["global.tempo", ...]')`` -> ``{"values": {addr: value}, "errors": {addr: message}}``
- ``set('[{"address", "value", "burst"?, "edit_all"?, "record"?}, ...]')`` (or one object)
  -> ``{"errors": {addr: message}}``; values arrive back through ``frame`` once applied
- ``act('{"verb", "args"?}')`` -> ``{"id": n}``; the result is the poll event with that id
- ``describe('"prefix"')`` (every registered address under it, ``""`` for all) or
  ``describe('["addr", ...]')`` -> ``{"addresses": {addr: metadata}, "errors": {...}}``
- ``gesture('begin' | 'end')``: brackets a drag as one undo step
- ``resync('null')``: the page (re)connected; the next frame carries all readouts
- ``fileDialog('{"mode": "open"|"save"|"folder", "title", "filters", "folder", "name", "suffix"}')``
  -> ``{"id": n}``; a window-modal ``QFileDialog`` opened with ``open()`` (never
  ``exec()``, which would stall the frame timer), its result emitted on ``dialog``
  as ``{"id", "path"}`` (``null`` when cancelled)
- ``resizeWindow('{"from": 1000, "to": 700}')`` -> ``{"size": [w, h]}``: the page's stage
  changed design height (the edit rack drawer closed or opened); the window grows or
  shrinks by the difference at the panel's scale (``PanelWindow.fit_stage_height``);
  ``{"size": null}`` without a window
- ``trigger('{"channel": 1..8, "velocity": 1..127}')`` -> ``{}``: hit a channel now
  (``core.trigger``, 0-based there), as a pad or a MIDI note would

Malformed requests answer ``{"error": message}``.
"""

import itertools
import json
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

FRAME_HZ = 60
FRAME_BUDGET_MS = 4.0


def _json_default(value):
    """JSON for the odd values a poll may carry (numpy scalars, sets, enums)."""
    item = getattr(value, 'item', None)
    if callable(item):
        return item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, 'value'):  # Enum
        return value.value
    return str(value)


def to_json(value):
    return json.dumps(value, separators=(',', ':'), default=_json_default)


class FrameStats:
    """Timings of the frame ticks, in milliseconds of Python time."""

    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.last_ms = 0.0
        self.over_budget = 0

    def add(self, ms):
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        self.max_ms = max(self.max_ms, ms)
        if ms > FRAME_BUDGET_MS:
            self.over_budget += 1

    @property
    def mean_ms(self):
        return self.total_ms / self.count if self.count else 0.0


class Bridge(QObject):
    """Published to the page as ``bridge``. See the module docstring."""

    frame = Signal(str)
    dialog = Signal(str)

    def __init__(self, core, parent=None, *, dialog_parent=None, window=None, hz=FRAME_HZ):
        super().__init__(parent)
        self.core = core
        self.dialog_parent = dialog_parent
        self.window = window
        self.stats = FrameStats()
        self._since = 0
        self._last_readouts = None
        self._dialog_ids = itertools.count(1)
        self._dialogs = {}
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(max(1, round(1000 / hz)))
        self._timer.timeout.connect(self.tick)

    # ------------------------------------------------------------------ frames
    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    @property
    def running(self):
        return self._timer.isActive()

    def tick(self):
        """Poll the core and push one frame (skipped when nothing changed)."""
        start = time.perf_counter()
        state = self.core.poll(self._since)
        self._since = state['version']
        readouts = [value for key, value in state.items()
                    if key not in ('version', 'changes', 'events')]
        if state['changes'] or state['events'] or readouts != self._last_readouts:
            self._last_readouts = readouts
            self.frame.emit(to_json(state))
        self.stats.add((time.perf_counter() - start) * 1000.0)

    # ------------------------------------------------------------------ slots
    @Slot(str, result=str)
    def get(self, request):
        try:
            addresses = _addresses(json.loads(request))
        except ValueError as exc:
            return to_json({'error': str(exc)})
        values, errors = {}, {}
        for address in addresses:
            try:
                values[address] = self.core.get(address)
            except Exception as exc:  # unknown address, a failing getter
                errors[address] = _message(exc)
        return to_json({'values': values, 'errors': errors})

    @Slot(str, result=str)
    def set(self, request):
        try:
            changes = json.loads(request)
        except ValueError as exc:
            return to_json({'error': str(exc)})
        if isinstance(changes, dict):
            changes = [changes]
        if not isinstance(changes, list):
            return to_json({'error': 'set takes a change or a list of changes'})
        errors = {}
        for change in changes:
            address = change.get('address') if isinstance(change, dict) else None
            if not isinstance(address, str):
                errors[str(address)] = 'a change needs an address'
                continue
            options = {key: change[key] for key in ('edit_all', 'burst', 'record')
                       if change.get(key) is not None}
            try:
                self.core.set(address, change.get('value'), **options)
            except Exception as exc:  # KeyError, ValueError, TypeError
                errors[address] = _message(exc)
        return to_json({'errors': errors})

    @Slot(str, result=str)
    def act(self, request):
        try:
            message = json.loads(request)
            verb = message['verb']
            args = message.get('args') or {}
            if not isinstance(verb, str) or not isinstance(args, dict):
                raise ValueError('act takes {"verb": name, "args": {...}}')
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return to_json({'error': _message(exc)})
        return to_json({'id': self.core.act(verb, **args)})

    @Slot(str, result=str)
    def describe(self, request):
        try:
            query = json.loads(request)
            if isinstance(query, str):
                addresses = self.core.registry.names(query)
            else:
                addresses = _addresses(query)
        except ValueError as exc:
            return to_json({'error': str(exc)})
        found, errors = {}, {}
        for address in addresses:
            try:
                found[address] = self.core.describe(address)
            except Exception as exc:
                errors[address] = _message(exc)
        return to_json({'addresses': found, 'errors': errors})

    @Slot(str)
    def resync(self, _request=''):
        """The page (re)connected: the next frame carries every readout even
        when unchanged (values are read with get)."""
        self._last_readouts = None

    @Slot(str)
    def gesture(self, phase):
        if phase == 'begin':
            self.core.begin_gesture()
        elif phase == 'end':
            self.core.end_gesture()

    @Slot(str, result=str)
    def fileDialog(self, request):  # noqa: N802 (JS-facing name)
        try:
            options = json.loads(request)
            if not isinstance(options, dict):
                raise ValueError('fileDialog takes an options object')
        except ValueError as exc:
            return to_json({'error': str(exc)})
        dialog_id = next(self._dialog_ids)
        dialog = build_file_dialog(options, self.dialog_parent)
        dialog.finished.connect(lambda result, i=dialog_id: self._dialog_finished(i, result))
        self._dialogs[dialog_id] = dialog
        dialog.open()
        return to_json({'id': dialog_id})

    @Slot(str, result=str)
    def resizeWindow(self, request):  # noqa: N802 (JS-facing name)
        try:
            options = json.loads(request)
            old, new = float(options['from']), float(options['to'])
            if old <= 0 or new <= 0:
                raise ValueError('stage heights must be positive')
        except (ValueError, KeyError, TypeError) as exc:
            return to_json({'error': _message(exc)})
        if self.window is None:
            return to_json({'size': None})
        return to_json({'size': list(self.window.fit_stage_height(old, new))})

    @Slot(str, result=str)
    def trigger(self, request):
        try:
            hit = json.loads(request)
            channel = hit['channel']
            velocity = hit.get('velocity', 127)
            if (not isinstance(channel, int) or isinstance(channel, bool)
                    or not 1 <= channel <= 8):
                raise ValueError(f'not a channel: {channel!r} (1-8)')
            if (not isinstance(velocity, int) or isinstance(velocity, bool)
                    or not 1 <= velocity <= 127):
                raise ValueError(f'not a velocity: {velocity!r} (1-127)')
        except (ValueError, KeyError, TypeError) as exc:
            return to_json({'error': _message(exc)})
        self.core.trigger(channel - 1, velocity)
        return to_json({})

    def _dialog_finished(self, dialog_id, result):
        dialog = self._dialogs.pop(dialog_id, None)
        path = None
        if dialog is not None:
            files = dialog.selectedFiles()
            if result == QFileDialog.DialogCode.Accepted and files:
                path = files[0]
            dialog.deleteLater()
        self.dialog.emit(to_json({'id': dialog_id, 'path': path}))

    def open_dialogs(self):
        """The file dialogs waiting for the user (tests)."""
        return list(self._dialogs.values())


def build_file_dialog(options, parent=None):
    """A QFileDialog set up from fileDialog options (not opened yet)."""
    dialog = QFileDialog(parent, options.get('title') or '')
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    mode = options.get('mode', 'open')
    if mode == 'folder':
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
    elif mode == 'save':
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        # The core refuses to replace a file and the page asks (the default
        # suffix is added after the dialog's own check), so one question only
        dialog.setOption(QFileDialog.Option.DontConfirmOverwrite, True)
    else:
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    if options.get('filters'):
        dialog.setNameFilters(list(options['filters']))
    if options.get('folder'):
        dialog.setDirectory(str(options['folder']))
    if options.get('name'):
        dialog.selectFile(str(options['name']))
    if options.get('suffix'):
        dialog.setDefaultSuffix(str(options['suffix']).lstrip('.'))
    return dialog


def _addresses(query):
    if isinstance(query, str):
        return [query]
    if isinstance(query, list) and all(isinstance(a, str) for a in query):
        return query
    raise ValueError('expected an address or a list of addresses')


def _message(exc):
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc) or type(exc).__name__
