# Web front-end

The new interface: an HTML/JS panel in a PySide6 QtWebEngine window, on top
of the app core ([core interface](core-interface.md)). No JS build step: ES
modules served as they are. Decisions: map #1, bridge #2, layout #7/#12,
frame budget #10, look #16, testing #17/#18, launch #19.

## Running it

```bash
pip install -e ".[dev]"            # PySide6 is a dependency (not on Windows ARM64)
pythonic                           # web interface (default); same as python run.py
pythonic --ui tk                   # tkinter interface
pythonic --devtools [PORT]         # Chromium DevTools on http://127.0.0.1:9222 (or PORT)
pythonic --quit-after 5            # close after 5 s; non-zero exit if the page did not boot
```

Without QtWebEngine, `--ui web` starts tkinter with a dialog that explains
why and offers to install PySide6 (on Windows ARM64 it only explains).
`pythonic/web/static/index.html` also opens in a plain browser served over
HTTP (`python -m http.server -d pythonic/web/static`): a fake bridge then
stands in for the core (tempo and START/STOP work).

Fonts: `python tools/fetch_fonts.py` downloads the bundled OFL fonts into
`pythonic/web/static/fonts/` (see `SOURCES.txt` there); until they are
committed the panel falls back to system fonts.

## Layout

```
pythonic/web/
  __init__.py      STATIC_DIR
  scheme.py        app:// scheme: register_scheme() before QApplication, install_handler(profile)
  bridge.py        Bridge QObject (slots + frame signal), FrameStats, FRAME_HZ = 60, FRAME_BUDGET_MS = 4
  window.py        PanelWindow(core, owns_core=True): view + page + channel; MIN_SIZE 1280x800; dispose()
  app.py           run(quit_after, devtools_port, core): the --ui web entry; prepare_headless()
  static/
    index.html     loads css/*, qrc:///qtwebchannel/qwebchannel.js, js/main.js
    css/tokens.css design tokens (palette, fonts, stage size) as custom properties
    css/fonts.css  @font-face of the bundled fonts
    css/panel.css  stage and panel styles
    fonts/         bundled fonts + licences (SOURCES.txt)
    js/bridge.js       transport: connectBridge(), webChannelBridge(remote), createFakeBridge(opts)
    js/core-client.js  createCoreClient(bridge, {schedule})
    js/store.js        createStore(), READOUTS
    js/stage.js        fitStage(w, h), mountStage(el), STAGE_WIDTH/HEIGHT
    js/panel.js        mountPanel(stage, {store, client, meta}), PANEL_ADDRESSES
    js/main.js         boot({bridge, stage}), demoBridge(); sets window.pythonic
    test/              runner.html, shim/, *.test.js (pure), *.engine-spec.js (DOM)
```

`window.pythonic` = `{ bridge, client, store, meta, panel, ready }` once
booted (`{ ready: false, error }` if boot failed).

## Bridge (Python <-> JS)

Published on the QWebChannel as `bridge`. JS -> Python slots take and return
**JSON text**; malformed requests answer `{"error": message}`.

| Slot | Request | Reply |
|---|---|---|
| `get` | `["addr", ...]` (or one address) | `{"values": {addr: value}, "errors": {addr: msg}}` |
| `set` | `[{"address", "value", "burst"?, "edit_all"?, "record"?}, ...]` (or one) | `{"errors": {addr: msg}}` |
| `act` | `{"verb", "args"?}` | `{"id": n}`; the result is the poll event with that id |
| `describe` | `"prefix"` (registered addresses under it, `""` = all) or `["addr", ...]` | `{"addresses": {addr: describe()}, "errors": {...}}` |
| `gesture` | `begin` / `end` (plain text) | none: brackets a drag as one undo step |
| `resync` | `null` | none: the next frame carries every readout |
| `fileDialog` | `{"mode": "open"\|"save"\|"folder", "title", "filters": ["Presets (*.mtpreset)"], "folder", "name", "suffix"}` | `{"id": n}`; then the `dialog` signal |

Signals (JSON text): `frame` = one `core.poll(since)` result per tick of a
60 Hz main-thread QTimer, sent when it has changes or events or a readout
changed (`version`, `changes`, `events`, `transport`, `modulation`, `audio`,
`midi`); `dialog` = `{"id", "path"}` (`null` when cancelled). Dialogs are
window-modal `QFileDialog`s opened with `open()` (never `exec()`): the frame
timer keeps running. Save dialogs do not confirm overwrites; the core's save
verbs refuse to replace a file and the page asks. Device lists come from
`describe()` labels (`pref.audio.device`, `midi.device`) and the rescan verbs.
`bridge.stats` (`FrameStats`) keeps tick timings against the 4 ms budget.

## JS modules

**bridge.js** - a bridge is `{ kind, call(slot, payload) -> Promise<reply>,
onFrame(fn) -> off, onDialog(fn) -> off }` with parsed values (JSON stays in
this module). `connectBridge()` resolves the QWebChannel bridge, or `null`
outside Qt. `createFakeBridge({describe, values, actions})` has the same
interface plus `calls` (`[slot, payload]`), `values`, `transport`,
`dialogAnswers` (paths the next dialogs return) and `pushFrame(extra)` (emit
the changes and events since the last frame); `actions[verb](args, fake)`
returns a verb's result.

**core-client.js** - `createCoreClient(bridge, {schedule})`:
`get(addresses) -> Promise<{addr: value}>`; `set(addr, value, {burst,
editAll})` (coalesced per animation frame, latest value per address, one
`set` call); `flush() -> Promise<errors>`; `act(verb, args) -> Promise<event>`
(`{status: 'done', result}` or `{status: 'error', error}`);
`describe(prefix | list) -> Promise<{addr: meta}>`; `beginGesture()`,
`endGesture()`; `openFile(opts)`, `saveFile(opts)`, `chooseFolder(opts) ->
Promise<path | null>`; `onFrame(fn) -> off`. It sends `resync` on creation.

**store.js** - `createStore()` (pure, no DOM): `seed({addr: value})`,
`apply(frame)`, `assume(addr, value)` (local echo), `value(addr)`, `has(addr)`,
`watch(addr, fn(value, addr), {now = true}) -> off` (called only when the value
changes), `watched()` (addresses with live watchers), `readout(name)`,
`watchReadout(name, fn, {now = true}) -> off` for `transport`, `modulation`,
`audio`, `midi`; `onEvent(fn) -> off`; `version`.

Boot (`main.js`): connect, client, store (`client.onFrame(store.apply)`),
`mountStage`, `describe(PANEL_ADDRESSES)`, `mountPanel`, then
`store.seed(await client.get(store.watched()))`.

**Controls declare what they edit**: `data-address="global.tempo"` on the
element of a control bound to an address, `data-verb="transport.toggle"` on
one that starts a verb. The parity guard reads them.

## Tests

All from pytest, offscreen, no Node and no display needed:

```bash
python -m pytest tests/web -q                      # the web layer (~7 s)
node --test pythonic/web/static/test/*.test.js     # optional local shortcut for pure specs
```

`tests/web/conftest.py` sets `QT_QPA_PLATFORM=offscreen` (unless set),
renders Chromium in software (`prepare_headless`) and registers the scheme
before pytest-qt makes the QApplication. CI also sets
`QTWEBENGINE_DISABLE_SANDBOX=1`. To watch a test, run it with
`QT_QPA_PLATFORM=xcb`.

- **JS specs** (`test_js_specs.py`): every `static/test/*.test.js` and
  `*.engine-spec.js` runs inside QtWebEngine via `test/runner.html?spec=...`
  (an import map maps `node:test` / `node:assert/strict` to small shims), one
  pytest test per file. Write specs in `node:test` style: `test(name, fn)`,
  `assert.equal/deepEqual/ok/throws/rejects`.
- **Page tests** (the main layer): fixtures `panel` (the real page over a
  `FakeCore`), `open_panel(core, owns_core=True)`, `fake_core`, `core_table`
  (session: `describe()` and values of every address of a real core),
  `real_core` (a started `AppCore` on `FakeAudioBackend`; `real_core.backend
  .stream.pull()` runs one audio callback). Windows are disposed at teardown
  and the test fails on JS console errors.
- **`Page`** (`tests/web/page.py`): `js(expr)` (value via JSON text),
  `run(statements)`, `wait_js(expr, timeout, pump)`, `wait_ready()`,
  `rect(sel)`, `center(sel)`, `click(sel)`, `wheel(sel, steps)` (QTest input on
  the view's focus proxy), `pixel(x, y)`, `color_at(sel, fx, fy)`,
  `close_to(color, expected, tolerance)`, `wait_pixels(check)` (coarse
  `grab()` checks, no screenshot diffs; painting lags the DOM, so wait for
  pixels instead of sleeping), `answer_dialog(path | None)`, `tick()`,
  `bound_addresses()`, `console_errors()`; `choose_in_dialog(qtbot, dialog,
  path)`, `same_path(a, b)` (Qt answers paths with `/` on Windows).
- **`FakeCore`** (`tests/web/fake_core.py`): get/set/describe/act/poll from
  the real metadata, sets coerced as the real core does and reported by the
  next poll; `calls` (`('set', addr, value, opts)`, `('act', verb, args)`,
  `('gesture', phase)`), `sets()`, `verbs_called()`, `post_change(addr, v)`
  (a change from the core's side), `transport`, `readouts`, `verbs` (verb ->
  handler), `closed`.
- **Real-core end-to-end tests** (`test_real_core.py`): play and the playhead,
  a tempo edit through a preset file round trip, the frame budget.
- **Parity guard** (`test_parity_guard.py`): every address the core's
  `describe()` lists is bound by a `data-address` control or listed in
  `tests/web/parity_absent.json`: `{"absent": {"<fnmatch glob>": "<reason>"}}`.
  It also fails on a bound address still listed and on globs that match
  nothing. A slice that binds controls deletes their entries; a core change
  that adds addresses lists them there (or binds them).

Each web slice adds page tests for the behaviours it builds and a manual
parity list in `docs/parity/`.
