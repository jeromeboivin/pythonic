# QtWebEngine ↔ Python bridge for real-time UI state

Research for [#2](https://github.com/jeromeboivin/pythonic/issues/2) (map: [#1](https://github.com/jeromeboivin/pythonic/issues/1)).
Date: 2026-10-04. Versions: PySide6 6.11.2 (Qt 6.11.2, Chromium 140, V8 14.0), Python 3.12.3.

## Answer

- **Channel: QWebChannel over QtWebEngine's built-in IPC transport** (`page.setWebChannel(channel)`).
  - Python → JS: use **plain signals**, not `Q_PROPERTY` notify updates. Notify updates are batched every 50 ms by default and gated by a client "idle" acknowledgement, which caps them at about 20 Hz.
  - JS → Python: use slots. JS gets a Promise back when a slot returns a value.
- **Don't stream with `runJavaScript`.** Use it only for one-off bootstrap or debug calls.
- **A local WebSocket is not worth it here.** It is no faster in-process, and it opens a TCP port that any local process or browser tab can reach.
- **Push a single coalesced "frame" message from a Qt main-thread timer at 30–60 Hz.** The audio callback never touches Qt. It only writes a small snapshot that the timer reads (pull model), as the tkinter GUI already does today with `current_play_position` and its 50 ms `after()` tick.
- **Serve the front-end from a custom URL scheme** (for example `app://ui/`). Register it before `QApplication` exists. Its `QWebEngineUrlSchemeHandler` returns files from the package directory with correct MIME types. ES modules, `fetch()` and a real origin all work, with no build step.
- **Dev tooling:** set `QTWEBENGINE_REMOTE_DEBUGGING=<port>` and open `http://127.0.0.1:<port>` in any Chromium-based browser. For an in-app inspector, use `QWebEnginePage.setDevToolsPage()`.
- **Distribution:** QtWebEngine ships in the `PySide6-Addons` wheel, which is abi3 (`cp310-abi3`, Python 3.10 to 3.14). Wheels exist for Linux x86_64/aarch64 (glibc ≥ 2.34), Windows x64 and macOS universal2 (≥ 13). **Windows ARM64 wheels have no QtWebEngine.** A full `pip install PySide6` is about 650 MB on disk on Linux x86_64, and about 270 MB of that is QtWebEngine.

## 1. Channel comparison

### How each channel works (from source and docs)

- **QWebChannel** publishes QObjects' properties, signals and slots to JS, and all communication is asynchronous ([Qt WebChannel JS API](https://doc.qt.io/qt-6/qtwebchannel-javascript.html)).
  - **Transport.** In QtWebEngine, `QWebEnginePage::setWebChannel` "connects it to web engine's transport using Chromium IPC messages". The transport is exposed in JS as `qt.webChannelTransport` ([QWebEnginePage](https://doc-snapshots.qt.io/qtwebengine/qwebenginepage.html)).
  - **Wire format.** Messages are JSON strings: `qwebchannel.js` calls `JSON.stringify` and `JSON.parse`, and the renderer transport hands UTF-8 JSON to Chromium IPC ([qwebchannel.js](https://code.qt.io/cgit/qt/qtwebchannel.git/tree/src/webchannel/qwebchannel.js?h=6.11), [web_channel_ipc_transport.cpp](https://code.qt.io/cgit/qt/qtwebengine.git/tree/src/core/renderer/web_channel_ipc_transport.cpp?h=6.11)).
  - **The client script ships with QtWebEngine.** `qrc:///qtwebchannel/qwebchannel.js` loads from any page (verified under `file:`, `qrc:` and `app:`). It is a classic script that defines a global `QWebChannel`, so load it with a plain `<script>` before the ES modules. Vendoring a copy is not required.
- **Signal vs property batching** is the key detail. In [`qmetaobjectpublisher.cpp`](https://code.qt.io/cgit/qt/qtwebchannel.git/tree/src/webchannel/qmetaobjectpublisher.cpp?h=6.11) (`signalEmitted`, `startPropertyUpdateTimer`, `sendEnqueuedPropertyUpdates`):
  - **Plain signals** (not a property's NOTIFY signal) are serialised and sent **immediately, once per emit**. There is no coalescing and no flow control, so a fast emitter can flood the page.
  - **Property NOTIFY changes** are collected into `pendingPropertyUpdates`. They are flushed when a timer of `propertyUpdateInterval` fires, which is **50 ms by default**. With `0` they flush once per event-loop pass, and with a negative value immediately.
  - A flush is also held back until the client has sent an `idle` message for the previous batch: `qwebchannel.js` sends `idle` after it has applied each property update. This makes property updates self-throttling and coalescing ("latest value wins"), but at the default rate they are too slow for a playhead or meters.
  - `blockUpdates` pauses property updates entirely ([QWebChannel](https://doc.qt.io/qt-6/qwebchannel.html)).
- **`runJavaScript`**:
  - It is asynchronous and runs the script "without checking whether the DOM of the page has been constructed".
  - Only plain data can come back: JSON types, `Date`, `ArrayBuffer`, but no `Promise` and no `Function`.
  - The callback "is always called, but it might be done during page destruction".
  - Qt warns not to run lengthy routines in the callback ([QWebEnginePage](https://doc-snapshots.qt.io/qtwebengine/qwebenginepage.html)).
  - Every push is a script string that has to be built in Python and compiled and evaluated in V8.
- **Local WebSocket** (`QWebSocketServer` from `PySide6.QtWebSockets`, or a Python library):
  - Messages go over a real loopback TCP socket.
  - The page can reach `ws://127.0.0.1` even when it is served from a secure custom scheme (verified).
  - The cost is an open port that any local process or web page can connect to, so it needs an auth token or an origin check.
  - Its one real benefit, running the front-end in a normal browser tab, is out of scope on the map.

### Measured (bench in [`qtwebengine-bridge-bench/`](qtwebengine-bridge-bench/))

Setup: Ubuntu 24.04.5, kernel 6.8, X11, 8 cores, PySide6 6.11.2, Python 3.12.3. The view was visible, the page was served from `app://`, and the payload was `{step, meters[8]}`. The table shows two runs (run 1 / run 2).
- **RTT** means the push plus the JS handler plus an ack coming back over the same channel.
- **"audio-sim"** is a Python thread that wakes every 128 frames at 44.1 kHz (2.9 ms) and holds the GIL for 0.3 ms. It records how late it wakes up, which is a stand-in for a sounddevice callback that needs the GIL.

| Test | QWebChannel | runJavaScript | WebSocket |
|---|---|---|---|
| Py→JS at 60 Hz, RTT median / p99 | 0.84 / 4.2 ms; 0.65 / 2.9 ms | 1.18 / 4.1 ms; 1.58 / 2.3 ms | 0.90 / 3.1 ms; 1.67 / 3.0 ms |
| Py→JS burst of 2000, all acked after | 190 ms; 271 ms (7–10 k msg/s) | 289 ms; 165 ms (7–12 k msg/s) | 163 ms; 180 ms (11–12 k msg/s) |
| JS→Py flood of 2000 fire-and-forget, received after | 142 ms; 75 ms (14–26 k msg/s) | n/a | 58 ms; 67 ms (30–34 k msg/s) |
| JS awaited call RTT, median / p99 (slot with return value vs WS echo) | 0.2 / 1.8 ms; 0.6 / 1.4 ms | n/a | 0.3 / 2.7 ms; 0.7 / 1.9 ms |

- **All three channels are far faster than the UI needs.** At 60 Hz the round trip is under about 2 ms, and capacity is thousands of messages per second. Knob drags (60–120 events/s) and 60 Hz frames use well under 1 % of it, so performance does not decide the choice.
- **NOTIFY property set 1000× over 1 s:** with the default 50 ms interval, JS saw **21–22 updates (about 20 Hz)**. With `propertyUpdateInterval=0` it saw 100, which is one per event-loop pass of the test.
- **Mixed load for 5 s** (60 Hz signal push plus a 200 Hz `setParam` stream from JS):
  - Push RTT was median 0.6 ms, p99 2.2–4.4 ms.
  - The audio-sim wake-up delay stayed at idle levels (p99 0.35–1.3 ms, against an idle p99 of 0.3–0.8 ms).
- **GIL signal:** during the 2000-message *bursts*, the audio-sim's worst wake-up delay rose to about **5–10 ms**. That matches `sys.getswitchinterval()` = 0.005 s, the GIL hand-off timeslice ([sys docs](https://docs.python.org/3.12/library/sys.html#sys.setswitchinterval)). A long run of Python work on the Qt main thread can therefore delay a GIL-needing audio callback by a whole buffer period. The fix is to keep each main-thread tick short (one coalesced message) and never burst.

### Batching patterns

- **Python → JS:**
  - A single `QTimer` on the main thread (30–60 Hz; `Qt.PreciseTimer` if needed) builds **one frame dict** each tick: playhead step, meter levels, and any parameter values that changed since the last tick (a dirty set filled by MIDI CC or morph).
  - It emits that dict with one signal, `frame = Signal("QVariantMap")`; a dict maps to a JS object (verified).
  - JS stores the latest frame and applies it in `requestAnimationFrame`.
  - Discrete, rare events (preset loaded, pattern changed) can be separate signals.
- **JS → Python:**
  - Coalesce knob or fader input per animation frame (latest value wins per parameter), then call a slot such as `setParams({id: value, ...})`.
  - Send gesture `begin` and `end` so the core can group undo steps.
  - Fire-and-forget calls need no Promise handling, and order is preserved over the single transport.
- **Don't use NOTIFY properties for fast state.** If properties are used for slow state, the 50 ms batching and idle gating are actually useful.

## 2. Threading: audio callback → Qt main thread → page

- **sounddevice/PortAudio rules.** The stream callback "runs at very high or real-time priority". Do "not allocate memory, access the file system, call library functions or call other functions … that may block" ([sounddevice streams](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)).
- **Qt rules.** GUI classes "can only be used from the main thread". A queued connection runs the slot "when control returns to the event loop of the receiver's thread". `QCoreApplication::postEvent()` is thread-safe ([Threads and QObjects](https://doc.qt.io/qt-6/threads-qobject.html)).
- **Recommended pattern (pull, not push):**
  1. The audio callback writes plain values into preallocated slots, such as an int for the play position and a numpy array for meters. It never emits Qt signals and never calls the bridge. A torn read of a meter value only matters for one frame, and the existing `position_lock` style is fine for anything that must be consistent.
  2. A main-thread `QTimer` (the frame tick above) reads that snapshot and emits the frame. This is what the tkinter GUI already does: `_audio_callback` stores `current_play_position` under `position_lock`, and `_ui_update_tick` polls every 50 ms.
  3. MIDI input threads (today: `root.after(0, …)`) become a queued signal emit to a main-thread QObject, or they push into a `queue.SimpleQueue` that the frame tick drains. Either way, the core state changes on the main thread and the next frame carries it to JS.
- **The bridge QObject and the QWebChannel stay on the main thread with the page.** Slot bodies run on the main thread, so heavy work (exports, AI generation) moves to worker threads and reports back through queued signals.
- **The GIL is the real risk, not the transport.** Measured above: short main-thread ticks leave the audio thread unaffected, but bursts of Python work add up to one switch interval (5 ms) of delay. Keep per-tick Python work small and bounded. Whether the numba render loop releases the GIL (`nogil=True`) is a separate concern, tracked under the map's performance-budget item.

## 3. Loading the front-end

Tested with an `index.html` that loads `qrc:///qtwebchannel/qwebchannel.js`, then `<script type="module" src="./main.js">`, which imports `./mod.js` and `fetch('./data.json')`:

| Scheme | ES module import | `fetch()` relative file | Origin | Notes |
|---|---|---|---|---|
| `file://` (`QUrl.fromLocalFile`) | works | works | `file://` | Relies on `LocalContentCanAccessFileUrls`, which is on by default ([QWebEngineSettings](https://doc-snapshots.qt.io/qtwebengine/qwebenginesettings.html)). Opaque-ish origin; path is on disk. |
| `qrc:///` (pyside6-rcc) | works | **fails**: `URL scheme "qrc" is not supported` | `qrc:` | Needs an `rcc` compile step, which conflicts with "no build step". |
| `app://ui/` custom scheme | works | works | `app://ui` | Recommended. |

- **Custom scheme recipe** ([QWebEngineUrlScheme](https://doc-snapshots.qt.io/qtwebengine/qwebengineurlscheme.html), [QWebEngineUrlSchemeHandler](https://doc-snapshots.qt.io/qtwebengine/qwebengineurlschemehandler.html)):
  - **Register before the app exists.** Call `QWebEngineUrlScheme.registerScheme()` "before a QGuiApplication or QApplication instance is created". Use `Syntax.Host` and the flags `SecureScheme | LocalAccessAllowed | CorsEnabled | FetchApiAllowed`.
  - **Install the handler** with `QWebEngineProfile.installUrlSchemeHandler(b"app", handler)`. The profile does not take ownership, so keep a reference.
  - **Reply** from `requestStarted(job)` with `job.reply(mime, QBuffer)`. Guard against path traversal, and `job.fail(UrlNotFound)` otherwise.
- **Module MIME pitfall.** Module scripts get strict MIME checking. A Qt maintainer's diagnosis on [QTBUG-77282](https://bugreports.qt.io/browse/QTBUG-77282) is that modules "must come with a Content-Type: text/javascript header". The reported error otherwise is "non-JavaScript MIME type of ''". Map `.js`/`.mjs` → `text/javascript` explicitly; don't rely on `mimetypes`, which varies by OS and registry on Windows.
- **Old qrc HTML regression.** [QTBUG-97392](https://bugreports.qt.io/browse/QTBUG-97392) ("WebEngine fails to load urls from resources") was fixed in 5.15.11 / 6.5.0 and does not affect 6.11.
- **PySide pitfall seen while testing:** `QTimer.singleShot(ms, obj.method)` on a plain-Python object that is not referenced anywhere silently never fired. Keep references to anything whose bound methods are used as callbacks.

## 4. Dev tooling

- **Remote DevTools** ([Qt WebEngine debugging](https://doc-snapshots.qt.io/qtwebengine/qtwebengine-debugging.html)): `QTWEBENGINE_REMOTE_DEBUGGING=9222` (or `--webEngineArgs --remote-debugging-port=9222`), then browse to `http://127.0.0.1:9222`. Verified:
  - `/json/list` listed `app://ui/index.html` with a `webSocketDebuggerUrl`.
  - Qt 6.11 adds `--remote-allow-origins=*` automatically, which the docs otherwise tell you to add.
- **In-app inspector:** a second `QWebEngineView` whose page is attached with `page.setDevToolsPage(devtools_page)`. The DevTools resources ship in the wheel (`qtwebengine_devtools_resources.pak`, 12 MB).
- **Console forwarding:** subclass `QWebEnginePage.javaScriptConsoleMessage` to route JS console output to Python logging. By default, only warnings and above go to Qt's `js` logging category.
- **Chromium flags for diagnosis:**
  - `QTWEBENGINE_CHROMIUM_FLAGS` sets them, for example `--disable-gpu` for GL problems, `--single-process` for crash stacks, or `--enable-logging --log-level=0`.
  - `chrome://sandbox` and `chrome://gpu` load inside the view.

## 5. Distribution

From PyPI JSON for 6.11.2 (2026-08-18), plus wheel contents read from each wheel's zip directory:

- **`PySide6`** is a 0.6 MB metapackage that requires `shiboken6`, `PySide6_Essentials` and `PySide6_Addons` at the same version. `requires_python` is `>=3.10,<3.15`, and every wheel is `cp310-abi3`, so one wheel per platform covers 3.12.
- **QtWebEngine and QtWebChannel are in `PySide6-Addons`**, not Essentials. A minimal dependency set is `PySide6-Essentials` plus `PySide6-Addons`, which is effectively all of `PySide6`.

| Platform tag | Addons wheel (download) | WebEngine files (unpacked) | Essentials wheel |
|---|---|---|---|
| `manylinux_2_34_x86_64` | 175 MB | 267 MB (`libQt6WebEngineCore.so.6` 204 MB) | 80 MB |
| `manylinux_2_39_aarch64` | 171 MB | 254 MB | 80 MB |
| `win_amd64` | 168 MB | 346 MB (includes a 76 MB `.debug.pak`) | 77 MB |
| `macosx_13_0_universal2` | 332 MB | 625 MB (universal binary, 474 MB core) | 111 MB |
| `win_arm64` | 36 MB | **none** (only QtWebChannel) | 58 MB |

- **Measured install:** `pip install PySide6==6.11.2` into a venv on Linux x86_64 gave `site-packages/PySide6` = **648 MB**.
- **Linux baseline:** glibc ≥ 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+). macOS needs 13+.
- **Linux sandbox** ([platform notes](https://doc-snapshots.qt.io/qtwebengine/qtwebengine-platform-notes.html)):
  - The Chromium sandbox needs seccomp-bpf and unprivileged user namespaces.
  - Ubuntu's AppArmor can restrict those (`/proc/sys/kernel/apparmor_restrict_unprivileged_userns`).
  - The escape hatch is `QTWEBENGINE_DISABLE_SANDBOX=1` or `--no-sandbox`, which is a security trade-off.
  - Observed on this machine: Ubuntu 24.04 with `apparmor_restrict_unprivileged_userns=1`, running the pip wheel from a venv. `chrome://sandbox` reported "You are adequately sandboxed" (namespace + seccomp-BPF), so no flag was needed.
  - Recommendation: don't ship `--no-sandbox` by default. Document the env var as a fallback for locked-down distros, containers and root.
- **GPU:** no flags were needed on this X11 machine. If a custom OpenGL `QSurfaceFormat` is ever set, it must happen before `QApplication` exists; on macOS, setting it later is fatal (platform notes). `--disable-gpu` is the standard workaround for broken GL drivers, which matters little for a 2D panel.
- **Process model:** QtWebEngine starts a separate `QtWebEngineProcess` helper (shipped in the wheel under `PySide6/Qt/libexec`). Freezing tools must bundle it, which belongs to the packaging ticket.

## Caveats

- Numbers come from one Linux/X11 machine with a synthetic GIL-holding thread, not the real numba render loop. They show relative behaviour and orders of magnitude, not a guarantee. Re-run the bench on Windows and macOS during the platform smoke checks.
- Wheel sizes are for 6.11.2. Qt minor releases move by tens of MB.
- `file://` loading works today only because of a default-on setting. The custom scheme removes that dependency and gives a stable origin, for example for `localStorage` keyed per origin.
- Keep the JS side's transport behind one small module (`bridge.js`: `send(params)`, `onFrame(cb)`). Swapping QWebChannel for a WebSocket later, for example for a test harness, then touches one file.
