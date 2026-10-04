# Testing the HTML front-end inside QtWebEngine

Research for [#17](https://github.com/jeromeboivin/pythonic/issues/17) (map: [#1](https://github.com/jeromeboivin/pythonic/issues/1)). Builds on [`qtwebengine-bridge.md`](https://github.com/jeromeboivin/pythonic/blob/research/qtwebengine-bridge/docs/research/qtwebengine-bridge.md) (#2).
Date: 2026-10-04. Versions: PySide6 6.11.2 (Qt 6.11.2, Chromium 140, V8 14.0), pytest-qt 4.5.0, pytest 9.1, Playwright (Python) 1.63.0, puppeteer-core 24.43.1, Node 20.19.0, Python 3.12.3, Ubuntu 24.04, kernel 6.8.

Every "verified" claim below was run with the spike in [`frontend-testing-spike/`](frontend-testing-spike/), with `DISPLAY`/`WAYLAND_DISPLAY` unset and `QT_QPA_PLATFORM=offscreen`. That is the same situation as a Linux CI runner with no X server.

## Answer

- **Everything that matters runs headless on Linux with `QT_QPA_PLATFORM=offscreen`. No xvfb is needed.** This covers loading `app://` pages, ES modules, Web Components, QWebChannel in both directions, `runJavaScript`, QTest mouse input into the page, `QWidget.grab()` pixels, and the remote-debugging port with raw CDP and Puppeteer. Chromium falls back to software rendering, which is fine for a 2D panel (verified).
- **Recommended stack: three layers, all started from pytest (one command, `./run_tests.sh`).**
  1. **JS unit tests with no build step.** Write pure-logic modules (value scaling, drag maths, step-lane state, frame diffing) as plain ES modules with no DOM or Qt imports. Write their tests against the `node:test` + `node:assert/strict` API.
     - They run under `node --test` (zero npm dependencies, Node ≥ 20) for a ~0.2 s edit loop.
     - The same files also run inside QtWebEngine: a test page maps `node:test`/`node:assert/strict` to two ~10-line shims with an **import map** (verified).
     - Component tests that need `customElements` and a real DOM run only in the engine, under a name `node --test` does not match (`*.engine-spec.js`).
     - CI therefore needs **no Node at all**: pytest loads the in-engine test page and collects the results.
  2. **Page + bridge integration in-process with pytest-qt.** Load the real page from `app://` in a `QWebEngineView`, with the **real bridge QObject over a fake core**. A fake core is an in-memory object with the same `get`/`set`/`describe`/`act`/`poll` interface as the app core (ADR 0001). The test then:
     - drives the page with **QTest mouse events on `view.focusProxy()`** (real pointer events, verified) or with `runJavaScript`;
     - asserts on the fake core's recorded calls;
     - pushes frames and reads the DOM back.

     This takes about 0.3 s per test, in the same process as the rest of the Python tests.
  3. **Optional: drive the running app from outside over CDP** for anything layer 2 can't do: screenshots of the composited page, tracing, a long-running real app. Use **raw CDP** (about 60 lines of Python with `websockets`) or **Puppeteer `connect`**.
     - **Playwright's `connect_over_cdp` does not work against QtWebEngine.** It fails on connect with `Browser.setDownloadBehavior: Browser context management is not supported` (verified). Upstream closed this as "not planned" ([playwright#36961](https://github.com/microsoft/playwright/issues/36961)).
- **Don't add a JS toolchain** (Web Test Runner, Vitest, jsdom) for testing. Each one brings `node_modules` and tests a different engine or a fake DOM. The in-engine page already tests the exact Chromium that ships.
- **Pitfalls found (all verified):**
  - **`runJavaScript` returns only scalars in PySide6 6.11.2.** Arrays, objects and Promises arrive as `''`, so return `JSON.stringify(...)` and parse in Python.
  - **Install the `app://` scheme handler once per profile, not per view or test.** A second install is refused, and the first handler dies with its view, so later loads hang.
  - **`qtbot.waitUntil` callbacks must return a `bool`/`None`,** not a truthy value.
- **Cost:** the main cost is the PySide6 dependency itself (about 650 MB installed, about 1.5 min to download here). It is not yet in `requirements.txt`, and the new GUI needs it anyway. Everything else is either zero-dependency (`node --test`, the shims, raw CDP) or small (`pytest-qt`, `websockets`; `puppeteer-core` is 54 MB if wanted).

## 1. Unit-testing ES modules without a build step

### Node's built-in test runner

- `node:test` is **stable since Node 20.0.0**. `node --test` discovers `**/*.test.{cjs,mjs,js}`, `**/*-test.*`, `**/*_test.*`, `**/test-*.*`, `**/test.*` and `**/test/**/*.*`, and it runs ES modules ([Node test runner](https://nodejs.org/api/test.html)). It also has `mock.fn`, timer mocking and `mock.module()` (same page).
- **Verified:** `node --test ui/` ran `model.test.js`, which imports `./model.js`, with no `package.json` at all: 3 tests in 0.11–0.18 s wall. Node 20.19 detected the ES module syntax on its own. A `package.json` with `"type": "module"` makes that explicit on older Node versions and has no dependencies.
- **Limit:** Node has no DOM, so no `HTMLElement`, `customElements`, layout or pointer events. A Web Component class can't even be defined there. Adding jsdom or happy-dom would mean an npm dependency and a fake DOM that differs from Chromium 140. That makes Node the right tool only for **pure modules**. Keep that logic out of the components on purpose (the "humble view" split): components forward events and render state, and modules decide.

### Running the same tests inside QtWebEngine (no Node needed)

- **Import maps** let a page remap module specifiers ([HTML spec, import maps](https://html.spec.whatwg.org/multipage/webappapis.html#import-maps)). In the spike, `ui/test.html` maps `node:test` → `./shim/node-test.js` and `node:assert/strict` → `./shim/node-assert.js`. It then imports the spec modules, runs what they registered, and leaves `window.__results = [{name, ok, error}]`.
- **Verified:** the unchanged `model.test.js` (3 tests) and the DOM-only `knob.engine-spec.js` (2 tests, one failing on purpose) all ran in the engine. The failure came back with a real stack (`at app://ui/knob.engine-spec.js:14:53`). The pytest side is about 15 lines: load `app://ui/test.html`, wait for `window.__results`, read it as JSON, and assert. It took 0.27 s.
- This also makes JS failures show up in the normal pytest report. It is easy to emit one pytest item per JS test with `pytest_generate_tests` or a small collector plugin.
- **Off-the-shelf alternatives** (not needed):
  - Mocha's browser build and Chai can be vendored as two plain files.
  - [Web Test Runner](https://modern-web.dev/docs/test-runner/overview/) is npm-installed and drives *its own* browser via Puppeteer, Playwright or Selenium, so it would not test QtWebEngine.

## 2. Driving the real page inside QtWebEngine

### 2a. In-process with pytest-qt (recommended default)

- **pytest-qt basics:**
  - `qtbot.waitSignal` blocks until a signal fires.
  - `qtbot.waitUntil` polls a callback, which must return `True`/`False`/`None` or raise `AssertionError` ([pytest-qt reference](https://pytest-qt.readthedocs.io/en/latest/reference.html)).
  - `qtbot.waitCallback()` (pytest-qt ≥ 3.1) returns a callable that blocks the test until it is called, which fits `runJavaScript`'s result callback exactly (pytest-qt source, `qtbot.py`).
- **`runJavaScript(src, worldId, callback)`:**
  - Qt documents that "only plain data can be returned", including "all of the JSON data types", but not `Function` or `Promise` ([QWebEnginePage](https://doc-snapshots.qt.io/qtwebengine/qwebenginepage.html)).
  - **Observed in PySide6 6.11.2:** numbers (as `float`), strings and booleans come back, but `[1,2]`, `({a:1})`, `[{a:1}]` and `Promise.resolve(1)` all arrive as `''` (`tests/test_rjs_types.py`). This is probably a binding-conversion gap; it was not checked with PyQt6. Rule: **always return `JSON.stringify(...)`** for anything structured.
- **Real input without CDP.** `QTest.mousePress/mouseMove/mouseRelease` sent to `view.focusProxy()` (the Chromium render widget, a plain `QWidget`) produced `pointerdown:80, pointermove:30, pointerup:30, click:30` in the page, offscreen (`tests/test_qtest_input.py`). That is enough to test the knob and fader vertical drag (#9) and pad painting (#8) through the real event path, without calling component methods. Wheel input was not tried: QTest has no wheel helper, so it would need a hand-built `QWheelEvent` sent with `QApplication.sendEvent`.
- **Pixels.** `view.grab()` returned the page's real pixels offscreen: a solid `rgb(0,200,0)` page read back exactly (`tests/test_in_engine.py`). That makes coarse visual checks possible, such as "the LED is lit" or "the fader's lit segment moved". Full screenshot diffs are possible too, but software rendering and fonts make them fragile across machines. Keep them out of the default run.
- **Measured:**
  - The full spike (6 tests + 1 expected failure, including two subprocess app launches) took **4.0–4.3 s**.
  - One in-process page test: **0.25–0.35 s**.
  - The whole pytest process for one page test: **0.84 s wall, 239 MB max RSS**.
- **Order of initialisation:** `QWebEngineUrlScheme.registerScheme()` must run before the `QApplication` exists. In the spike, `conftest.py` calls it at import time, before pytest-qt's `qapp` fixture creates the app. The scheme *handler* must be installed **once per profile** and kept alive for the whole session. Installing it per test made every later load hang (`loadFinished` never fired), because Qt keeps the first, now-deleted handler.

### 2b. From outside over the remote-debugging port (CDP)

- **Enabling it:** set `QTWEBENGINE_REMOTE_DEBUGGING=<port>` or `--remote-debugging-port=<port>`. Add `--remote-allow-origins` "to avoid WebSocket errors" ([Qt WebEngine debugging](https://doc-snapshots.qt.io/qtwebengine/qtwebengine-debugging.html)); Qt 6.11 already adds `*` (#2). `/json/version` reported `Protocol-Version: 1.3`, `QtWebEngine/6.11.2 Chrome/140`.
  - The app runs as a **subprocess** of the test. The test then finds the `app://` target in `/json/list`, which took **0.35 s** from launch.
  - It has to be a separate process because QtWebEngine runs Chromium's browser side on the Qt main thread. A test that blocks that thread can't also be the CDP client.
- **Raw CDP (verified, `tests/test_rawcdp.py`):** a ~30-line client over `websockets` connects to the page's `webSocketDebuggerUrl` and works with no other dependency. It used:
  - `Runtime.evaluate` (`returnByValue`, `awaitPromise`), which returns structured values, unlike `runJavaScript`;
  - `Input.dispatchMouseEvent`;
  - `Page.captureScreenshot`.

  The test took 0.83 s including app start ([CDP Runtime](https://chromedevtools.github.io/devtools-protocol/tot/Runtime/), [Input](https://chromedevtools.github.io/devtools-protocol/tot/Input/), [Page](https://chromedevtools.github.io/devtools-protocol/tot/Page/)).
- **Puppeteer (verified, `puppeteer_connect.mjs`):** `puppeteer.connect({browserURL, defaultViewport: null})` listed the `app://` page, waited for a function, clicked by bounding box, evaluated, read text and took a screenshot ([ConnectOptions](https://pptr.dev/api/puppeteer.connectoptions)). The fake core received the resulting `set`. It costs Node plus `puppeteer-core` (54 MB `node_modules`, no browser download).
- **Playwright (verified failure, `tests/test_cdp.py`, marked `xfail(strict=True)`):** `chromium.connect_over_cdp("http://127.0.0.1:<port>")` fails during connect with `Protocol error (Browser.setDownloadBehavior): Browser context management is not supported`.
  - Playwright documents this connection as "significantly lower fidelity" than its own protocol ([BrowserType.connect_over_cdp](https://playwright.dev/python/docs/api/class-browsertype)).
  - The identical QtWebEngine report [playwright#36961](https://github.com/microsoft/playwright/issues/36961) was closed as not planned, and CEF hits the same error ([playwright#10927](https://github.com/microsoft/playwright/issues/10927)).
  - **Don't build on Playwright here.**
- **When CDP earns its cost:** screenshots of the composited page, performance traces (frame budget, #10), console and exception capture, and debugging a live app session. For functional assertions, layer 2a is faster, in-process, and needs no port.

## 3. A fake core behind QWebChannel

- **What to fake:** the **core**, not the bridge. The bridge QObject (a frame signal pushed by a main-thread `QTimer`, plus `set`/`act`/gesture slots) is the code under test, together with the JS. Behind it, put a `FakeCore` that implements the ADR 0001 interface in memory:
  - `get`/`set` on a dict of addresses;
  - `describe()` from a fixture of metadata;
  - `act()` that records the verb and posts a scripted result;
  - `poll(since)` returning queued changes.

  The spike's `FakeCore` is the minimal version: one `set` slot that records `(address, value)` and one `frame` signal (`app.py`).
- **Verified round trip** (`tests/test_qt_page.py`): a knob drag in the page reached `core.calls == [('ch1.tune', 24.0)]`, and `core.frame.emit({'step': 7})` showed up in the DOM. This runs through the real QWebChannel IPC transport and `qrc:///qtwebchannel/qwebchannel.js`.
- **Why the fake core and not the real one:** the real core starts audio I/O, the numba render loop, MIDI and preferences. In CI there is no sound card and no MIDI, and the tests become timing-dependent. The real core is already tested through its interface in plain pytest (slicing plan, #6). Bridge and page tests only need to show that the page and the bridge speak that interface correctly.
  - **One shared contract test suite** runs against both `FakeCore` and the real core (with audio disabled or driven by a manual block clock). It keeps the fake honest.
  - A few end-to-end smoke tests can use the real core once it can run without an audio device.
- **JS side:** keep the transport behind one module (`bridge.js`: `send`, `onFrame`), as #2 recommends. Engine-spec tests of components can then pass a fake `bridge` object directly, with no QWebChannel at all. That gives three levels:
  - component + fake bridge.js (JS only);
  - page + real QWebChannel + fake core;
  - optionally, the real app over CDP.

## 4. What runs headless on Linux CI

- **Platform plugin:** `-platform`/`QT_QPA_PLATFORM` picks the QPA plugin; Qt lists `offscreen` and `minimal` (the latter "to run GUI applications in environments without a GUI") ([QGuiApplication](https://doc.qt.io/qt-6/qguiapplication.html)). The PySide6 wheel ships `libqoffscreen.so`.
  - **Verified with no `DISPLAY`:** every spike test passed under `offscreen`. QtWebEngine logged that GBM and Vulkan were unavailable and fell back to software rendering. That is harmless and needs no `--disable-gpu`.
  - pytest-qt's troubleshooting page still says it "needs a DISPLAY" and recommends xvfb / pytest-xvfb, listing the xcb libraries needed on `ubuntu-latest` ([pytest-qt troubleshooting](https://pytest-qt.readthedocs.io/en/latest/troubleshooting.html)). That applies to the `xcb` plugin. With `offscreen`, **neither xvfb nor the xcb libraries are needed**.
  - Keep xvfb (`xvfb-run` or `pytest-xvfb`) as the fallback only if something xcb-specific ever needs testing.
- **Sandbox:**
  - The Chromium sandbox needs seccomp-bpf and unprivileged user namespaces. Qt's switches to turn it off are `QTWEBENGINE_DISABLE_SANDBOX=1`, `--no-sandbox` or `QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox` ([platform notes](https://doc-snapshots.qt.io/qtwebengine/qtwebengine-platform-notes.html)).
  - Ubuntu 23.10+ restricts unprivileged user namespaces through AppArmor (`kernel.apparmor_restrict_unprivileged_userns=1`). Chromium's own docs give three fixes, from easiest to safest: the sysctl, a targeted AppArmor profile, or the SUID helper. `--no-sandbox` is the no-root fallback ([Chromium: AppArmor userns restrictions](https://chromium.googlesource.com/chromium/src/+/main/docs/security/apparmor-userns-restrictions.md)).
  - Containers running as root also need `--no-sandbox`.
  - This machine (Ubuntu 24.04, restriction on) ran the spike with and without `QTWEBENGINE_DISABLE_SANDBOX=1`, with the same result. **Recommendation for CI:** set `QTWEBENGINE_DISABLE_SANDBOX=1` in the test job only. The tests load only the repo's own `app://` files, so nothing untrusted runs. Never set it in the shipped app (#2).
- **System libraries:** even under `offscreen`, `libQt6WebEngineCore.so.6` links against system `libnss3`/`libnspr4`, `libasound2`, `libgbm1`, `libEGL`/`libGL`, `libfontconfig`, `libdbus-1`, `libxkbcommon`, `libxkbfile`, and X11 client libs (`libX11`, `libxcb`, `libXcomposite`, `libXdamage`, `libXrandr`, `libXtst`, …) (from `ldd`).
  - A slim Docker image needs these installed with `apt`.
  - A full desktop or a GitHub-hosted Ubuntu image very likely has them already. **Not verified on a hosted runner:** the repo has no CI workflow yet (`.github/` is absent).
- **Audio and MIDI:** the bridge and page tests never open a stream (fake core), so a runner with no sound card or MIDI is fine. `libasound` must be present to load the library, but no device is needed.
- **A minimal CI job** (sketch, unverified on GitHub Actions):

  ```yaml
  runs-on: ubuntu-24.04
  env: { QT_QPA_PLATFORM: offscreen, QTWEBENGINE_DISABLE_SANDBOX: "1" }
  steps:
    - uses: actions/checkout@v4
      with: { lfs: true }
    - uses: actions/setup-python@v5
      with: { python-version: "3.12", cache: pip }
    - run: pip install -r requirements.txt PySide6 pytest-qt
    - run: python -m pytest tests/ -q
  ```

  No Node step is needed, because the JS tests run in the engine. Add `actions/setup-node` only if the `node --test` fast loop should also gate CI. Cache pip: the PySide6 download is the slowest step.

## 5. Cost of each approach

Costs are measured on this machine where marked. "Setup" means code to write in this repo.

| Approach | Extra dependencies | Setup | Time per run | Headless Linux CI | Catches | Misses |
|---|---|---|---|---|---|---|
| `node --test` on pure modules | Node ≥ 20 (dev only) | none | 0.11–0.18 s for 3 tests (measured) | yes (no Qt at all) | logic bugs in scaling, step state, frame diffing | DOM, components, the real engine, bridge |
| Same specs + DOM specs **in the engine** (import-map shims) | PySide6 + pytest-qt | ~40 lines (test page, 2 shims, collector) | ~0.3 s per page load for all specs (measured) | yes, `offscreen` (verified) | the above + Web Components, events, layout in Chromium 140 | bridge, Python side |
| **pytest-qt page + real bridge + fake core** | PySide6 + pytest-qt | fake core (~100 lines for the ADR interface) + helpers (`js()`, scheme fixture) | 0.25–0.35 s per test; 0.84 s / 239 MB process (measured) | yes, `offscreen` (verified) | JS↔Python contract, frame handling, real pointer input via QTest, coarse pixels via `grab()` | the real core's timing, audio |
| Raw CDP against the running app | `websockets` | ~60 lines client + subprocess fixture | 0.35 s to first target, 0.83 s per test (measured) | yes (verified) | real app process, screenshots, traces, structured `Runtime.evaluate` | slower, port management, flakier waits |
| Puppeteer `connect` | Node + `puppeteer-core` (54 MB) | small; tests in JS | similar to raw CDP | yes (verified locally) | as raw CDP, nicer API | a second test language and toolchain |
| Playwright `connect_over_cdp` | Playwright | - | - | **fails on connect** (verified) | - | - |
| Web Test Runner / Vitest + jsdom | npm toolchain | config + `node_modules` | fast | yes | logic, approximate DOM | **not QtWebEngine**, no bridge; against the no-build rule |

The fixed cost behind every Qt row is **PySide6 in the test environment**:

- **Download and disk:** `pip install PySide6==6.11.2 pytest pytest-qt playwright websockets` into a fresh venv took **1 min 39 s** and **823 MB** here, of which PySide6 is about 650 MB (#2).
- **Not yet a repo dependency:** PySide6 is not in `requirements.txt`, and the repo venv has no PySide6. It becomes one when the new GUI lands.
- **Skipping without PySide6:** until then, a front-end test module should `pytest.importorskip("PySide6.QtWebEngineWidgets")`, so `./run_tests.sh` still passes on machines without it (the spike's `conftest.py` does this).

## Recommendation

1. **Split the JS into pure modules and thin components, and test the modules with `node:test`-style specs.** Run them in the engine through a test page with an import map, so `./run_tests.sh` runs them with no Node in CI. `node --test` remains a fast local loop.
2. **Make pytest-qt + the real bridge + a fake core the main front-end test layer.**
   - Ship reusable fixtures in the repo: scheme registration in `conftest.py`, one handler per session, a `page` fixture, `js()` built on `waitCallback`, and a QTest drag helper on `focusProxy()`.
   - Back them with one shared contract test suite that runs against both `FakeCore` and the real core.
3. **Run CI with `QT_QPA_PLATFORM=offscreen` and `QTWEBENGINE_DISABLE_SANDBOX=1`** in the test job, with no xvfb, and pip caching for PySide6.
4. **Keep CDP as a tool, not a test layer:** use raw CDP (or Puppeteer locally) for screenshots, traces and live debugging. Skip Playwright: it can't connect to QtWebEngine.
5. Keep screenshot comparisons out of the default run, or very coarse ("pixel at the LED is green"), because software rendering and fonts vary by machine.

## Caveats

- All runs used one Ubuntu 24.04 machine, not a hosted CI runner. The library list and the sandbox behaviour on GitHub's image are inferred, not observed.
- The `runJavaScript` empty-string result for arrays and objects was seen in PySide6 6.11.2 only. Re-check it when upgrading, since a fix would make `JSON.stringify` unnecessary (but harmless).
- The QTest input check covered mouse press, move and release only. Wheel, double-click and right-click (all used by #9) still need a check when the real controls exist.
- The spike's fake core implements only `set` and a frame signal. The full ADR 0001 interface (`describe`, `act` results through `poll`, gestures) is a build-ticket task.

## Running the spike

```sh
cd docs/research/frontend-testing-spike
pip install PySide6 pytest-qt websockets playwright   # playwright only for the xfail test
QT_QPA_PLATFORM=offscreen python -m pytest tests -q   # 6 passed, 1 xfailed
node --test ui/                                       # pure-module specs only
QT_QPA_PLATFORM=offscreen QTWEBENGINE_REMOTE_DEBUGGING=9339 python app.py &
npm i puppeteer-core && node puppeteer_connect.mjs
```
