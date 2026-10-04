"""Throwaway harness: module loading per scheme + bridge latency/throughput.

Throwaway research code (issue #2), not part of the app.

usage: python bench.py load file|qrc|app|<any url>   # which scheme can load ES modules / fetch()
       python bench.py bench                         # latency/throughput of the three channels
"qrc" mode first needs:  pyside6-rcc www.qrc -o www_rc.py
"""
import json, os, sys, time, threading, statistics

from PySide6.QtCore import (QObject, Signal, Slot, Property, QTimer, QUrl, QBuffer, QByteArray,
                            QIODevice, Qt)
from PySide6.QtWebEngineCore import (QWebEngineUrlScheme, QWebEngineUrlSchemeHandler, QWebEnginePage,
                                     QWebEngineProfile)

HERE = os.path.dirname(os.path.abspath(__file__))
WWW = os.path.join(HERE, "www")

scheme = QWebEngineUrlScheme(b"app")
scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
scheme.setFlags(QWebEngineUrlScheme.Flag.SecureScheme | QWebEngineUrlScheme.Flag.LocalAccessAllowed
                | QWebEngineUrlScheme.Flag.CorsEnabled | QWebEngineUrlScheme.Flag.FetchApiAllowed)
QWebEngineUrlScheme.registerScheme(scheme)

from PySide6.QtWidgets import QApplication
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel

TYPES = {".js": b"text/javascript", ".html": b"text/html", ".json": b"application/json", ".css": b"text/css"}


class Handler(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        path = os.path.normpath(os.path.join(WWW, job.requestUrl().path().lstrip("/")))
        if not path.startswith(WWW) or not os.path.isfile(path):
            job.fail(job.Error.UrlNotFound)
            return
        with open(path, "rb") as f:
            data = f.read()
        buf = QBuffer(parent=job)
        buf.setData(QByteArray(data))
        buf.open(QIODevice.ReadOnly)
        job.reply(TYPES.get(os.path.splitext(path)[1], b"application/octet-stream"), buf)


class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, msg, line, src):
        print(f"  [js console] {msg} ({src}:{line})")


class Bridge(QObject):
    tick = Signal(int, "QVariantMap")
    levelChanged = Signal()

    def __init__(self):
        super().__init__()
        self._level = 0.0
        self.acks = {}
        self.params = []
        self.reports = {}
        self.ready_state = None

    @Slot(int)
    def ack(self, seq):
        self.acks[seq] = time.perf_counter()

    @Slot(int, float)
    def setParam(self, idx, val):
        self.params.append(time.perf_counter())

    @Slot(int, float, result=float)
    def echo(self, i, v):
        return v

    @Slot(str, str)
    def report(self, kind, data):
        self.reports[kind] = json.loads(data)

    @Slot(str)
    def ready(self, what):
        self.ready_state = what

    def _get_level(self):
        return self._level

    def set_level(self, v):
        self._level = v
        self.levelChanged.emit()

    level = Property(float, _get_level, notify=levelChanged)


def payload(seq):
    return {"step": seq % 16, "meters": [0.1 * i + seq * 1e-4 for i in range(8)]}


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return "n=0"
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    return f"n={len(xs)} median={statistics.median(xs):.2f}ms p95={p(0.95):.2f}ms p99={p(0.99):.2f}ms max={xs[-1]:.2f}ms"


class Runner:
    """Drives a generator; yields float seconds or (predicate, timeout)."""

    def __init__(self, gen, app):
        self.gen, self.app = gen, app
        QTimer.singleShot(0, self.step)

    def step(self):
        try:
            y = next(self.gen)
        except StopIteration:
            self.app.quit()
            return
        if isinstance(y, (int, float)):
            QTimer.singleShot(int(y * 1000), self.step)
        else:
            pred, timeout = y
            deadline = time.perf_counter() + timeout

            def poll():
                if pred() or time.perf_counter() > deadline:
                    self.step()
                else:
                    QTimer.singleShot(1, poll)
            poll()


class AudioSim(threading.Thread):
    """Wakes every 128 frames @ 44.1 kHz, holds the GIL ~0.3 ms, records lateness."""

    PERIOD = 128 / 44100

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = False

    def run(self):
        nxt = time.perf_counter() + self.PERIOD
        while not self.stop:
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            now = time.perf_counter()
            self.samples.append((now, (now - nxt) * 1000))
            t_end = now + 0.0003
            x = 0
            while time.perf_counter() < t_end:
                x += 1
            nxt += self.PERIOD
            if nxt < now:
                nxt = now + self.PERIOD

    def window(self, t0, t1):
        return [l for t, l in self.samples if t0 <= t <= t1]


def run_load(mode):
    app = QApplication(sys.argv)
    handler = Handler()
    QWebEngineProfile.defaultProfile().installUrlSchemeHandler(b"app", handler)
    view = QWebEngineView()
    page = Page(view)
    view.setPage(page)
    if mode == "file":
        url = QUrl.fromLocalFile(os.path.join(WWW, "index.html"))
    elif mode == "qrc":
        import www_rc  # noqa: F401  (pyside6-rcc output)
        url = QUrl("qrc:///www/index.html")
    elif mode == "app":
        url = QUrl("app://ui/index.html")
    else:
        url = QUrl(mode)
    view.resize(400, 300)
    view.show()
    page.loadFinished.connect(lambda ok: print('  loadFinished', ok, flush=True))
    page.renderProcessTerminated.connect(lambda st, code: print('  render process terminated', st, code, flush=True))
    out = {}

    def gen():
        view.load(url)
        print('  loading', flush=True)
        yield 3.0
        print('  loaded? running js', flush=True)
        page.runJavaScript("JSON.stringify({status: window.__status || null, errors: window.__errors})" if mode in ("file", "qrc", "app")
                           else "document.body.innerText.slice(0, 3000)",
                           0, lambda r: out.setdefault("r", r))
        yield (lambda: "r" in out, 2)
        print(f"{mode}: {url.toString()} -> {out.get('r')}")

    runner = Runner(gen(), app)  # keep a reference: PySide holds bound-method slots weakly
    app.exec()


def run_bench():
    from PySide6.QtWebSockets import QWebSocketServer
    from PySide6.QtNetwork import QHostAddress
    app = QApplication(sys.argv)
    handler = Handler()
    QWebEngineProfile.defaultProfile().installUrlSchemeHandler(b"app", handler)

    server = QWebSocketServer("bench", QWebSocketServer.SslMode.NonSecureMode)
    server.listen(QHostAddress.LocalHost, 0)
    ws = {"sock": None, "acks": {}, "params": []}

    def on_conn():
        sock = server.nextPendingConnection()
        ws["sock"] = sock

        def on_msg(text):
            m = json.loads(text)
            if m["t"] == "ack":
                ws["acks"][m["seq"]] = time.perf_counter()
            elif m["t"] == "param":
                ws["params"].append(time.perf_counter())
            elif m["t"] == "echo":
                sock.sendTextMessage(json.dumps({"t": "echo", "i": m["i"]}))
        sock.textMessageReceived.connect(on_msg)
    server.newConnection.connect(on_conn)

    bridge = Bridge()
    channel = QWebChannel()
    channel.registerObject("bridge", bridge)
    view = QWebEngineView()
    page = Page(view)
    page.setWebChannel(channel)
    view.setPage(page)
    view.resize(400, 300)
    view.show()

    audio = AudioSim()
    audio.start()
    results = []

    def log(s):
        print(s, flush=True)
        results.append(s)

    def js(code, store, key):
        page.runJavaScript(code, 0, lambda r: store.__setitem__(key, r))

    def paced(send, acks, n=180, hz=60):
        sent = {}
        t0 = time.perf_counter()
        for seq in range(n):
            target = t0 + seq / hz
            while time.perf_counter() < target:
                yield 0.001
            sent[seq] = time.perf_counter()
            send(seq)
        yield (lambda: len(acks) >= n, 5)
        t1 = time.perf_counter()
        rtts = [(acks[s] - sent[s]) * 1000 for s in sent if s in acks]
        return rtts, (t0, t1)

    def burst(send, acks, n=2000):
        t0 = time.perf_counter()
        for seq in range(n):
            send(seq)
        t_sent = time.perf_counter()
        yield (lambda: len(acks) >= n, 20)
        t1 = time.perf_counter()
        return len(acks), (t_sent - t0) * 1000, (t1 - t0) * 1000

    def gen():
        view.load(QUrl(f"app://ui/index.html?ws={server.serverPort()}"))
        yield (lambda: bridge.ready_state is not None, 10)
        yield 1.0
        log(f"ready: {bridge.ready_state}, ws connected: {ws['sock'] is not None}")
        idle_t0 = time.perf_counter()
        yield 2.0
        log(f"audio-sim lateness idle: {stats(audio.window(idle_t0, time.perf_counter()))}")

        # --- Python -> JS at 60 Hz, RTT = push + JS handler + ack back
        for name, send, acks in [
            ("QWebChannel signal", lambda s: bridge.tick.emit(s, payload(s)), bridge.acks),
            ("runJavaScript", lambda s: page.runJavaScript(f"onTick({s},{json.dumps(payload(s))})"), bridge.acks),
            ("WebSocket", lambda s: ws["sock"].sendTextMessage(json.dumps({"t": "tick", "seq": s, **payload(s)})), ws["acks"]),
        ]:
            acks.clear()
            rtts, (a, b) = yield from paced(send, acks)
            log(f"60 Hz push RTT  {name:20s}: {stats(rtts)}; audio-sim lateness: {stats(audio.window(a, b))}")
            yield 0.5
        for name, send, acks in [
            ("QWebChannel signal", lambda s: bridge.tick.emit(s, payload(s)), bridge.acks),
            ("runJavaScript", lambda s: page.runJavaScript(f"onTick({s},{json.dumps(payload(s))})"), bridge.acks),
            ("WebSocket", lambda s: ws["sock"].sendTextMessage(json.dumps({"t": "tick", "seq": s, **payload(s)})), ws["acks"]),
        ]:
            acks.clear()
            t0 = time.perf_counter()
            got, send_ms, total_ms = yield from burst(send, acks)
            log(f"burst 2000 push {name:20s}: acked {got}, py send loop {send_ms:.0f} ms, all acked after {total_ms:.0f} ms "
                f"(~{got / total_ms * 1000:.0f} msg/s); audio-sim lateness: {stats(audio.window(t0, time.perf_counter()))}")
            yield 0.5

        # --- JS -> Python flood (fire and forget)
        for name, fn, store in [("QWebChannel slot", "chFlood", bridge.params), ("WebSocket", "wsFlood", ws["params"])]:
            store.clear()
            r = {}
            t0 = time.perf_counter()
            js(f"{fn}(2000)", r, "ms")
            yield (lambda: len(store) >= 2000, 20)
            t1 = time.perf_counter()
            span = (store[-1] - store[0]) * 1000 if store else float("nan")
            log(f"JS->Py flood 2000 {name:18s}: received {len(store)} in {(t1 - t0) * 1000:.0f} ms "
                f"(first->last {span:.0f} ms, ~{len(store) / max(t1 - t0, 1e-9):.0f} msg/s), JS send loop {r.get('ms', 0):.0f} ms")
            yield 0.5

        # --- JS awaited round trip (call slot with return value / ws echo)
        for name, fn, key in [("QWebChannel slot+return", "chRtt", "chRtt"), ("WebSocket echo", "wsRtt", "wsRtt")]:
            js(f"{fn}(300)", {}, "_")
            yield (lambda: key in bridge.reports, 20)
            log(f"JS awaited RTT {name:24s}: {stats(bridge.reports.get(key, []))}")

        # --- notify-property coalescing (propertyUpdateInterval default)
        log(f"channel.propertyUpdateInterval default = {channel.propertyUpdateInterval()} ms")
        for interval in (channel.propertyUpdateInterval(), 0):
            channel.setPropertyUpdateInterval(interval)
            r0 = {}
            js("getPropCount()", r0, "c")
            yield (lambda: "c" in r0, 2)
            t0 = time.perf_counter()
            for i in range(1000):
                bridge.set_level(i / 1000)
                if i % 10 == 9:
                    yield 0.01  # ~1000 sets/s for ~1 s
            elapsed = time.perf_counter() - t0
            yield 0.3
            r1 = {}
            js("getPropCount()", r1, "c")
            yield (lambda: "c" in r1, 2)
            log(f"notify property set 1000x over {elapsed:.2f}s, interval={interval} ms: JS saw {r1['c'] - r0['c']} updates")

        # --- sustained mixed load: 60 Hz push + 200 Hz param stream, audio jitter
        bridge.params.clear()
        js("window.__pt = setInterval(() => bridge.setParam(1, Math.random()), 5)", {}, "_")
        bridge.acks.clear()
        rtts, (a, b) = yield from paced(lambda s: bridge.tick.emit(s, payload(s)), bridge.acks, n=300)
        js("clearInterval(window.__pt)", {}, "_")
        log(f"mixed 5 s (60 Hz signal push + ~200 Hz setParam): push RTT {stats(rtts)}; params received {len(bridge.params)}; "
            f"audio-sim lateness {stats(audio.window(a, b))}")
        audio.stop = True

    runner = Runner(gen(), app)  # keep a reference: PySide holds bound-method slots weakly
    app.exec()


if __name__ == "__main__":
    print(f"PySide6 / Qt", __import__("PySide6").__version__, __import__("PySide6.QtCore").QtCore.qVersion(),
          "switchinterval", sys.getswitchinterval(), flush=True)
    if sys.argv[1] == "load":
        run_load(sys.argv[2])
    else:
        run_bench()
