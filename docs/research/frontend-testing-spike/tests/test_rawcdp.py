import base64, itertools, json, pathlib, tempfile, time, urllib.request
from websockets.sync.client import connect
from test_cdp import app_proc, PORT, HERE  # noqa: F401  (fixture reuse)


class Cdp:
    def __init__(self, ws_url):
        self.ws = connect(ws_url, max_size=None)
        self.ids = itertools.count(1)

    def send(self, method, **params):
        i = next(self.ids)
        self.ws.send(json.dumps({'id': i, 'method': method, 'params': params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get('id') == i:
                if 'error' in msg:
                    raise RuntimeError(msg['error'])
                return msg['result']

    def eval(self, expr):
        r = self.send('Runtime.evaluate', expression=expr, returnByValue=True, awaitPromise=True)
        return r['result'].get('value')

    def wait(self, expr, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            if self.eval(expr):
                return
            time.sleep(0.05)
        raise TimeoutError(expr)


def test_raw_cdp(app_proc):
    targets = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json/list'))
    page = next(t for t in targets if t['url'].startswith('app://'))
    print('version', json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json/version')))
    cdp = Cdp(page['webSocketDebuggerUrl'])
    cdp.wait('window.__ready === true')
    cdp.wait("document.getElementById('step').textContent !== '-'")
    # real input events through the compositor
    box = cdp.eval("(()=>{const r=document.getElementById('k').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()")
    print('knob box', box)
    for t in ('mousePressed', 'mouseReleased'):
        cdp.send('Input.dispatchMouseEvent', type=t, x=box[0], y=box[1], button='left', clickCount=1)
    cdp.eval("document.getElementById('k').drag(-50)")
    shot = cdp.send('Page.captureScreenshot', format='png')
    (pathlib.Path(tempfile.gettempdir()) / 'spike-cdp-shot.png').write_bytes(base64.b64decode(shot['data']))
    for line in app_proc.stdout:
        if line.startswith('SET'):
            break
    assert line.strip() == 'SET ch1.tune 12.0'
