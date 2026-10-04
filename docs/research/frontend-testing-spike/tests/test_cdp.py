import os, subprocess, sys, time, json, urllib.request, pathlib
import pytest
from playwright.sync_api import sync_playwright

PORT = 9339
HERE = pathlib.Path(__file__).parents[1]


@pytest.fixture
def app_proc():
    env = dict(os.environ, QTWEBENGINE_REMOTE_DEBUGGING=str(PORT))
    p = subprocess.Popen([sys.executable, str(HERE / 'app.py')], env=env,
                         stdout=subprocess.PIPE, text=True)
    t0 = time.time()
    for _ in range(150):
        try:
            targets = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/json/list'))
            if any(t['url'].startswith('app://') for t in targets):
                break
        except OSError:
            pass
        time.sleep(0.1)
    print(f'startup to CDP target: {time.time() - t0:.2f}s')
    yield p
    p.terminate()
    p.wait(5)


@pytest.mark.xfail(strict=True, reason='QtWebEngine rejects Browser.setDownloadBehavior: '
                   '"Browser context management is not supported" (playwright#36961)')
def test_playwright_over_cdp(app_proc):
    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(f'http://127.0.0.1:{PORT}')
        print('version', browser.version)
        page = next(pg for c in browser.contexts for pg in c.pages if pg.url.startswith('app://'))
        page.wait_for_function('window.__ready === true')
        page.wait_for_function("document.getElementById('step').textContent !== '-'")
        page.evaluate("document.getElementById('k').drag(-50)")
        page.screenshot()
        browser.close()
    line = ''
    for _ in range(50):
        line = app_proc.stdout.readline()
        if line.startswith('SET'):
            break
    assert line.strip() == 'SET ch1.tune 12.0'
