"""
Start a built app the way a user would and check it comes up (release CI).

    python packaging/smoke.py dist/<edition>

Starts the web interface and the tkinter interface for 20 seconds each
(``--quit-after``), then the AI worker: it must answer ``hello`` and, in
the ai edition (when the bundle holds the models), generate a drum patch
from the bundled model. Exits non-zero on the first failure.
"""

import json
import subprocess
import sys
from pathlib import Path


def executable(dist):
    for path in (dist / 'Pythonic.app' / 'Contents' / 'MacOS' / 'Pythonic',
                 dist / 'Pythonic' / 'Pythonic.exe', dist / 'Pythonic' / 'Pythonic'):
        if path.is_file():
            return path
    raise SystemExit(f'no Pythonic executable in {dist}')


# Seconds each interface stays open. The page must have loaded by then; a
# first start can be slow (Windows Defender scans a fresh bundle).
QUIT_AFTER = 20


def start(exe, ui):
    print(f'--- {ui} interface', flush=True)
    proc = subprocess.run([str(exe), '--ui', ui, '--quit-after', str(QUIT_AFTER)], timeout=300)
    if proc.returncode:
        raise SystemExit(f'the {ui} interface exited with {proc.returncode}')


def worker(exe, model):
    print('--- AI worker', flush=True)
    requests = [{'id': 1, 'op': 'hello'}]
    if model is not None:
        requests.append({'id': 2, 'op': 'patches', 'path': str(model), 'drum_type': 'bd',
                         'n': 1, 'temperature': 1.0, 'seed': 1})
    requests.append({'id': 3, 'op': 'quit'})
    proc = subprocess.run([str(exe), '--ai-worker'], input=''.join(json.dumps(r) + '\n' for r in requests),
                          capture_output=True, text=True, timeout=600)
    sys.stderr.write(proc.stderr[-4000:])
    replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    print(json.dumps(replies)[:2000])
    if len(replies) != len(requests) or not all(r.get('ok') for r in replies):
        raise SystemExit('the AI worker did not answer every request')
    if model is not None and not replies[1]['result']['candidates']:
        raise SystemExit('the AI worker generated no drum patch')


def main():
    dist = Path(sys.argv[1])
    exe = executable(dist)
    start(exe, 'web')
    start(exe, 'tk')
    models = sorted(dist.rglob('drum_cvae_best.pt'))
    worker(exe, models[0] if models else None)
    print('smoke test passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
