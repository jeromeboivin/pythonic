"""
A stand-in for the AI worker (pythonic/app/ai_worker.py) that speaks its
JSON-lines protocol without torch or models, for the AI core tests.

- ``patches`` answers n candidates named ``<TYPE> <i>`` (1-based) whose
  oscillator frequency is ``100 * i`` Hz, decay ``50 * i`` ms.
- ``patterns`` answers n patterns: pattern k triggers channel c on every
  (k + c + 1)-th step.
- ``--gate PATH``: while PATH exists, requests other than ``hello`` wait.
- ``--crash TYPE``: a ``patches`` request of that drum type ends the process.
- ``--log PATH``: every request is appended to PATH as a JSON line.
- A load of a path containing ``broken`` fails as a model load.
"""

import argparse
import json
import os
import sys
import time


def patches(drum_type, n):
    return [{
        'Name': f'{drum_type.upper()} {i}', 'OscFreq': 100.0 * i, 'OscWave': 'Sine',
        'OscAtk': 0.0, 'OscDcy': 50.0 * i, 'ModMode': 'Decay', 'ModAmt': 0.0,
        'ModRate': 100.0, 'NFilMod': 'LP', 'NFilFrq': 5000.0, 'NFilQ': 1.0,
        'NStereo': False, 'NEnvMod': 'Exp', 'NEnvAtk': 0.0, 'NEnvDcy': 100.0, 'Mix': 60.0,
        'DistAmt': 0.0, 'EQFreq': 1000.0, 'EQGain': 0.0, 'Level': -3.0, 'Pan': 0.0,
        'Output': 'A', 'OscVel': 50.0, 'NVel': 50.0, 'ModVel': 0.0,
    } for i in range(1, n + 1)]


def pattern(k):
    data = {'Length': 16, 'Chained': False}
    for c in range(8):
        every = k + c + 1
        data[str(c + 1)] = {
            'Triggers': ''.join('#' if s % every == 0 else '-' for s in range(16)),
            'Accents': '#' + '-' * 15,
            'Fills': '-' * 16,
        }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate')
    parser.add_argument('--crash')
    parser.add_argument('--log')
    args = parser.parse_args()
    out = sys.stdout
    for line in sys.stdin:
        request = json.loads(line)
        if args.log:
            with open(args.log, 'a') as f:
                f.write(json.dumps(request) + '\n')
        op = request.get('op')
        while args.gate and op != 'hello' and os.path.exists(args.gate):
            time.sleep(0.01)
        reply = {'id': request['id'], 'ok': True}
        path = request.get('path') or ''
        if op == 'quit':
            reply['result'] = {}
        elif 'broken' in path:
            reply = {'id': request['id'], 'ok': False, 'load': True,
                     'error': 'the model could not be loaded: broken'}
        elif op == 'load':
            reply['result'] = {'path': path, 'sampling': 'prior'}
        elif op == 'patches':
            if request['drum_type'] == args.crash:
                os._exit(3)
            reply['result'] = {'candidates': patches(request['drum_type'], request['n']),
                               'sampling': 'prior'}
        elif op == 'patterns':
            reply['result'] = {'patterns': [pattern(k) for k in range(request['n'])]}
        else:
            reply = {'id': request['id'], 'ok': False, 'error': f'unknown op {op}'}
        out.write(json.dumps(reply) + '\n')
        out.flush()
        if op == 'quit':
            break


if __name__ == '__main__':
    main()
