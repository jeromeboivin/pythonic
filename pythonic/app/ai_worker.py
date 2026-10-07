"""
The AI worker: a subprocess that runs the drum patch and pattern models for
the app core's AI module (``ai.py``), so nothing in the GUI process imports
torch (importing it and loading the patch model hold the GIL for 126-381 ms,
long enough to break the audio stream).

Run as a script (``python ai_worker.py``): it loads ``drum_generator.py`` and
``pattern_generator.py`` by file path, without importing the ``pythonic``
package (the synth and its compiled kernels are not needed here).

Protocol: JSON lines. Each request on stdin is ``{"id": n, "op": ..., ...}``
and gets one reply on stdout, in order: ``{"id": n, "ok": true, "result":
{...}}`` or ``{"id": n, "ok": false, "error": "message", "load": bool}``
(``load`` is true when the model could not be loaded). Anything a library
prints goes to stderr. Ops:

- ``hello``: ``{"pid"}``.
- ``load`` (kind ``patch`` or ``pattern``, path): load a checkpoint (kept
  until another path is asked for). ``{"path", "sampling"}``.
- ``patches`` (path, drum_type, n, temperature, seed): ``{"candidates": [raw
  patch dicts], "sampling"}``; the model is loaded first if needed.
- ``patterns`` (path, raw_patches, tempo, swing, fill_rate, step_rate, n,
  temperature, seed): ``{"patterns": [pattern dicts]}``.
- ``quit``: replies, then exits.

The worker ends when stdin closes.
"""

import importlib.util
import json
import os
import sys

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    """Load pythonic/<name>.py on its own (numpy only at import time)."""
    spec = importlib.util.spec_from_file_location(f'_pythonic_ai_{name}',
                                                  os.path.join(_PACKAGE, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LoadError(Exception):
    """A checkpoint could not be loaded."""


class Models:
    """The loaded models, one per kind, loaded on first use."""

    def __init__(self):
        self._modules = {}
        self._loaded = {}  # kind -> (path, generator)

    def _module(self, name):
        if name not in self._modules:
            self._modules[name] = _load(name)
        return self._modules[name]

    def get(self, kind, path):
        loaded = self._loaded.get(kind)
        if loaded is not None and loaded[0] == path:
            return loaded[1]
        if not path or not os.path.isfile(path):
            raise LoadError(f'model file not found: {path}')
        if kind == 'patch':
            generator = self._module('drum_generator').PatchGenerator()
        elif kind == 'pattern':
            generator = self._module('pattern_generator').PatternGenerator()
        else:
            raise ValueError(f'not a model kind: {kind!r}')
        try:
            generator.load_model(path)
        except Exception as exc:
            raise LoadError(f'{type(exc).__name__}: {exc}') from None
        self._loaded[kind] = (path, generator)
        return generator

    def handle(self, request):
        op = request.get('op')
        if op == 'hello':
            return {'pid': os.getpid()}
        if op == 'load':
            generator = self.get(request['kind'], request['path'])
            return {'path': request['path'], 'sampling': _sampling(generator)}
        if op == 'patches':
            generator = self.get('patch', request['path'])
            candidates = generator.generate(request['drum_type'], n=int(request['n']),
                                            temperature=float(request['temperature']),
                                            seed=request.get('seed'))
            return {'candidates': candidates, 'sampling': _sampling(generator)}
        if op == 'patterns':
            generator = self.get('pattern', request['path'])
            patterns = generator.generate(
                request['raw_patches'], tempo=float(request['tempo']),
                swing=float(request['swing']), fill_rate=float(request['fill_rate']),
                step_rate=request['step_rate'], n=int(request['n']),
                temperature=float(request['temperature']), seed=request.get('seed'))
            return {'patterns': patterns}
        raise ValueError(f'unknown op: {op!r}')


def _sampling(generator):
    return getattr(generator, 'sampling_summary', None)


def main():
    # Keep fd 1 for the protocol; everything printed goes to stderr
    out = os.fdopen(os.dup(1), 'w', buffering=1, encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    models = Models()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        reply = {'id': request.get('id')}
        try:
            if request.get('op') == 'quit':
                result = {}
            else:
                result = models.handle(request)
        except LoadError as exc:
            reply.update(ok=False, error=f'the model could not be loaded: {exc}', load=True)
        except Exception as exc:
            reply.update(ok=False, error=f'{type(exc).__name__}: {exc}', load=False)
        else:
            reply.update(ok=True, result=result)
        out.write(json.dumps(reply) + '\n')
        out.flush()
        if request.get('op') == 'quit':
            break


if __name__ == '__main__':
    main()
