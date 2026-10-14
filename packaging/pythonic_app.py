"""
The standalone app's entry point (PyInstaller, see pythonic.spec).

    Pythonic [--ui web|tk] [--devtools [PORT]] [--quit-after SECONDS]
    Pythonic --ai-worker

Without ``--ai-worker`` it is the ``pythonic`` command. With it, the
executable is the AI worker subprocess (``pythonic/app/ai_worker.py``, which
the app core starts this way when frozen): the worker is run by file path, as
a script, so it never imports the ``pythonic`` package.

Nothing here imports ``pythonic`` before the environment is set: numba reads
NUMBA_CACHE_DIR when it is first imported, and the bundle's own folder may be
read-only (an app in /Applications).
"""

import os
import sys

AI_WORKER_FLAG = '--ai-worker'


def bundle_dir():
    return getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def use_bundled_portaudio(bundle):
    """Linux: let sounddevice fall back on the bundled PortAudio when the
    system has none (the system's comes first: it matches the local audio
    setup)."""
    lib = os.path.join(bundle, 'libportaudio.so.2')
    if not sys.platform.startswith('linux') or not os.path.exists(lib):
        return
    import ctypes.util
    find = ctypes.util.find_library
    ctypes.util.find_library = lambda name: (find(name) or lib) if name == 'portaudio' else find(name)


def run_ai_worker(bundle):
    import runpy
    # A windowed app (Windows) starts without stderr: the worker sends what
    # libraries print there
    try:
        os.fstat(2)
    except OSError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')
    sys.argv = [sys.argv[0]]
    runpy.run_path(os.path.join(bundle, 'pythonic', 'app', 'ai_worker.py'), run_name='__main__')
    return 0


def main():
    bundle = bundle_dir()
    if sys.argv[1:2] == [AI_WORKER_FLAG]:
        return run_ai_worker(bundle)
    from platformdirs import user_cache_dir
    os.environ.setdefault('NUMBA_CACHE_DIR',
                          os.path.join(user_cache_dir('Pythonic', appauthor=False), 'numba'))
    use_bundled_portaudio(bundle)
    from pythonic.launch import main as pythonic_main
    return pythonic_main()


if __name__ == '__main__':
    sys.exit(main())
