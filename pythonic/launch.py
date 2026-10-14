"""
The ``pythonic`` command: starts one interface per launch.

    pythonic [--ui web|tk] [--devtools [PORT]] [--quit-after SECONDS]

``--ui web`` (the default) opens the HTML panel in a QtWebEngine window;
``--ui tk`` the tkinter window. No interface preference is stored: only the
flag picks the GUI. When the web interface is asked for but PySide6 with
QtWebEngine cannot load (a broken install, or Windows ARM64 where no
QtWebEngine exists), tkinter starts instead and the terminal says why.
"""

import argparse
import os
import sys

DEFAULT_DEVTOOLS_PORT = 9222


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='pythonic', description='Pythonic drum synthesizer')
    parser.add_argument('--ui', choices=('web', 'tk'), default='web',
                        help='interface to start (default: web)')
    parser.add_argument('--devtools', nargs='?', type=int, const=DEFAULT_DEVTOOLS_PORT,
                        default=None, metavar='PORT',
                        help='web interface: open the Chromium DevTools port '
                             f'(default {DEFAULT_DEVTOOLS_PORT}); browse to http://127.0.0.1:PORT')
    parser.add_argument('--quit-after', type=float, default=None, metavar='SECONDS',
                        help='close the window after this many seconds (start-up checks)')
    return parser.parse_args(argv)


def configure_environment(args, environ=None):
    """Environment the web interface reads before Qt starts."""
    environ = os.environ if environ is None else environ
    if args.devtools is not None:
        environ['QTWEBENGINE_REMOTE_DEBUGGING'] = str(args.devtools)


def web_unavailable_reason():
    """None when PySide6 with QtWebEngine can be imported, else why not."""
    try:
        import PySide6.QtWebChannel  # noqa: F401
        import PySide6.QtWebEngineWidgets  # noqa: F401
    except ImportError as exc:
        return str(exc) or type(exc).__name__
    return None


def run_tk(args):
    from gui.main_window import PythonicGUI

    gui = PythonicGUI()
    if args.quit_after is not None:
        gui.root.after(int(args.quit_after * 1000), gui.root.destroy)
    gui.run()
    return 0


def run_web(args):
    from pythonic.web.app import run
    return run(quit_after=args.quit_after, devtools_port=args.devtools)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.ui == 'web':
        reason = web_unavailable_reason()
        if reason is None:
            configure_environment(args)
            return run_web(args)
        print(f'The web interface could not start ({reason}); starting the '
              f'classic interface. ./run.sh or pip install -e . repairs the install.',
              file=sys.stderr)
        return run_tk(args)
    return run_tk(args)


if __name__ == '__main__':
    sys.exit(main())
