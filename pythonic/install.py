"""
In-app installers for optional parts of Pythonic.

The packages of each extra mirror ``pyproject.toml`` (a test keeps them in
step): the AI drum generator's ``ml`` extra, and ``qt`` for the web interface
(PySide6 with QtWebEngine, a required dependency except on Windows ARM64 where
no QtWebEngine wheels exist). Installing runs pip in the current environment.
"""

import platform
import subprocess
import sys

EXTRAS = {
    'ml': ['torch>=2.0.0'],
    'qt': ['PySide6-Essentials>=6.8', 'PySide6-Addons>=6.8'],
}


def windows_arm64():
    """True on Windows ARM64, where PySide6 has no QtWebEngine."""
    return sys.platform == 'win32' and platform.machine().lower() in ('arm64', 'aarch64')


def pip_command(extra):
    """The pip command line that installs an extra's packages."""
    return [sys.executable, '-m', 'pip', 'install', *EXTRAS[extra]]


def install_extra(extra, on_output=None):
    """Run pip for an extra's packages; on_output(line) streams its output.
    Returns True on success."""
    try:
        proc = subprocess.Popen(pip_command(extra), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if on_output:
                on_output(line.rstrip())
        proc.wait()
        return proc.returncode == 0
    except Exception as exc:  # pip missing, no permission, ...
        if on_output:
            on_output(f'Install failed: {exc}')
        return False
