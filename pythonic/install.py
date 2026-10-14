"""
The in-app installer of the AI drum generator's ``ml`` extra: its packages
mirror ``pyproject.toml`` (a test keeps them in step), and installing runs pip
in the current environment. The standalone app has no pip: its AI generator
comes with the Pythonic + AI edition instead.
"""

import sys

EXTRAS = {
    'ml': ['torch>=2.0.0'],
}


def pip_command(extra):
    """The pip command line that installs an extra's packages; empty in the
    standalone app."""
    if getattr(sys, 'frozen', False):
        return []
    return [sys.executable, '-m', 'pip', 'install', *EXTRAS[extra]]
