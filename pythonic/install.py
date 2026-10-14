"""
The in-app installer of the AI drum generator's ``ml`` extra: its packages
mirror ``pyproject.toml`` (a test keeps them in step), and installing runs pip
in the current environment.
"""

import sys

EXTRAS = {
    'ml': ['torch>=2.0.0'],
}


def pip_command(extra):
    """The pip command line that installs an extra's packages."""
    return [sys.executable, '-m', 'pip', 'install', *EXTRAS[extra]]
