#!/bin/bash
#
# Start Pythonic. The first launch sets everything up: a virtual environment
# in venv/ with the app, both interfaces and the AI drum generator, plus the
# panel's fonts. Later launches start straight away; the setup runs again
# when pyproject.toml changes (new dependencies after an update) or when the
# web interface's QtWebEngine no longer imports (a broken install).
#
# Arguments pass through: --ui web|tk, --devtools, ...

set -e
cd "$(dirname "$0")"

VENV=venv
PY=$VENV/bin/python
STAMP=$VENV/.pythonic-setup  # pyproject.toml's checksum at the last setup

fail() {
    echo "$1" >&2
    exit 1
}

setup() {
    if [ ! -x "$PY" ]; then
        echo "First launch: setting up Pythonic (a few minutes; about 1 GB to download, 2 GB on disk)."
        BASE=$(command -v python3 || command -v python) || fail "Python 3.10 or newer is needed: install it first."
        "$BASE" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
            || fail "Python 3.10 or newer is needed (found $("$BASE" --version 2>&1))."
        "$BASE" -m venv "$VENV" \
            || fail "Could not create the virtual environment. On Debian or Ubuntu: sudo apt install python3-venv"
    else
        echo "Updating Pythonic's packages."
    fi
    # Older pips cannot install a pyproject-only project in editable mode
    "$PY" -m pip install --quiet --upgrade pip
    # The CPU build of PyTorch on Linux: PyPI's default one brings several GB of
    # CUDA libraries, and the drum generator runs fine on the CPU
    if [ "$(uname -s)" = Linux ] && ! "$PY" -c 'import torch' 2>/dev/null; then
        "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    "$PY" -m pip install -e ".[ml]"
    "$PY" tools/fetch_fonts.py || echo "Some fonts could not be downloaded: the panel uses system fonts for them."
    # The AI drum generator's models are Git LFS files: fetch them if the clone skipped them
    if [ -e .git ] && git lfs version > /dev/null 2>&1; then
        git lfs pull || echo "The AI drum generator's models could not be downloaded (git lfs pull)."
    fi
    cksum < pyproject.toml > "$STAMP"
}

# The web interface's QtWebEngine is the part an install most often loses
if [ ! -f "$STAMP" ] || [ "$(cksum < pyproject.toml)" != "$(cat "$STAMP")" ] \
        || ! "$PY" -c 'import PySide6.QtWebEngineWidgets' 2> /dev/null; then
    setup
fi

exec "$PY" run.py "$@"
