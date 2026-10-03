#!/bin/bash
#
# Build the drum_patches_knn/ training set for train.py.
#
# Usage:
#   ./run_classify_drum_patches.sh <reference_dir> <patterns_dir> [extra classify_drum_patches.py args]
#
# Or set REFERENCE_DIR and PATTERNARIUM_DIR and pass only the extra args:
#   REFERENCE_DIR=... PATTERNARIUM_DIR=... ./run_classify_drum_patches.sh --dry-run
#
#   reference_dir  labelled .mtdrum reference patches (names carry the drum type)
#   patterns_dir   root folder of .mtpreset files to classify

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

if [[ $# -ge 2 && "$1" != -* && "$2" != -* ]]; then
    REFERENCE_DIR="$1"
    INPUT_DIR="$2"
    shift 2
else
    REFERENCE_DIR="${REFERENCE_DIR:-}"
    INPUT_DIR="${PATTERNARIUM_DIR:-}"
fi

OUTPUT_DIR="$SCRIPT_DIR/drum_patches_knn"
MANIFEST_PATH="$OUTPUT_DIR/manifest.csv"
K_VALUE=5

if [[ -z "$REFERENCE_DIR" || -z "$INPUT_DIR" ]]; then
    echo "Usage: $0 <reference_dir> <patterns_dir> [extra args]" >&2
    echo "   or: REFERENCE_DIR=... PATTERNARIUM_DIR=... $0 [extra args]" >&2
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: virtualenv Python not found at $VENV_PYTHON" >&2
    echo "Create the venv first or adjust this script." >&2
    exit 1
fi

if [[ ! -d "$REFERENCE_DIR" ]]; then
    echo "Error: reference directory not found: $REFERENCE_DIR" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: input directory not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"

"$VENV_PYTHON" classify_drum_patches.py \
    --reference "$REFERENCE_DIR" \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --manifest "$MANIFEST_PATH" \
    --k "$K_VALUE" \
    "$@"
