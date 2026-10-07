#!/bin/bash

# Activate the Python 3 virtual environment
source venv/bin/activate

# Run the Python program (arguments pass through: --ui web|tk, --devtools, ...)
python3 run.py "$@"
