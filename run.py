"""
Pythonic - Main Entry Point
Run this script to start the application
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from gui.main_window import main

if __name__ == '__main__':
    main()
