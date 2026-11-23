"""PyInstaller entry point: run from the exe folder so assets/ and artifacts/ resolve beside it."""
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    os.chdir(Path(sys.executable).resolve().parent)

from chess_engine.main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--ui", "qt", *sys.argv[1:]]
    main()
