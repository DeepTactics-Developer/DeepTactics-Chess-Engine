# Third-party notices

DeepTactics Chess Engine itself is licensed under the GNU GPL v3 (see `LICENSE`).
The Windows release bundles the following third-party components, which keep their own licenses.

## Qt 6 / PySide6 (Qt for Python)

- Used modules: QtCore, QtGui, QtWidgets, QtNetwork (shipped as unmodified shared libraries: `Qt6*.dll`, `PySide6/*.pyd`).
- License: GNU Lesser General Public License v3 (LGPL-3.0) — https://www.gnu.org/licenses/lgpl-3.0.html
- Copyright (C) The Qt Company Ltd. and other contributors.
- Source code: https://download.qt.io/official_releases/qt/ and https://pypi.org/project/PySide6/
- The libraries are dynamically linked and live in the `_internal` folder next to the executable, so you may replace them with your own compatible build of the same version.

## Python

- Python runtime, PSF License — https://docs.python.org/3/license.html

## PyInstaller

- Used to create the executable bundle. GPL-2.0 with the bootloader exception, which permits distributing the produced bundle under any license — https://pyinstaller.org/en/stable/license.html

## Opening data

- Opening names/lines derived from lichess-org/chess-openings, CC0 1.0 — https://github.com/lichess-org/chess-openings
