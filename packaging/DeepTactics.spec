# Build: ..\_deeptactics_build_venv\Scripts\pyinstaller packaging\DeepTactics.spec --noconfirm
from pathlib import Path
ROOT = Path(SPECPATH).parent
import sys; sys.path.insert(0, str(ROOT))
from PyInstaller.utils.hooks import collect_submodules
SKIP = ('chess_engine.tests', 'chess_engine.devtools')
HIDDEN = collect_submodules('chess_engine', filter=lambda m: not m.startswith(SKIP))

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[(str(ROOT / "chess_engine/native/deeptactics_native_runtime_v2.dll"), "chess_engine/native")],
    datas=[(str(ROOT / "assets" / d), f"assets/{d}") for d in
           ("benchmarks", "i18n", "icon", "openings")],
    hiddenimports=HIDDEN,
    excludes=["tkinter", "pytest", "unittest", "chess", "numpy",
              "chess_engine.tests",
              "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DeepTactics", console=False,
          icon=str(ROOT / "assets/icon/program_icon.png"))
coll = COLLECT(exe, a.binaries, a.datas, name="DeepTactics")
