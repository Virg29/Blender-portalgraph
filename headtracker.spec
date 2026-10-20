# Сборка: pyinstaller headtracker.spec  ->  dist/headtracker/headtracker(.exe)
import glob
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [("face_landmarker.task", ".")]
datas += collect_data_files("mediapipe")
binaries = collect_dynamic_libs("mediapipe")

# Python от uv (python-build-standalone) на Linux держит Tcl/Tk 9 отдельными .so в lib/,
# и PyInstaller их не находит — без них tkinter не импортируется. На Windows ничего не найдётся.
for pattern in ("libtcl*.so*", "libtk*.so*"):
    for path in glob.glob(os.path.join(sys.base_prefix, "lib", pattern)):
        binaries.append((path, "."))

a = Analysis(
    ["main.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=["PIL._tkinter_finder"],  # Pillow грузит его динамически для ImageTk
    excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="headtracker", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="headtracker")
