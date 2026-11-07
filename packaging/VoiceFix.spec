# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for VoiceFix (Windows, one folder). Build with packaging/build.py.
from pathlib import Path

ROOT = Path(SPECPATH).parent

# The optional AI denoiser (PyTorch, ~1 GB) is not shipped; the app hides that
# option when it is missing. Other Qt bindings and dev tools are never used.
EXCLUDES = [
    "torch", "torchaudio", "df", "deepfilternet", "DeepFilterLib",
    "PyQt5", "PyQt6", "PySide2", "matplotlib", "tkinter", "IPython",
    "pytest", "pyloudnorm", "PyInstaller",
]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceFix",
    console=False,  # a desktop app: no console window
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="VoiceFix", upx=False)
