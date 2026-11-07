"""Build the Windows release: ``python packaging/build.py``.

1. PyInstaller (one folder) -> dist/VoiceFix/VoiceFix.exe
2. LICENSE and README next to the exe (GPL: the licence travels with the program)
3. run the packaged self-test; stop if anything fails
4. zip -> dist/VoiceFix-<version>-windows-x64.zip
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP = DIST / "VoiceFix"


def version() -> str:
    namespace: dict[str, str] = {}
    exec((ROOT / "voicefix" / "__init__.py").read_text(encoding="utf-8"), namespace)
    return namespace["__version__"]


def main() -> int:
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(ROOT / "build"), str(ROOT / "packaging" / "VoiceFix.spec")],
        check=True,
    )
    for name in ("LICENSE", "README.md"):
        shutil.copy2(ROOT / name, APP / name)

    report = DIST / "selftest.txt"
    result = subprocess.run([str(APP / "VoiceFix.exe"), "--self-test", str(report)])
    print(report.read_text(encoding="utf-8"))
    if result.returncode != 0:
        print("Self-test failed; not packaging.", file=sys.stderr)
        return 1

    archive = DIST / f"VoiceFix-{version()}-windows-x64"
    shutil.make_archive(str(archive), "zip", DIST, "VoiceFix")
    print(f"Built {archive}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
