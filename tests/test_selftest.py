from pathlib import Path

from voicefix.selftest import CHECKS, run


def test_self_test_passes_and_writes_a_report(tmp_path: Path):
    report = tmp_path / "report.txt"
    assert run(report) == 0
    lines = report.read_text(encoding="utf-8").splitlines()
    assert len([line for line in lines if line.startswith("OK")]) == len(CHECKS)
    assert lines[-1].endswith("全部通過")
