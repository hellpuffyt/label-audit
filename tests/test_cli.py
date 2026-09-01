from __future__ import annotations

import json
from pathlib import Path

import pytest

from label_audit import __version__, cli


def write_csv(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    return str(path)


CLEAN = "id,text,label\n" + "".join(
    f"a{i},text number {i},{'pos' if i % 2 else 'neg'}\n" for i in range(10)
)

CONTAMINATED = (
    "id,text,label,split\n"
    "a,shared sentence,pos,train\n"
    "b,shared  SENTENCE,pos,test\n"
    "c,unrelated,neg,train\n"
    "d,also unrelated,neg,test\n"
)


class TestExitCodes:
    def test_clean_dataset_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main([write_csv(tmp_path / "d.csv", CLEAN)]) == cli.EXIT_OK

    def test_error_finding_exits_three(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main([write_csv(tmp_path / "d.csv", CONTAMINATED)]) == cli.EXIT_FINDINGS

    def test_fail_on_never_exits_zero_despite_errors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_csv(tmp_path / "d.csv", CONTAMINATED)
        assert cli.main([path, "--fail-on", "never"]) == cli.EXIT_OK

    def test_fail_on_warning_trips_on_warnings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        body = "id,text,label\n" + "a,same,x\nb,same,x\n" + "".join(
            f"c{i},unique {i},y\n" for i in range(4)
        )
        path = write_csv(tmp_path / "d.csv", body)
        assert cli.main([path]) == cli.EXIT_OK
        assert cli.main([path, "--fail-on", "warning"]) == cli.EXIT_FINDINGS

    def test_missing_file_exits_one_with_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main([str(tmp_path / "absent.csv")])
        captured = capsys.readouterr()
        assert code == cli.EXIT_USAGE
        assert captured.err.startswith("error:")
        assert captured.out == ""


class TestOutput:
    def test_text_is_the_default_format(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main([write_csv(tmp_path / "d.csv", CLEAN)])
        out = capsys.readouterr().out
        assert "10 items" in out

    def test_json_format_is_machine_readable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main([write_csv(tmp_path / "d.csv", CONTAMINATED), "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["items"] == 4
        assert any(f["check"] == "duplicates" for f in payload["findings"])

    def test_check_filter_restricts_findings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_csv(tmp_path / "d.csv", CONTAMINATED)
        cli.main([path, "--format", "json", "--check", "leakage"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["findings"] == []

    def test_check_filter_still_governs_exit_code(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_csv(tmp_path / "d.csv", CONTAMINATED)
        assert cli.main([path, "--check", "leakage"]) == cli.EXIT_OK

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestModuleEntryPoint:
    def test_python_dash_m_invocation(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        path = write_csv(tmp_path / "d.csv", CLEAN)
        done = subprocess.run(
            [sys.executable, "-m", "label_audit", path],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0
        assert "10 items" in done.stdout
