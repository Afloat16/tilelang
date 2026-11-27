import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("returncode", [0, 7])
def test_regression_runner_drains_stderr(tmp_path, returncode):
    prefix = "__TILELANG_PERF_RESULTS_JSON__="
    benchmark = tmp_path / "regression_stderr.py"
    benchmark.write_text(
        "import sys\n"
        "print('benchmark progress', flush=True)\n"
        f"print({prefix + json.dumps({'smoke': 1.0})!r}, flush=True)\n"
        "sys.stderr.write('diagnostic-start\\n' + 'x' * (1024 * 1024) + '\\ndiagnostic-end\\n')\n"
        f"sys.stderr.write({prefix + json.dumps({'stderr-only': 99.0}) + chr(10)!r})\n"
        f"sys.exit({returncode})\n"
    )
    runner = Path(__file__).resolve().parents[3] / "maint" / "scripts" / "regression_all.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--examples-root", str(tmp_path), "--format", "json"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "benchmark progress" in completed.stdout
    if returncode == 0:
        assert completed.returncode == 0, completed.stderr
        result_lines = [line[len(prefix) :] for line in completed.stdout.splitlines() if line.startswith(prefix)]
        assert [json.loads(line) for line in result_lines] == [{"smoke": 1.0}]
    else:
        assert completed.returncode != 0
        assert "All benchmark drivers failed" in completed.stderr
        assert "STDOUT:\nbenchmark progress" in completed.stderr
        assert "STDERR:\ndiagnostic-start" in completed.stderr
        assert "diagnostic-end" in completed.stderr
