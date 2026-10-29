"""Exercise the real shell supervisor with local dummy jobs, without API calls."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("failed_stage", ["", "correct", "false"])
@pytest.mark.parametrize("retry_seed", [0, 1])
def test_all_starts_both_jobs_and_propagates_failure(tmp_path, failed_stage, retry_seed):
    executable = tmp_path / "fake-python"
    executable.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import os
            from pathlib import Path
            import sys
            import time

            root = Path(os.environ['FAKE_GENERATION_DIR'])
            kind = 'correct' if 'generate-correct-documents' in sys.argv else 'false'
            seed = sys.argv[sys.argv.index('--retry-seed') + 1]
            (root / (kind + '.seed')).write_text(seed)
            (root / (kind + '.started')).touch()
            deadline = time.monotonic() + 5
            # A sequential launcher cannot pass this barrier.
            while not all((root / (k + '.started')).exists() for k in ('correct', 'false')):
                if time.monotonic() >= deadline:
                    raise SystemExit(8)
                time.sleep(0.01)
            (root / (kind + '.finished')).touch()
            raise SystemExit(7 if os.environ['FAKE_FAILED_STAGE'] == kind else 0)
            """
        )
    )
    executable.chmod(0o700)
    script = Path(__file__).resolve().parents[1] / "scripts/run_together_generation.sh"
    result = subprocess.run(
        ["bash", str(script), "all", "--retry-seed", str(retry_seed)],
        env={
            **os.environ,
            "PYTHON": str(executable),
            "TOGETHER_API_KEY": "local-test-only",
            "FAKE_GENERATION_DIR": str(tmp_path),
            "FAKE_FAILED_STAGE": failed_stage,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (7 if failed_stage else 0), result.stderr
    assert (tmp_path / "correct.finished").exists()
    assert (tmp_path / "false.finished").exists()
    assert (tmp_path / "correct.seed").read_text() == str(retry_seed)
    assert (tmp_path / "false.seed").read_text() == str(retry_seed)
