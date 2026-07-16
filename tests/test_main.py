"""Basic end-to-end test for the submission pipeline in ``main.py``.

``main.py`` is a script that runs the whole pipeline at import time, so instead
of importing it we execute it as a subprocess and validate the observable
result: a well-formed ``output/submission.csv``.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_PATH = PROJECT_ROOT / "output" / "submission.csv"
X_TEST_PATH = PROJECT_ROOT / "data" / "X_test.csv"


@pytest.fixture(scope="module")
def run_main() -> subprocess.CompletedProcess[str]:
    """Run ``main.py`` once and return the completed process."""
    return subprocess.run(
        [sys.executable, "main.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_main_runs_successfully(run_main: subprocess.CompletedProcess[str]) -> None:
    """The program exits cleanly and reports that it saved the submission."""
    assert run_main.returncode == 0, run_main.stderr
    assert "Submission saved to" in run_main.stdout


def test_submission_file_created(run_main: subprocess.CompletedProcess[str]) -> None:
    """A submission file is written to the expected location."""
    assert SUBMISSION_PATH.is_file()


def test_submission_format(run_main: subprocess.CompletedProcess[str]) -> None:
    """The submission has the required ``id,y`` columns, one row per test sample,
    sequential ids, and no missing predictions."""
    submission = pd.read_csv(SUBMISSION_PATH)
    n_test = len(pd.read_csv(X_TEST_PATH))

    assert list(submission.columns) == ["id", "y"]
    assert len(submission) == n_test
    assert submission["id"].tolist() == list(range(n_test))
    assert submission["y"].notna().all()
