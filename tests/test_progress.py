from __future__ import annotations

from io import StringIO

import pytest

from mlchallenge.progress import ProgressBar


def test_progress_bar_reports_start_and_completion() -> None:
    output = StringIO()
    progress = ProgressBar("Outer CV", total=2, width=10, stream=output)

    progress.start()
    progress.update()
    progress.update()

    lines = output.getvalue().splitlines()
    assert "[----------] 0/2" in lines[0]
    assert "[#####-----] 1/2" in lines[1]
    assert "[##########] 2/2" in lines[2]


def test_progress_bar_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="total"):
        ProgressBar("invalid", total=0)
    with pytest.raises(ValueError, match="negative"):
        ProgressBar("valid", total=1).update(-1)
