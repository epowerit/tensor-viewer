import os
import signal
import subprocess
import sys
import time

import pytest

from tensorviewer.runner import _wait


def started(code: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True)


def test_a_job_that_ends_is_seen_as_soon_as_it_ends():
    process = started("pass")
    _wait(process, timeout=20)
    assert process.returncode == 0


def test_a_job_that_runs_past_its_time_raises_in_time_and_can_still_be_stopped():
    process = started("import time; time.sleep(30)")
    begun = time.perf_counter()
    with pytest.raises(subprocess.TimeoutExpired):
        _wait(process, timeout=0.3)
    assert time.perf_counter() - begun < 5
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    assert process.returncode == -signal.SIGKILL
