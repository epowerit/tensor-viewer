"""A worker started ahead of its job, with its libraries already imported.

Importing PyTorch takes most of a second, longer than many runs. The server
keeps one worker of each kind started and waiting, so a run or a watch only
waits for its own work. Each job still gets a fresh process of its own: the
waiting worker takes one job, runs it and exits, and the server starts the
next one.
"""

import json
import os
import sys

STANDBY = "--standby"


def receive() -> None:
    """In a worker started on standby, wait for the job: its folder and its
    arguments, as the server would have passed them on a cold start."""
    if sys.argv[1:] != [STANDBY]:
        return
    line = sys.stdin.readline()
    if not line:
        # The server went away before giving a job.
        raise SystemExit(0)
    job = json.loads(line)
    os.chdir(job["cwd"])
    sys.argv = [sys.argv[0], *job["args"]]
