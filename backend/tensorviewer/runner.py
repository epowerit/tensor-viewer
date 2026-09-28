import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .models import ProjectDraft, RunError, Trace

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def run_project(project: ProjectDraft, timeout: float = 20) -> Trace:
    with tempfile.TemporaryDirectory(prefix="tensorviewer-run-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        request.write_text(project.model_dump_json())
        env = {
            **os.environ,
            "PYTHONPATH": str(BACKEND_ROOT),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "tensorviewer.worker", str(request), str(response)],
            cwd=directory,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return Trace(
                error=RunError(
                    type="TimeoutError",
                    message=f"Execution exceeded {timeout:g} seconds. Reduce the example or check for an infinite loop.",
                )
            )
        finally:
            # Also clean up any child processes left behind by trusted user code.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode != 0 or not response.exists():
            return Trace(
                error=RunError(
                    type="WorkerError",
                    message="The execution worker exited without a trace. Check for process exits or excessive resource use in the code.",
                )
            )
        if response.stat().st_size > 16_000_000:
            return Trace(
                error=RunError(
                    type="TraceLimitError", message="The trace exceeded the response size limit."
                )
            )
        try:
            return Trace.model_validate_json(response.read_text())
        except ValueError:
            return Trace(
                error=RunError(type="WorkerError", message="The worker returned an invalid trace.")
            )
    return None
