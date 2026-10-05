import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .models import (
    GradientFlow,
    GradientRequest,
    KnockoutSweep,
    LearnStep,
    ProjectDraft,
    RunError,
    Sensitivity,
    SensitivityRequest,
    SweepResult,
    Trace,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _run_worker(
    project: ProjectDraft,
    timeout: float,
    snapshot_dir: Path | None,
    input_dir: Path | None,
    weights_dir: Path | None,
    mode: str,
    python_executable: Path | None,
    extra: str = "",
) -> tuple[str | None, RunError | None]:
    """Runs the worker on a project; its response text, or why there is none."""
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
            [
                str(python_executable or sys.executable),
                "-m",
                "tensorviewer.worker",
                str(request),
                str(response),
                str(snapshot_dir or Path(directory) / "snapshots"),
                str(input_dir.resolve()) if input_dir else "",
                str(weights_dir.resolve()) if weights_dir else "",
                mode,
                extra,
            ],
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
            return None, RunError(
                type="TimeoutError",
                message=f"Execution exceeded {timeout:g} seconds. Reduce the example or check for an infinite loop.",
            )
        finally:
            # Also clean up any child processes left behind by trusted user code.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode != 0 or not response.exists():
            return None, RunError(
                type="WorkerError",
                message="The execution worker exited without a trace. Check for process exits or excessive resource use in the code.",
            )
        if response.stat().st_size > 16_000_000:
            return None, RunError(
                type="TraceLimitError", message="The trace exceeded the response size limit."
            )
        return response.read_text(), None


def run_project(
    project: ProjectDraft,
    timeout: float = 20,
    snapshot_dir: Path | None = None,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    check_weights_only: bool = False,
    python_executable: Path | None = None,
    learn: LearnStep | None = None,
) -> Trace:
    """Runs and records a project; with `learn`, after one training step."""
    text, error = _run_worker(
        project,
        timeout,
        snapshot_dir,
        input_dir,
        weights_dir,
        "check" if check_weights_only else "learn" if learn else "run",
        python_executable,
        learn.model_dump_json() if learn else "",
    )
    if error:
        return Trace(error=error)
    try:
        return Trace.model_validate_json(text)
    except ValueError:
        return Trace(
            error=RunError(type="WorkerError", message="The worker returned an invalid trace.")
        )


def run_gradients(
    project: ProjectDraft,
    target: GradientRequest,
    timeout: float = 20,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    python_executable: Path | None = None,
) -> GradientFlow:
    """Runs the project again with gradients on, for every step's gradient size."""
    empty = GradientFlow(tensor_id=target.tensor_id, index=target.index)
    text, error = _run_worker(
        project,
        timeout,
        None,
        input_dir,
        weights_dir,
        "gradients",
        python_executable,
        target.model_dump_json(),
    )
    if error:
        return empty.model_copy(update={"error": error})
    try:
        return GradientFlow.model_validate_json(text)
    except ValueError:
        return empty.model_copy(
            update={
                "error": RunError(
                    type="WorkerError", message="The worker returned an invalid answer."
                )
            }
        )


def run_sensitivity(
    project: ProjectDraft,
    target: SensitivityRequest,
    timeout: float = 20,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    python_executable: Path | None = None,
) -> Sensitivity:
    """Runs the project again with gradients on, for one result cell's."""
    empty = Sensitivity(
        tensor_id=target.tensor_id, index=target.index, input_id="", kind="gradient", values=[]
    )
    text, error = _run_worker(
        project,
        timeout,
        None,
        input_dir,
        weights_dir,
        "sensitivity",
        python_executable,
        target.model_dump_json(),
    )
    if error:
        return empty.model_copy(update={"error": error})
    try:
        return Sensitivity.model_validate_json(text)
    except ValueError:
        return empty.model_copy(
            update={
                "error": RunError(
                    type="WorkerError", message="The worker returned an invalid answer."
                )
            }
        )


def run_sweep(
    project: ProjectDraft,
    request: KnockoutSweep,
    timeout: float = 60,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    python_executable: Path | None = None,
) -> SweepResult:
    """Runs the project once per slice, each with that slice knocked out."""
    empty = SweepResult(step=request.step, axis=request.axis, mode=request.mode)
    text, error = _run_worker(
        project,
        timeout,
        None,
        input_dir,
        weights_dir,
        "sweep",
        python_executable,
        request.model_dump_json(),
    )
    if error:
        return empty.model_copy(update={"error": error})
    try:
        return SweepResult.model_validate_json(text)
    except ValueError:
        return empty.model_copy(
            update={
                "error": RunError(
                    type="WorkerError", message="The worker returned an invalid answer."
                )
            }
        )


def run_evaluation(
    request: dict, timeout: float = 10, python_executable: Path | None = None
) -> dict:
    """Evaluates a watch expression in its own process; its answer as a dict."""
    with tempfile.TemporaryDirectory(prefix="tensorviewer-watch-") as directory:
        asked = Path(directory) / "request.json"
        answer = Path(directory) / "response.json"
        asked.write_text(json.dumps(request))
        process = subprocess.Popen(
            [
                str(python_executable or sys.executable),
                "-m",
                "tensorviewer.evaluate",
                str(asked),
                str(answer),
            ],
            cwd=directory,
            env={**os.environ, "PYTHONPATH": str(BACKEND_ROOT), "OMP_NUM_THREADS": "1"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"kind": "error", "text": f"The expression took longer than {timeout:g} s."}
        if process.returncode != 0 or not answer.exists():
            return {"kind": "error", "text": "The expression stopped its process."}
        return json.loads(answer.read_text())
