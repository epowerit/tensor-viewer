import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .models import (
    CausalTrace,
    CausalTraceJob,
    GradientFlow,
    GradientRequest,
    KnockoutSweep,
    LearnStep,
    LogitLens,
    ProjectDraft,
    RunError,
    Sensitivity,
    SensitivityRequest,
    SweepResult,
    Timings,
    Trace,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]

WORKER = "tensorviewer.worker"
EVALUATOR = "tensorviewer.evaluate"


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(BACKEND_ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


# One worker of each kind waits, started and with PyTorch imported, for the
# next job on the server's own Python (see standby.py).
_standby: dict[str, subprocess.Popen] = {}
_standby_lock = threading.Lock()


def warm(module: str) -> None:
    """Start a worker of this kind on standby, unless one is waiting."""
    if os.environ.get("TENSORVIEWER_NO_STANDBY"):
        return
    with _standby_lock:
        waiting = _standby.get(module)
        if waiting is not None and waiting.poll() is None:
            return
        _standby[module] = subprocess.Popen(
            [sys.executable, "-m", module, "--standby"],
            cwd=BACKEND_ROOT,
            env=_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _start(
    module: str, args: list[str], cwd: str, python_executable: Path | None
) -> subprocess.Popen:
    """Start a job: on the waiting worker when one is ready for this Python,
    otherwise on a fresh process, as before."""
    if python_executable is None or Path(python_executable) == Path(sys.executable):
        with _standby_lock:
            waiting = _standby.pop(module, None)
        if waiting is not None and waiting.poll() is None and waiting.stdin:
            try:
                waiting.stdin.write((json.dumps({"cwd": cwd, "args": args}) + "\n").encode())
                waiting.stdin.close()
                return waiting
            except OSError:
                waiting.kill()
                waiting.wait()
    return subprocess.Popen(
        [str(python_executable or sys.executable), "-m", module, *args],
        cwd=cwd,
        env=_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


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
        process = _start(
            WORKER,
            [
                str(request),
                str(response),
                str(snapshot_dir or Path(directory) / "snapshots"),
                str(input_dir.resolve()) if input_dir else "",
                str(weights_dir.resolve()) if weights_dir else "",
                mode,
                extra,
            ],
            directory,
            python_executable,
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
            warm(WORKER)
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


def run_timings(
    project: ProjectDraft,
    passes: int = 5,
    timeout: float = 60,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    python_executable: Path | None = None,
) -> Timings:
    """Runs the project again, a warm-up pass and `passes` timed ones."""
    empty = Timings(passes=passes)
    text, error = _run_worker(
        project,
        timeout,
        None,
        input_dir,
        weights_dir,
        "timings",
        python_executable,
        str(passes),
    )
    if error:
        return empty.model_copy(update={"error": error})
    try:
        return Timings.model_validate_json(text)
    except ValueError:
        return empty.model_copy(
            update={
                "error": RunError(
                    type="WorkerError", message="The worker returned an invalid answer."
                )
            }
        )


def run_logit_lens(
    project: ProjectDraft,
    timeout: float = 60,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    python_executable: Path | None = None,
) -> LogitLens:
    """Runs the project once per layer, each read by the final layers."""
    text, error = _run_worker(
        project, timeout, None, input_dir, weights_dir, "lens", python_executable
    )
    if error:
        return LogitLens(error=error)
    try:
        return LogitLens.model_validate_json(text)
    except ValueError:
        return LogitLens(
            error=RunError(type="WorkerError", message="The worker returned an invalid answer.")
        )


def run_causal_trace(
    project: ProjectDraft,
    job: CausalTraceJob,
    timeout: float = 120,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    python_executable: Path | None = None,
) -> CausalTrace:
    """Runs the project once per layer and position, each patched from the clean run."""
    text, error = _run_worker(
        project,
        timeout,
        None,
        input_dir,
        weights_dir,
        "trace",
        python_executable,
        job.model_dump_json(),
    )
    if error:
        return CausalTrace(against=job.against, error=error)
    try:
        return CausalTrace.model_validate_json(text)
    except ValueError:
        return CausalTrace(
            against=job.against,
            error=RunError(type="WorkerError", message="The worker returned an invalid answer."),
        )


def run_evaluation(
    request: dict, timeout: float = 10, python_executable: Path | None = None
) -> dict:
    """Evaluates a watch expression in its own process; its answer as a dict."""
    with tempfile.TemporaryDirectory(prefix="tensorviewer-watch-") as directory:
        asked = Path(directory) / "request.json"
        answer = Path(directory) / "response.json"
        asked.write_text(json.dumps(request))
        process = _start(EVALUATOR, [str(asked), str(answer)], directory, python_executable)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"kind": "error", "text": f"The expression took longer than {timeout:g} s."}
        finally:
            warm(EVALUATOR)
        if process.returncode != 0 or not answer.exists():
            return {"kind": "error", "text": "The expression stopped its process."}
        return json.loads(answer.read_text())
