"""One trusted local code execution per process. This is not a security sandbox."""

import contextlib
import inspect
import io
import json
import sys
import time
import traceback
from math import prod
from pathlib import Path

import torch

from .input_files import load_array
from .models import InputSpec, ProjectDraft, RunError, Trace
from .tracing import Recorder, tensors_in


class LimitedOutput(io.StringIO):
    def write(self, text):
        remaining = max(0, 8000 - self.tell())
        super().write(text[:remaining])
        return len(text)


def make_input(spec: InputSpec, shapes: bool, uploaded=None) -> torch.Tensor:
    dtype = getattr(torch, spec.dtype)
    if shapes:
        return torch.empty(spec.shape, dtype=dtype, device="meta")
    if spec.generator == "uploaded":
        return torch.from_numpy(uploaded)
    if spec.generator == "arange":
        return torch.arange(prod(spec.shape), dtype=dtype).reshape(spec.shape)
    if spec.generator == "random":
        generator = (
            torch.Generator(device="cpu").manual_seed(spec.seed)
            if spec.random_stream == "input"
            else None
        )
        return torch.randn(spec.shape, dtype=dtype, generator=generator)
    return getattr(torch, spec.generator)(spec.shape, dtype=dtype)


def execute(
    project: ProjectDraft, snapshot_dir: Path | None = None, input_dir: Path | None = None
) -> Trace:
    started = time.perf_counter()
    trace = Trace()
    recorder = None
    stream = LimitedOutput()
    filename = "<tensorviewer-project>"
    try:
        torch.set_num_threads(1)
        torch.manual_seed(project.input.seed)
        shapes = project.capture_mode == "shapes"
        device = "meta" if shapes else "cpu"
        inputs = project.forward_inputs
        # Validate every numeric upload before any project source is executed.
        uploaded = {}
        for item in inputs:
            if item.input.uploaded and not shapes:
                if input_dir is None:
                    raise ValueError("The uploaded input directory is unavailable.")
                uploaded[item.name] = load_array(input_dir, item.input.uploaded)
        with (
            contextlib.redirect_stdout(stream),
            contextlib.redirect_stderr(stream),
            torch.device(device),
        ):
            namespace = {"__name__": "tensorviewer_user_project"}
            exec(compile(project.code, filename, "exec"), namespace)
            module_class = namespace.get(project.class_name)
            if not isinstance(module_class, type) or not issubclass(module_class, torch.nn.Module):
                raise ValueError(
                    f"{project.class_name} must be a torch.nn.Module class in this file."
                )
            model = module_class(**project.constructor).to(device=device).eval()
            dtype = getattr(torch, project.input.dtype)
            if dtype.is_floating_point:
                model = model.to(dtype=dtype)
            positional = [item for item in inputs if item.binding == "positional"]
            named = [item for item in inputs if item.binding == "keyword"]
            try:
                inspect.signature(model.forward).bind(
                    *[None for _ in positional], **{item.name: None for item in named}
                )
            except TypeError as exc:
                raise ValueError(
                    f"Forward inputs do not match {project.class_name}.forward: {exc}"
                ) from None
            values = {
                item.name: make_input(item.input, shapes, uploaded.get(item.name))
                for item in inputs
            }
            recorder = Recorder(
                project.code, filename, model, snapshot_dir=snapshot_dir, shapes=shapes
            )
            trace = recorder.trace
            trace.input_ids = [
                recorder.capture(
                    values[item.name], item.name, axes=item.input.axis_names, role="input"
                )
                for item in inputs
            ]
            with torch.no_grad(), recorder:
                output = model(
                    *[values[item.name] for item in positional],
                    **{item.name: values[item.name] for item in named},
                )
            trace.output_ids = [recorder.capture(t, "output") for t in tensors_in(output)]
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(frames) if f.filename == filename), None)
        if isinstance(exc, SyntaxError):
            line = exc.lineno
        trace.error = RunError(type=type(exc).__name__, message=str(exc), line=line)
    finally:
        if recorder:
            recorder.close()
    trace.stdout = stream.getvalue()
    trace.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return trace


if __name__ == "__main__":
    request_path, response_path = map(Path, sys.argv[1:3])
    project = ProjectDraft.model_validate_json(request_path.read_text())
    result = execute(
        project,
        Path(sys.argv[3]) if len(sys.argv) > 3 else None,
        Path(sys.argv[4]) if len(sys.argv) > 4 else None,
    )
    response_path.write_text(json.dumps(result.model_dump(), allow_nan=False))
