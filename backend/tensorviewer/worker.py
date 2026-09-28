"""One trusted local code execution per process. This is not a security sandbox."""

import contextlib
import io
import json
import sys
import time
import traceback
from math import prod
from pathlib import Path

import torch

from .models import ProjectDraft, RunError, Trace
from .tracing import Recorder, tensors_in


class LimitedOutput(io.StringIO):
    def write(self, text):
        remaining = max(0, 8000 - self.tell())
        super().write(text[:remaining])
        return len(text)


def execute(project: ProjectDraft) -> Trace:
    started = time.perf_counter()
    trace = Trace()
    recorder = None
    stream = LimitedOutput()
    filename = "<tensorviewer-project>"
    try:
        torch.set_num_threads(1)
        torch.manual_seed(project.input.seed)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            namespace = {"__name__": "tensorviewer_user_project"}
            exec(compile(project.code, filename, "exec"), namespace)
            module_class = namespace.get(project.class_name)
            if not isinstance(module_class, type) or not issubclass(module_class, torch.nn.Module):
                raise ValueError(
                    f"{project.class_name} must be a torch.nn.Module class in this file."
                )
            model = module_class(**project.constructor).cpu().eval()
            spec = project.input
            dtype = getattr(torch, spec.dtype)
            if dtype.is_floating_point:
                model = model.to(dtype=dtype)
            if spec.generator == "arange":
                x = torch.arange(prod(spec.shape), dtype=dtype).reshape(spec.shape)
            elif spec.generator == "random":
                x = torch.randn(spec.shape, dtype=dtype)
            else:
                x = getattr(torch, spec.generator)(spec.shape, dtype=dtype)
            recorder = Recorder(project.code, filename, model)
            trace = recorder.trace
            trace.input_ids = [recorder.capture(x, "x", axes=spec.axis_names, role="input")]
            with torch.no_grad(), recorder:
                output = model(x)
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
    result = execute(project)
    response_path.write_text(json.dumps(result.model_dump(), allow_nan=False))
