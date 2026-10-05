"""A recorded run as a pytest file: the model, its inputs, and what it made.

The test rebuilds the run's inputs, runs the model under forward hooks, and
checks the shape of every module's output, call by call, and the model's
result: its values to float rounding when they are small enough to write
down, or their statistics when not. It needs PyTorch and pytest, not
TensorViewer.
"""

import math
import pprint
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from .models import Run

# Tensors up to this many elements are written into the test value by value.
WRITTEN = 4096


class ExportError(ValueError):
    """Why a run cannot become a test."""


def literal(value, indent: int = 0) -> str:
    """A Python literal for a value, wrapped to the line length."""
    text = pprint.pformat(value, width=88 - indent, compact=True, sort_dicts=False)
    return text.replace("\n", "\n" + " " * indent)


def number(value) -> float | int:
    """A recorded value as a number; NaN, infinities, and big integers come as text."""
    if isinstance(value, str):
        return int(value) if re.fullmatch(r"-?\d+", value) else float(value)
    return value


def finite_or_text(value):
    """A number Python can write as a literal: NaN and infinities stay text."""
    return value if not isinstance(value, float) or math.isfinite(value) else str(value)


def recorded_values(run: Run, snapshot_dir: Path, tensor_id: str) -> np.ndarray | None:
    """A recorded state's values, inline or from its snapshot."""
    tensor = run.trace.tensors[tensor_id]
    if tensor.value_source == "inline" and len(tensor.values) == tensor.numel:
        return np.array([number(v) for v in tensor.values]).reshape(tensor.shape)
    path = snapshot_dir / run.id / f"{tensor_id}.npy"
    if path.exists():
        return np.load(path, allow_pickle=False)
    return None


def described(run: Run, snapshot_dir: Path, tensor_id: str, *, name: str) -> dict:
    """What the test needs to rebuild or check one recorded tensor."""
    tensor = run.trace.tensors[tensor_id]
    entry: dict = {"shape": tensor.shape, "dtype": tensor.dtype}
    values = recorded_values(run, snapshot_dir, tensor_id)
    if values is None:
        raise ExportError(f"The run recorded {name}'s shape only, not its values.")
    if tensor.numel <= WRITTEN:
        entry["values"] = [finite_or_text(v) for v in values.reshape(-1).tolist()]
    else:
        floats = values.astype(np.float64)
        entry["stats"] = {
            "sum": float(np.nansum(floats)),
            "mean": float(np.nanmean(floats)),
            "std": float(np.nanstd(floats)),
            "min": float(np.nanmin(floats)),
            "max": float(np.nanmax(floats)),
        }
    return entry


def module_shapes(run: Run) -> dict[str, list[list[list[int]]]]:
    """Every module's output shapes, per call, in the order the calls ended."""
    trace = run.trace
    shapes: dict[str, list[list[list[int]]]] = {}
    for call in sorted(trace.module_calls, key=lambda each: (each.end_index, each.start_index)):
        shapes.setdefault(call.path, []).append(
            [trace.tensors[t].shape for t in call.outputs if t in trace.tensors]
        )
    return shapes


def pytest_source(run: Run, snapshot_dir: Path) -> str:
    """The run as a self-contained pytest file."""
    project = run.project
    trace = run.trace
    if trace.error:
        raise ExportError("Only a run that finished can become a test.")
    if project.capture_mode == "shapes":
        raise ExportError("A shapes-only run recorded no values to check; record values first.")
    if project.blueprint:
        raise ExportError("A diagram project's run cannot be exported yet; export from code.")
    if project.files or project.entry_path != "model.py" or project.import_root != ".":
        raise ExportError("Only single-file projects can be exported yet.")
    if project.weights:
        raise ExportError("A run that loaded saved weights needs its checkpoint; not exported yet.")
    if not trace.output_ids:
        raise ExportError("The model returned no tensor to check.")
    inputs = []
    for item, tensor_id in zip(project.forward_inputs, trace.input_ids):
        if trace.tensors[tensor_id].numel > WRITTEN:
            raise ExportError(
                f"The input {item.name} has more than {WRITTEN} values to write into a test."
            )
        entry = described(run, snapshot_dir, tensor_id, name=item.name)
        inputs.append({"name": item.name, "binding": item.binding, **entry})
    outputs = [
        described(run, snapshot_dir, tensor_id, name="the result") for tensor_id in trace.output_ids
    ]
    root = project.class_name
    shapes = module_shapes(run)
    precision = project.input.precision or (
        project.input.dtype if project.input.dtype.startswith("float") else None
    )
    stem = re.sub(r"\W+", "_", root).strip("_").lower() or "model"
    when = datetime.now().strftime("%Y-%m-%d")
    return f'''"""Regression test for {root}, written by TensorViewer from run {run.id[:8]} ({when}).

It rebuilds the run's inputs, runs the model, and checks the shape of every
module's output, call by call, and the model's result against what the run
recorded. Run it with:

    pytest test_{stem}_{run.id[:8]}.py
"""

# ---- The model, as the run recorded it ---------------------------------------

{project.code.rstrip()}


# ---- What the run recorded ---------------------------------------------------

import math as _math  # noqa: E402

import torch as _torch  # noqa: E402

_SEED = {project.input.seed}
_CONSTRUCTOR = {literal(project.constructor)}
_PRECISION = {precision!r}
_INPUTS = {literal(inputs)}
_MODULE_SHAPES = {literal(shapes)}
_OUTPUTS = {literal(outputs)}


def _tensors(value):
    """The tensors in a module's result, in order, as the run counted them."""
    if isinstance(value, _torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _number(value):
    return float(value) if isinstance(value, str) else value


def _close(actual, expected, dtype):
    """Values to float rounding: float32 to 1e-4, float64 to 1e-10, others exactly."""
    tolerance = 1e-10 if dtype == "float64" else 1e-4 if dtype.startswith("float") else 0.0
    for at, (a, b) in enumerate(zip(actual, expected)):
        b = _number(b)
        if isinstance(b, float) and _math.isnan(b):
            assert _math.isnan(a), f"value {{at}}: expected NaN, got {{a}}"
            continue
        assert a == b or abs(a - b) <= tolerance * (1 + abs(b)), (
            f"value {{at}}: expected {{b}}, got {{a}}"
        )


def _build_inputs():
    positional, named = [], {{}}
    for item in _INPUTS:
        dtype = getattr(_torch, item["dtype"])
        value = _torch.tensor([_number(v) for v in item["values"]], dtype=dtype)
        value = value.reshape(item["shape"])
        if _PRECISION and value.dtype.is_floating_point:
            value = value.to(getattr(_torch, _PRECISION))
        if item["binding"] == "keyword":
            named[item["name"]] = value
        else:
            positional.append(value)
    return positional, named


def test_{stem}_matches_the_recorded_run():
    _torch.manual_seed(_SEED)
    model = {root}(**_CONSTRUCTOR).eval()
    if _PRECISION:
        model = model.to(getattr(_torch, _PRECISION))
    positional, named = _build_inputs()
    seen = {{}}

    def record(name):
        def hook(module, args, output):
            seen.setdefault(name, []).append([list(t.shape) for t in _tensors(output)])

        return hook

    hooks = [
        module.register_forward_hook(record(name or {root!r}))
        for name, module in model.named_modules()
    ]
    try:
        with _torch.no_grad():
            result = model(*positional, **named)
    finally:
        for hook in hooks:
            hook.remove()

    for name, expected in _MODULE_SHAPES.items():
        assert seen.get(name) == expected, (
            f"{{name}} made {{seen.get(name)}}; the run recorded {{expected}}"
        )

    outputs = list(_tensors(result))
    assert [list(t.shape) for t in outputs] == [o["shape"] for o in _OUTPUTS]
    for output, expected in zip(outputs, _OUTPUTS):
        if "values" in expected:
            _close(output.reshape(-1).tolist(), expected["values"], expected["dtype"])
        else:
            values = output.double()
            stats = expected["stats"]
            for name, actual in [
                ("mean", values.nanmean()),
                ("std", values[~values.isnan()].std(unbiased=False)),
                ("min", values[~values.isnan()].min()),
                ("max", values[~values.isnan()].max()),
            ]:
                assert abs(float(actual) - stats[name]) <= 1e-3 * (1 + abs(stats[name])), (
                    f"{{name}}: expected {{stats[name]}}, got {{float(actual)}}"
                )
'''
