"""One trusted local code execution per process. This is not a security sandbox."""

import contextlib
import importlib.metadata
import inspect
import io
import json
import math
import platform
import sys
import tempfile
import time
import traceback
from math import prod
from pathlib import Path

import numpy as np
import torch

from .input_files import load_array
from .layers import layer_stack
from .models import (
    CausalTrace,
    CausalTraceJob,
    GradientFlow,
    GradientRequest,
    InputSpec,
    Knockout,
    KnockoutSweep,
    LearnStep,
    LensState,
    LogitLens,
    ProjectDraft,
    RunError,
    Sensitivity,
    SensitivityRequest,
    SweepResult,
    Timings,
    Trace,
)
from .samples import sample_image, token_ids
from .source_projects import project_namespace
from .tracing import Recorder, tensors_in
from .weights import check_compatibility, load_weights


class LimitedOutput(io.StringIO):
    def write(self, text):
        remaining = max(0, 8000 - self.tell())
        super().write(text[:remaining])
        return len(text)


def make_input(spec: InputSpec, shapes: bool, uploaded=None) -> torch.Tensor:
    tensor = generate_input(spec, shapes, uploaded)
    if spec.precision and tensor.dtype.is_floating_point:
        tensor = tensor.to(getattr(torch, spec.precision))
    if not spec.edits or shapes:
        return tensor
    # A what-if: the same input with some cells set. Setting them draws no
    # random numbers, so the model's weights stay as they were.
    tensor = tensor.clone().contiguous()
    flat = tensor.view(-1)
    for edit in spec.edits:
        flat[edit.index] = edit.value
    return tensor


def generate_input(spec: InputSpec, shapes: bool, uploaded=None) -> torch.Tensor:
    dtype = getattr(torch, spec.dtype)
    if shapes:
        return torch.empty(spec.shape, dtype=dtype, device="meta")
    if spec.generator == "uploaded":
        return torch.from_numpy(uploaded)
    if spec.generator == "image":
        return sample_image(spec.shape, dtype)
    if spec.generator == "text":
        return torch.tensor([token_ids(spec.text)], dtype=dtype)
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


def sensitivity_of(
    recorder: Recorder,
    trace: Trace,
    x: torch.Tensor,
    input_id: str,
    target: SensitivityRequest,
) -> Sensitivity:
    """How much one recorded result cell moves per unit change of each input cell."""
    live = {tensor_id: tensor for tensor, _, tensor_id in recorder.live.values()}
    tensor = live.get(target.tensor_id)
    if tensor is None:
        raise ValueError(
            "This state was written over in place later in the run; choose its latest state."
        )
    if target.index >= tensor.numel() or not tensor.dtype.is_floating_point:
        raise ValueError("Choose a cell of a floating-point result.")
    cell = tensor.reshape(-1)[target.index]
    if x.dtype.is_floating_point:
        kind, sources = "gradient", [x]
    else:
        # Token ids have no gradient: ask instead about the embedding each id
        # was looked up as, a vector per id, from every step that read them.
        kind = "embedding"
        sources = [
            live[op.outputs[0]]
            for op in trace.operations
            if input_id in op.inputs
            and op.outputs
            and op.outputs[0] in live
            and trace.tensors[op.outputs[0]].shape[:-1] == list(x.shape)
            and live[op.outputs[0]].dtype.is_floating_point
        ]
        if not sources:
            raise ValueError(
                "An integer input needs an embedding that reads it before its sensitivity can be found."
            )
    if cell.requires_grad and any(source.requires_grad for source in sources):
        found = torch.autograd.grad(
            cell, [s for s in sources if s.requires_grad], allow_unused=True
        )
        grads = iter(found)
        gradients = [(next(grads) if source.requires_grad else None) for source in sources]
    else:
        # Nothing this cell computes from depends on the input.
        gradients = [None for _ in sources]
    gradients = [
        torch.zeros_like(source) if gradient is None else gradient
        for source, gradient in zip(sources, gradients)
    ]
    if kind == "gradient":
        values = gradients[0].detach().reshape(-1)
    else:
        values = sum(g.detach().to(torch.float64).norm(dim=-1) for g in gradients).reshape(-1)
    value = float(cell.detach())
    return Sensitivity(
        tensor_id=target.tensor_id,
        index=target.index,
        input_id=input_id,
        kind=kind,
        values=[v if math.isfinite(v) else None for v in values.tolist()],
        value=value if math.isfinite(value) else None,
    )


def gradient_flow(recorder: Recorder, target: GradientRequest) -> GradientFlow:
    """∂ target / ∂ every recorded tensor it depends on, as a size per tensor."""
    live = {tensor_id: tensor for tensor, _, tensor_id in recorder.live.values()}
    tensor = live.get(target.tensor_id)
    if tensor is None:
        raise ValueError(
            "This state was written over in place later in the run; choose its latest state."
        )
    if not tensor.dtype.is_floating_point:
        raise ValueError("Choose a floating-point result.")
    if target.index is not None and target.index >= tensor.numel():
        raise ValueError("That cell is outside the result.")
    value = tensor.sum() if target.index is None else tensor.reshape(-1)[target.index]
    found = [
        (tensor_id, each)
        for tensor_id, each in live.items()
        if each.dtype.is_floating_point and each.requires_grad
    ]
    norms: dict[str, float | None] = {}
    if value.requires_grad and found:
        grads = torch.autograd.grad(value, [each for _, each in found], allow_unused=True)
        for (tensor_id, _), grad in zip(found, grads):
            if grad is not None:
                size = float(grad.detach().to(torch.float64).norm())
                norms[tensor_id] = size if math.isfinite(size) else None
    return GradientFlow(tensor_id=target.tensor_id, index=target.index, norms=norms)


def learned_value(recorder: Recorder, step: LearnStep) -> torch.Tensor:
    """The value a training step aims at, as this pass computed it."""
    live = {tensor_id: tensor for tensor, _, tensor_id in recorder.live.values()}
    tensor = live.get(step.tensor_id)
    if tensor is None or step.index >= tensor.numel():
        raise ValueError("The value to learn on is not in this run's latest states.")
    return tensor if step.sentence else tensor.reshape(-1)[step.index]


def objective(recorder: Recorder, step: LearnStep, ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    """What a training step raises, and the number its curve reports.

    For one value: the value (or its log). For a sentence: minus the mean
    cross-entropy of each word's prediction of the word after it, reported
    as the loss itself.
    """
    value = learned_value(recorder, step)
    if not step.sentence:
        seen = float(value.detach())
        if step.log:
            if not seen > 0:
                raise ValueError("Only a positive value has a logarithm to learn on.")
            value = value.log()
        return value * step.direction, seen
    if ids.dtype.is_floating_point or list(value.shape[:-1]) != list(ids.shape):
        raise ValueError(
            "Training on the sentence needs scores over the vocabulary at each of the input's tokens."
        )
    if ids.shape[-1] < 2 or int(ids.max()) >= value.shape[-1]:
        raise ValueError("The sentence needs two tokens, all inside the vocabulary.")
    scores = value[..., :-1, :]
    following = ids[..., 1:]
    if step.log:
        chosen = scores.gather(-1, following.unsqueeze(-1)).squeeze(-1)
        loss = -chosen.clamp_min(1e-30).log().mean()
    else:
        loss = torch.nn.functional.cross_entropy(
            scores.reshape(-1, scores.shape[-1]).float(), following.reshape(-1)
        )
    return -loss, float(loss.detach())


def learn_step(
    model: torch.nn.Module,
    record,
    step: LearnStep,
    ids: torch.Tensor,
) -> list[float | None]:
    """Moves every weight `step.steps` gradient steps toward raising (or
    lowering) a value, and returns the value before each step.

    `record` runs the model once under a fresh recorder and returns it, so the
    value is found by the id the recorded run gave it. Each pass draws the same
    random numbers the original run did.
    """
    rng = torch.get_rng_state()
    curve: list[float | None] = []
    weights = [p for p in model.parameters() if p.requires_grad]
    for _ in range(step.steps):
        torch.set_rng_state(rng)
        with tempfile.TemporaryDirectory(prefix="tensorviewer-learn-") as scratch:
            with torch.enable_grad():
                recorder = record(Path(scratch))
            try:
                value, seen = objective(recorder, step, ids)
                curve.append(seen if math.isfinite(seen) else None)
                if not value.requires_grad or not weights:
                    raise ValueError(
                        "No weight moves this value, so a training step changes nothing."
                    )
                grads = torch.autograd.grad(value, weights, allow_unused=True)
                with torch.no_grad():
                    for weight, grad in zip(weights, grads):
                        if grad is not None:
                            weight.add_(grad, alpha=step.rate)
            finally:
                recorder.close()
    # The recorded run draws the same random numbers the original did.
    torch.set_rng_state(rng)
    return curve


def patch_values(rule: Knockout | KnockoutSweep | None) -> torch.Tensor | None:
    """The other run's result a patch puts in, as the backend wrote it."""
    if rule is None or rule.mode != "patch":
        return None
    if not rule.patch_path:
        raise ValueError("The run to patch from has no recorded values for that step.")
    return torch.from_numpy(np.array(np.load(rule.patch_path, allow_pickle=False)))


def finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def knockout_sweep(record, request: KnockoutSweep) -> SweepResult:
    """Knocks out each slice of one step's result along an axis in turn, and
    measures how far the model's first output moves each time."""
    rng = torch.get_rng_state()
    patch = patch_values(request)

    def output_of(knockout: Knockout | None):
        torch.set_rng_state(rng)
        with tempfile.TemporaryDirectory(prefix="tensorviewer-sweep-") as scratch:
            made = record(Path(scratch), knockout=knockout, patch=patch)
            try:
                if knockout and not made.knocked:
                    raise ValueError(f"The run never reached step {request.step + 1}.")
                if not made.results:
                    raise ValueError("The model returned no tensor to measure.")
                return made.results[0].detach().to(torch.float64), made.trace
            finally:
                made.close()

    baseline, trace = output_of(None)
    if request.step >= len(trace.operations):
        raise ValueError(f"The run has {len(trace.operations)} steps, not {request.step + 1}.")
    outputs = trace.operations[request.step].outputs
    if request.output >= len(outputs):
        raise ValueError(f"Step {request.step + 1} has no result {request.output + 1}.")
    shape = trace.tensors[outputs[request.output]].shape
    if request.axis >= len(shape):
        raise ValueError(f"Step {request.step + 1}'s result {shape} has no axis {request.axis}.")
    if shape[request.axis] > 64:
        raise ValueError(
            f"A sweep knocks out up to 64 slices; this axis has {shape[request.axis]}."
        )
    scale = float(baseline.norm())
    flat = baseline.reshape(-1)
    cell = int(flat.nan_to_num(nan=-math.inf).argmax()) if flat.numel() else None
    result = SweepResult(
        step=request.step,
        axis=request.axis,
        mode=request.mode,
        cell=cell,
        cell_value=finite(float(flat[cell])) if cell is not None else None,
    )
    for index in range(shape[request.axis]):
        knockout = Knockout(
            step=request.step,
            output=request.output,
            mode=request.mode,
            axis=request.axis,
            index=index,
            patch_from=request.patch_from,
        )
        output, _ = output_of(knockout)
        if output.shape != baseline.shape:
            result.effects.append(None)
            result.cell_values.append(None)
            continue
        moved = float((output - baseline).norm())
        result.effects.append(finite(moved / scale) if scale else finite(moved))
        result.cell_values.append(
            finite(float(output.reshape(-1)[cell])) if cell is not None else None
        )
    return result


def step_timings(record, passes: int) -> Timings:
    """Each step's median time over `passes` passes, after a warm-up pass.

    The warm-up pass sets up kernels and caches, and its steps are the ones
    timed. Every pass draws the random numbers the recorded run did; a pass
    whose steps differ from the warm-up's is not counted.
    """
    rng = torch.get_rng_state()
    kinds: list[str] = []
    measured: list[list[float]] = []
    for at in range(passes + 1):
        torch.set_rng_state(rng)
        with tempfile.TemporaryDirectory(prefix="tensorviewer-time-") as scratch:
            made = record(Path(scratch))
            made.close()
        ops = made.trace.operations
        if at == 0:
            kinds = [op.kind for op in ops]
            measured = [[] for _ in ops]
            continue
        if [op.kind for op in ops] != kinds:
            continue
        for op in ops:
            if op.duration_us is not None:
                measured[op.index].append(op.duration_us)
    torch.set_rng_state(rng)
    return Timings(
        passes=passes,
        durations_us=[round(float(np.median(times)), 2) if times else None for times in measured],
    )


def top_predictions(output: torch.Tensor, probabilities: bool) -> list[list[tuple[int, float]]]:
    """The three likeliest ids at each position of the first batch item."""
    scores = output.detach().to(torch.float64)
    scores = scores.reshape(-1, scores.shape[-2], scores.shape[-1])[0]
    probs = scores if probabilities else scores.softmax(-1)
    values, ids = probs.topk(min(3, probs.shape[-1]), dim=-1)
    return [
        [(int(i), float(v)) for i, v in zip(row_ids, row_values)]
        for row_ids, row_values in zip(ids.tolist(), values.tolist())
    ]


def logit_lens(record) -> LogitLens:
    """Reads every layer's state with the model's own final layers.

    The last block's result is patched, in turn, with the state entering the
    first block and with each earlier block's result; the layers after the
    stack then turn each into predictions, as they do the real one.
    """
    rng = torch.get_rng_state()

    def run_once(**knockout):
        torch.set_rng_state(rng)
        with tempfile.TemporaryDirectory(prefix="tensorviewer-lens-") as scratch:
            made = record(Path(scratch), **knockout)
            made.close()
        return made

    base = run_once()
    trace = base.trace
    stack = layer_stack(trace)
    if not stack:
        raise ValueError("The model has no stack of repeated blocks to read layer by layer.")
    if not stack[0].inputs or not stack[-1].outputs or not base.results:
        raise ValueError("The block stack's input or output was not recorded.")
    output = base.results[0]
    if output.ndim < 2 or not output.dtype.is_floating_point:
        raise ValueError("The model's result is not scores over a vocabulary at each position.")
    final_id = stack[-1].outputs[0]
    maker = next((op for op in trace.operations if final_id in op.outputs), None)
    if maker is None:
        raise ValueError("No recorded step made the last block's result.")
    made_by = next((op for op in trace.operations if trace.output_ids[0] in op.outputs), None)
    probabilities = made_by is not None and made_by.kind == "softmax"
    live = {tensor_id: tensor for tensor, _, tensor_id in base.live.values()}
    states = [(f"before {stack[0].path}", stack[0].inputs[0])] + [
        (call.path, call.outputs[0]) for call in stack
    ]
    found = []
    for name, tensor_id in states:
        state = live.get(tensor_id)
        if tensor_id == final_id:
            top = top_predictions(output, probabilities)
        elif state is None or list(state.shape) != trace.tensors[final_id].shape:
            continue
        else:
            patched = run_once(
                knockout=Knockout(
                    step=maker.index,
                    output=maker.outputs.index(final_id),
                    mode="patch",
                    patch_from="logit-lens",
                ),
                patch=state.detach().clone(),
            )
            if not patched.knocked or not patched.results:
                continue
            top = top_predictions(patched.results[0], probabilities)
        found.append(LensState(name=name, tensor_id=tensor_id, top=top))
    torch.set_rng_state(rng)
    return LogitLens(states=found)


def causal_trace(record, job: CausalTraceJob) -> CausalTrace:
    """Patches each layer's state from the clean run into this one, one
    position at a time, and measures how much of the clean result returns."""
    rng = torch.get_rng_state()

    def output_of(**knockout) -> torch.Tensor:
        torch.set_rng_state(rng)
        with tempfile.TemporaryDirectory(prefix="tensorviewer-trace-") as scratch:
            made = record(Path(scratch), **knockout)
            made.close()
        if knockout and not made.knocked:
            raise ValueError("The run never reached a layer it was asked to patch.")
        if not made.results:
            raise ValueError("The model returned no tensor to measure.")
        return made.results[0].detach().to(torch.float64)

    corrupt = output_of()
    clean = torch.from_numpy(np.array(np.load(job.clean_path, allow_pickle=False))).to(
        torch.float64
    )
    if clean.shape != corrupt.shape:
        raise ValueError(
            f"The other run's result is {list(clean.shape)}, not {list(corrupt.shape)}."
        )
    gap = float((corrupt - clean).norm())
    if not gap:
        raise ValueError("Both runs give the same result: change an input in one of them first.")
    patches = [
        torch.from_numpy(np.array(np.load(state.patch_path, allow_pickle=False)))
        for state in job.states
    ]
    if not patches or patches[0].ndim < 2:
        raise ValueError("The layers' states have no position axis to trace along.")
    axis = patches[0].ndim - 2
    positions = patches[0].shape[axis]
    if len(patches) * positions > 600:
        raise ValueError(
            f"Tracing {len(patches)} layers at {positions} positions is more than 600 runs."
        )
    recovery = []
    for state, patch in zip(job.states, patches):
        row = []
        for index in range(positions):
            output = output_of(
                knockout=Knockout(
                    step=state.step,
                    output=state.output,
                    mode="patch",
                    axis=axis,
                    index=index,
                    patch_from=job.against,
                ),
                patch=patch,
            )
            row.append(finite(1 - float((output - clean).norm()) / gap))
        recovery.append(row)
    torch.set_rng_state(rng)
    return CausalTrace(
        against=job.against,
        states=[state.name for state in job.states],
        positions=positions,
        recovery=recovery,
    )


def execute(
    project: ProjectDraft,
    snapshot_dir: Path | None = None,
    input_dir: Path | None = None,
    weights_dir: Path | None = None,
    check_weights_only: bool = False,
    target: SensitivityRequest | GradientRequest | None = None,
    learn: LearnStep | None = None,
    sweep: KnockoutSweep | None = None,
    timing: int | None = None,
    lens: bool = False,
    tracing: CausalTraceJob | None = None,
) -> Trace | Sensitivity | GradientFlow | SweepResult | Timings | LogitLens | CausalTrace:
    """Runs a project and records its trace; with a target, that cell's sensitivity."""
    started = time.perf_counter()
    trace = Trace()
    result = None
    recorder = None
    stream = LimitedOutput()
    filename = "<tensorviewer-project>"
    source_files = {}
    try:
        torch.set_num_threads(1)
        torch.manual_seed(project.input.seed)
        learn = learn or project.input.learn
        shapes = project.capture_mode == "shapes" or check_weights_only
        device = "meta" if shapes else "cpu"
        weights = (
            load_weights(weights_dir, project.weights, metadata=shapes) if project.weights else None
        )
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
            project_namespace(project) as (namespace, source_files),
        ):
            module_class = namespace.get(project.class_name)
            if not isinstance(module_class, type) or not issubclass(module_class, torch.nn.Module):
                raise ValueError(
                    f"{project.class_name} must be a torch.nn.Module class in this file."
                )
            model = module_class(**project.constructor).to(device=device).eval()
            dtype = getattr(torch, project.input.dtype)
            # A what-if may compute in another precision than the input's.
            if project.input.precision:
                model = model.to(dtype=getattr(torch, project.input.precision))
            elif dtype.is_floating_point:
                model = model.to(dtype=dtype)
            if weights is not None:
                trace.weight_check = check_compatibility(model, weights)
                if not trace.weight_check.compatible:
                    raise ValueError(
                        "Checkpoint does not match this model. "
                        + " ".join(trace.weight_check.issues)
                    )
                try:
                    model.load_state_dict(weights, strict=True)
                except Exception as exc:
                    trace.weight_check.compatible = False
                    trace.weight_check.issues = [f"Could not load checkpoint: {exc}"]
                    raise
            if not check_weights_only:
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
                primary = values[project.input_name]
                if target and primary.dtype.is_floating_point:
                    primary.requires_grad_(True)

                knockout = project.input.knockout
                patch = patch_values(knockout)

                def record(directory, main=False, knockout=None, patch=None):
                    """One forward pass under a new recorder, ids as a run gives them.

                    The main pass's trace is the run's from the start, so a step
                    that fails keeps the steps recorded before it.
                    """
                    nonlocal recorder, trace
                    made = Recorder(
                        project.code,
                        filename,
                        model,
                        snapshot_dir=directory,
                        shapes=shapes,
                        source_files=source_files,
                        knockout=knockout,
                        patch=patch,
                    )
                    if main:
                        report = trace.weight_check
                        recorder, trace = made, made.trace
                        trace.weight_check = report
                    made.trace.input_ids = [
                        made.capture(
                            values[item.name], item.name, axes=item.input.axis_names, role="input"
                        )
                        for item in inputs
                    ]
                    with made:
                        output = model(
                            *[values[item.name] for item in positional],
                            **{item.name: values[item.name] for item in named},
                        )
                    made.results = list(tensors_in(output))
                    made.trace.output_ids = [made.capture(t, "output") for t in made.results]
                    return made

                # A trained what-if trains again first, so every look at it
                # sees the weights its run had.
                curve = learn_step(model, record, learn, primary) if learn else None
                if sweep:
                    with torch.no_grad():
                        result = knockout_sweep(record, sweep)
                    return result
                if timing:
                    with torch.no_grad():
                        result = step_timings(record, timing)
                    return result
                if lens:
                    with torch.no_grad():
                        result = logit_lens(record)
                    return result
                if tracing:
                    with torch.no_grad():
                        result = causal_trace(record, tracing)
                    return result
                with torch.enable_grad() if target else torch.no_grad():
                    record(snapshot_dir, main=True, knockout=knockout, patch=patch)
                if knockout and not recorder.knocked:
                    trace.warnings.append(
                        f"The run never reached step {knockout.step + 1}, so nothing was knocked out."
                    )
                if learn:
                    # The value after the last step, as the recorded run has it.
                    with torch.no_grad():
                        final = objective(recorder, learn, primary)[1]
                    trace.learn_curve = [*curve, final if math.isfinite(final) else None]
                if isinstance(target, GradientRequest):
                    result = gradient_flow(recorder, target)
                elif target:
                    result = sensitivity_of(recorder, trace, primary, trace.input_ids[0], target)
    except Exception as exc:
        source_files = getattr(exc, "_tensorviewer_source_files", source_files)
        frames = traceback.extract_tb(exc.__traceback__)
        frame = next(
            (f for f in reversed(frames) if f.filename == filename or f.filename in source_files),
            None,
        )
        line = frame.lineno if frame else None
        source_file = source_files.get(frame.filename, (None,))[0] if frame else None
        if isinstance(exc, SyntaxError):
            line = exc.lineno
            source_file = source_files.get(exc.filename, (source_file,))[0]
        trace.error = RunError(
            type=type(exc).__name__, message=str(exc), line=line, file=source_file
        )
    finally:
        if recorder:
            recorder.close()
    trace.stdout = stream.getvalue()
    trace.runtime = {
        "Python": platform.python_version(),
        **{
            d.metadata["Name"]: importlib.metadata.version(d.metadata["Name"])
            for d in importlib.metadata.distributions()
            if d.metadata.get("Name")
        },
    }
    trace.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    if tracing:
        return CausalTrace(
            against=tracing.against,
            error=trace.error
            or RunError(type="ValueError", message="The run could not be traced."),
        )
    if lens:
        return LogitLens(
            error=trace.error
            or RunError(type="ValueError", message="The run could not be read layer by layer.")
        )
    if timing:
        return Timings(
            passes=timing,
            error=trace.error or RunError(type="ValueError", message="The run could not be timed."),
        )
    if sweep:
        return SweepResult(
            step=sweep.step,
            axis=sweep.axis,
            mode=sweep.mode,
            error=trace.error or RunError(type="ValueError", message="The sweep measured nothing."),
        )
    if isinstance(target, GradientRequest):
        return result or GradientFlow(
            tensor_id=target.tensor_id,
            index=target.index,
            error=trace.error
            or RunError(type="ValueError", message="The run recorded no result to ask about."),
        )
    if target:
        return result or Sensitivity(
            tensor_id=target.tensor_id,
            index=target.index,
            input_id="",
            kind="gradient",
            values=[],
            error=trace.error
            or RunError(type="ValueError", message="The run recorded no result to ask about."),
        )
    return trace


if __name__ == "__main__":
    request_path, response_path = map(Path, sys.argv[1:3])
    project = ProjectDraft.model_validate_json(request_path.read_text())
    result = execute(
        project,
        Path(sys.argv[3]) if len(sys.argv) > 3 else None,
        Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None,
        Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None,
        len(sys.argv) > 6 and sys.argv[6] == "check",
        SensitivityRequest.model_validate_json(sys.argv[7])
        if len(sys.argv) > 7 and sys.argv[6] == "sensitivity"
        else GradientRequest.model_validate_json(sys.argv[7])
        if len(sys.argv) > 7 and sys.argv[6] == "gradients"
        else None,
        LearnStep.model_validate_json(sys.argv[7])
        if len(sys.argv) > 7 and sys.argv[6] == "learn"
        else None,
        KnockoutSweep.model_validate_json(sys.argv[7])
        if len(sys.argv) > 7 and sys.argv[6] == "sweep"
        else None,
        int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[6] == "timings" else None,
        len(sys.argv) > 6 and sys.argv[6] == "lens",
        CausalTraceJob.model_validate_json(sys.argv[7])
        if len(sys.argv) > 7 and sys.argv[6] == "trace"
        else None,
    )
    response_path.write_text(json.dumps(result.model_dump(), allow_nan=False))
