"""Who sees whom: which input slices each slice of a result depends on.

The recorded code runs again with gradients on. Each output slice along the
chosen axis is weighted at random and summed, so that no gradient cancels by
chance (a softmax's slice sums to 1 whatever its input), and differentiated
once. Where the gradient at an input slice is exactly zero, that output slice
does not read it: proved by autograd over the code that ran, not guessed from
its shapes.
"""

import math
import re

import torch

from .models import Dependencies, DependencyBreach, DependencyRequest, Operation, Trace

# A gradient smaller than this share of its row's largest counts as not read.
THRESHOLD = 1e-6
# Input slices per row; past this the matrix stops being a picture.
MAX_INPUTS = 1024
GENERIC = re.compile(r"^axis \d+$")


def gradient_sources(
    live: dict[str, torch.Tensor], trace: Trace, x: torch.Tensor, input_id: str
) -> tuple[str, list[torch.Tensor]]:
    """What to differentiate with respect to: the input itself, or for token
    ids (which have no gradient) the embedding each id was looked up as, from
    every step that read them. An embedding's leading axes are the ids'."""
    if x.dtype.is_floating_point:
        return "gradient", [x]
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
            "An integer input needs an embedding that reads it before a gradient can reach it."
        )
    return "embedding", sources


def gradients_at(value: torch.Tensor, sources: list[torch.Tensor]) -> list[torch.Tensor | None]:
    """∂ value / ∂ each source; None where nothing connects them."""
    wanted = [source for source in sources if source.requires_grad]
    if not value.requires_grad or not wanted:
        return [None for _ in sources]
    found = iter(torch.autograd.grad(value, wanted, retain_graph=True, allow_unused=True))
    return [next(found) if source.requires_grad else None for source in sources]


def slice_sizes(
    gradients: list[torch.Tensor | None], sources: list[torch.Tensor], axis: int, count: int
) -> torch.Tensor:
    """The largest |gradient| in each slice along `axis`, over every source."""
    largest = torch.zeros(count, dtype=torch.float64)
    for gradient, source in zip(gradients, sources):
        if gradient is None or source.numel() == 0:
            continue
        sizes = gradient.detach().abs().to(torch.float64).movedim(axis, 0).reshape(count, -1)
        found = sizes.amax(1)
        # NaN beats any size, so a non-finite gradient is not lost.
        largest = torch.where(found.isnan() | (found > largest), found, largest)
    return largest


def row_of(sizes: torch.Tensor) -> tuple[list[float | None], list[bool], list[bool]]:
    """One output slice's strength at each input slice, relative to its
    largest finite one; whether each is read, and whether exactly zero."""
    finite = sizes[sizes.isfinite()]
    top = float(finite.max()) if finite.numel() else 0.0
    strength, reads, zero = [], [], []
    for size in sizes.tolist():
        if not math.isfinite(size):
            strength.append(None)
            reads.append(True)
            zero.append(False)
            continue
        share = size / top if top > 0 else 0.0
        strength.append(share)
        reads.append(top > 0 and share > THRESHOLD)
        zero.append(size == 0)
    return strength, reads, zero


def expected(output: int, input: int, expect: str | None) -> bool:
    if expect == "diagonal":
        return output == input
    if expect == "causal":
        return input <= output
    return True


def input_axis_for(request: DependencyRequest, result, source) -> int:
    """The input axis matching the result's: as asked, else of the same name,
    else the one axis of the same size."""
    rank = len(source.shape)
    if request.input_axis is not None:
        if request.input_axis >= rank:
            raise ValueError(f"The input has {rank} axes; choose one of them.")
        return request.input_axis
    name = result.axes[request.axis]
    if not GENERIC.match(name) and source.axes.count(name) == 1:
        return source.axes.index(name)
    size = result.shape[request.axis]
    same = [axis for axis, length in enumerate(source.shape) if length == size]
    if len(same) == 1:
        return same[0]
    raise ValueError("Choose the input axis to compare the result's axis with.")


def matching_axis(state, name: str | None, size: int) -> int | None:
    """The one axis of a recorded tensor that holds the same positions: the
    axis of that name and size, or, where no axis has the name, the one of
    that size whose name says nothing (`axis 2`)."""
    if name is not None and name in state.axes:
        named = [
            axis
            for axis, (each, length) in enumerate(zip(state.axes, state.shape))
            if each == name and length == size
        ]
        return named[0] if len(named) == 1 else None
    same = [axis for axis, length in enumerate(state.shape) if length == size]
    if len(same) == 1 and GENERIC.match(state.axes[same[0]]):
        return same[0]
    return None


def steps_between(
    trace: Trace,
    live: dict[str, torch.Tensor],
    starts: set[str],
    target_id: str,
    target_axis: int,
    name: str | None,
    size: int,
) -> list[tuple[Operation, str, int]]:
    """The recorded results that read the input and that the target reads,
    in the order they ran, each with its axis holding the target's positions;
    the target itself last."""
    made = {output: op for op in trace.operations for output in op.outputs}
    maker = made.get(target_id)
    if maker is None:
        return []
    ancestors: set[str] = set()
    waiting = [target_id]
    while waiting:
        tensor_id = waiting.pop()
        if tensor_id in ancestors:
            continue
        ancestors.add(tensor_id)
        if tensor_id in made:
            waiting.extend(made[tensor_id].inputs)
    reached = set(starts)
    found = []
    for op in trace.operations:
        if op.index > maker.index:
            break
        if not any(each in reached for each in op.inputs):
            continue
        reached.update(op.outputs)
        for output in op.outputs:
            if output == target_id or output not in ancestors or output not in live:
                continue
            if not live[output].dtype.is_floating_point:
                continue
            axis = matching_axis(trace.tensors[output], name, size)
            if axis is not None:
                found.append((op, output, axis))
    return [*found, (maker, target_id, target_axis)]


def dependencies_of(
    recorder, trace: Trace, x: torch.Tensor, input_id: str, request: DependencyRequest
) -> Dependencies:
    """Which input slices each slice of a recorded result reads, along one axis
    pair, held to an expected pattern; a breach is traced to its first step."""
    live = {tensor_id: tensor for tensor, _, tensor_id in recorder.live.values()}
    tensor = live.get(request.tensor_id)
    if tensor is None:
        raise ValueError(
            "This state was written over in place later in the run; choose its latest state."
        )
    if not tensor.dtype.is_floating_point:
        raise ValueError("Choose a floating-point result: an integer one has no gradient.")
    if request.axis >= tensor.ndim:
        raise ValueError(f"The result has {tensor.ndim} axes; choose one of them.")
    result = trace.tensors[request.tensor_id]
    source = trace.tensors[input_id]
    input_axis = input_axis_for(request, result, source)
    count = x.shape[input_axis]
    if count > MAX_INPUTS:
        raise ValueError(
            f"The input axis has {count} positions; at most {MAX_INPUTS} are compared."
        )
    kind, sources = gradient_sources(live, trace, x, input_id)
    size = tensor.shape[request.axis]
    rows = min(size, request.limit)
    # Seeded, so the same question gets the same answer.
    generator = torch.Generator().manual_seed(0)

    def reads_of(each: torch.Tensor, axis: int, at: int) -> tuple:
        weights = torch.randn(
            each.select(axis, at).shape, generator=generator, dtype=torch.float64
        ).to(each.dtype)
        value = (each.select(axis, at) * weights).sum()
        gradients = gradients_at(value, sources)
        return row_of(slice_sizes(gradients, sources, input_axis, count))

    answer = Dependencies(
        tensor_id=request.tensor_id,
        axis=request.axis,
        input_id=input_id,
        input_axis=input_axis,
        kind=kind,
        size=size,
        threshold=THRESHOLD,
        expect=request.expect,
    )
    for at in range(rows):
        strength, reads, zero = reads_of(tensor, request.axis, at)
        answer.strength.append(strength)
        answer.reads.append(reads)
        answer.zero.append(zero)
    answer.violations = [
        (at, read_at)
        for at, row in enumerate(answer.reads)
        for read_at, read in enumerate(row)
        if read and not expected(at, read_at, request.expect)
    ][:64]
    if not answer.violations:
        return answer
    # Bisect the steps between the input and the result for the first whose
    # result already reads what the first violation reads.
    output, wrong = answer.violations[0]
    name = result.axes[request.axis]
    if GENERIC.match(name):
        name = None if GENERIC.match(source.axes[input_axis]) else source.axes[input_axis]
    steps = steps_between(trace, live, {input_id}, request.tensor_id, request.axis, name, size)
    if not steps:
        return answer

    def breaks(at: int) -> bool:
        op, tensor_id, axis = steps[at]
        return reads_of(live[tensor_id], axis, output)[1][wrong]

    clean, broken = -1, len(steps) - 1
    while broken - clean > 1:
        middle = (clean + broken) // 2
        if breaks(middle):
            broken = middle
        else:
            clean = middle
    op, tensor_id, axis = steps[broken]
    answer.breach = DependencyBreach(
        output=output,
        input=wrong,
        operation_id=op.id,
        step=op.index,
        tensor_id=tensor_id,
        axis=axis,
        after=steps[clean][0].id if clean >= 0 else None,
    )
    return answer
