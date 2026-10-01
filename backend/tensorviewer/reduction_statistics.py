"""Complete mean/sum groups, read in bounded chunks from immutable snapshots."""

import math
import re
from pathlib import Path

import numpy as np

from .models import TensorState
from .operations.reduction import INTEGERS, reduction_spec
from .statistics import CHUNK_SIZE, snapshot_values


def supports_reference(x: TensorState, y: TensorState):
    return x.dtype == y.dtype or x.dtype in INTEGERS | {"bool"} and y.dtype == "int64"


def group_indices(shape, axes, output_shape, keepdim, output_index):
    """Yield at most CHUNK_SIZE logical indices, independent of physical strides."""
    coordinates = [0] * len(output_shape)
    remaining = output_index
    for i in range(len(output_shape) - 1, -1, -1):
        coordinates[i] = remaining % output_shape[i]
        remaining //= output_shape[i]
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    kept = [i for i in range(len(shape)) if i not in axes]
    base = sum(coordinates[i if keepdim else j] * strides[i] for j, i in enumerate(kept))
    sizes = [shape[i] for i in axes]
    count = math.prod(sizes)
    reduced_strides = [math.prod(sizes[i + 1 :]) for i in range(len(sizes))]
    for start in range(0, count, CHUNK_SIZE):
        terms = np.arange(start, min(start + CHUNK_SIZE, count), dtype=np.int64)
        indices = np.full(len(terms), base, dtype=np.int64)
        for axis, size, stride in zip(axes, sizes, reduced_strides):
            indices += (terms // stride % size) * strides[axis]
        yield indices


class NonFiniteGroup(Exception):
    pass


def snapshot_reduction(directory: Path, kind, args, inputs, outputs, output_index: int):
    spec = reduction_spec(kind, args, inputs, outputs)
    x, y = inputs[0], outputs[0]
    if type(output_index) is not int or not 0 <= output_index < y.numel:
        raise ValueError("Invalid reduction output index.")
    if not supports_reference(x, y):
        raise ValueError("Reference arithmetic does not emulate dtype conversions.")
    flat = snapshot_values(directory, x)
    integer = x.dtype in INTEGERS | {"bool"}

    def chunks():
        for indices in group_indices(x.shape, spec["axes"], y.shape, spec["keepdim"], output_index):
            yield flat[indices] if isinstance(flat, np.ndarray) else [flat[int(i)] for i in indices]

    def floating_values(divisor):
        for chunk in chunks():
            values = np.asarray(chunk, dtype=np.float64)
            if not np.isfinite(values).all():
                raise NonFiniteGroup()
            for value in values:
                yield float(value) / divisor

    def integer_values():
        info = None if x.dtype == "bool" else np.iinfo(x.dtype)
        low, high = (0, 1) if info is None else (info.min, info.max)
        for chunk in chunks():
            for value in chunk:
                if isinstance(value, (int, np.integer, bool, np.bool_)):
                    number = int(value)
                elif isinstance(value, str) and re.fullmatch(r"-?\d{1,20}", value):
                    number = int(value)
                elif (
                    isinstance(value, float)
                    and math.isfinite(value)
                    and value.is_integer()
                    and abs(value) < 2**53
                ):
                    number = int(value)
                else:
                    raise ValueError("Invalid integer snapshot value.")
                if not low <= number <= high:
                    raise ValueError("Snapshot value is outside its recorded dtype.")
                yield number

    if integer:
        total = sum(integer_values())  # Python integers cannot wrap at int64 boundaries.
        encoded = str(total) if abs(total) > 2**53 - 1 else total
        return {"status": "ok", "sum": encoded, "result": encoded}

    # Validate the full group and calculate its mean first. Scaling before fsum
    # keeps same-sign extremes bounded even when their unscaled total overflows.
    try:
        mean = math.fsum(floating_values(spec["size"]))
    except NonFiniteGroup:
        return {"status": "non_finite"}
    except OverflowError:
        return {"status": "overflow"}
    try:
        total = math.fsum(floating_values(1))
    except NonFiniteGroup:
        return {"status": "non_finite"}
    except OverflowError:
        total = None
    result = mean if kind == "mean" else total
    if result is None or not math.isfinite(result):
        return {"status": "overflow"}
    return {"status": "ok", "sum": total, "result": result}
