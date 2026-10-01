"""Full softmax denominators from bounded reads of saved logical snapshots."""

import math
from pathlib import Path

import numpy as np

from .operations.softmax import softmax_spec
from .statistics import CHUNK_SIZE, snapshot_values


def snapshot_softmax(directory: Path, args, inputs, outputs, group: int):
    spec = softmax_spec(args, inputs, outputs)
    if type(group) is not int or not 0 <= group < spec["groups"]:
        raise ValueError("Invalid softmax group.")
    flat = snapshot_values(directory, inputs[0])
    size, stride = spec["size"], spec["stride"]
    base = group // stride * size * stride + group % stride

    def chunks():
        for start in range(0, size, CHUNK_SIZE):
            indices = base + np.arange(start, min(start + CHUNK_SIZE, size)) * stride
            values = (
                flat[indices] if isinstance(flat, np.ndarray) else [flat[int(i)] for i in indices]
            )
            yield np.asarray(values, dtype=np.float64)

    maximum, masked = -math.inf, 0
    for values in chunks():
        if np.isnan(values).any() or np.isposinf(values).any():
            return {"status": "non_finite"}
        maximum = max(maximum, float(np.max(values)))
        masked += int(np.isneginf(values).sum())
    if masked == size:
        return {"status": "all_masked", "masked_count": masked}
    # Exp values are in [0, 1]. Overflow of x - maximum toward -inf correctly
    # contributes zero, including finite opposite float64 extremes.
    with np.errstate(over="ignore", under="ignore"):
        denominator = math.fsum(
            float(value) for values in chunks() for value in np.exp(values - maximum)
        )
    return {"status": "ok", "maximum": maximum, "denominator": denominator, "masked_count": masked}
