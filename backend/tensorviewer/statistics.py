"""Bounded, two-pass population statistics over immutable logical tensor groups."""

import math
from pathlib import Path

import numpy as np

from .models import PopulationStatistics, TensorState

CHUNK_SIZE = 16_384
MAX_ELEMENTS = 8_388_608


def snapshot_statistics(
    directory: Path, tensor: TensorState, start: int, count: int, eps: float
) -> PopulationStatistics:
    if (
        tensor.value_source == "shape"
        or not 0 < tensor.numel <= MAX_ELEMENTS
        or math.prod(tensor.shape) != tensor.numel
        or any(n < 1 for n in tensor.shape)
        or not 0 <= start < tensor.numel
        or not 0 < count <= tensor.numel - start
        or not math.isfinite(eps)
        or eps < 0
    ):
        raise ValueError("Invalid numeric group.")
    if tensor.value_source == "paged":
        data = np.load(directory / f"{tensor.id}.npy", mmap_mode="r", allow_pickle=False)
        dtype = "float32" if tensor.dtype == "bfloat16" else tensor.dtype
        if (
            list(data.shape) != tensor.shape
            or str(data.dtype) != dtype
            or not data.flags.c_contiguous
        ):
            raise ValueError("Snapshot does not match the captured tensor.")
        flat = data.reshape(-1)
    else:
        if len(tensor.values) != tensor.numel:
            raise ValueError("Incomplete inline snapshot.")
        flat = tensor.values

    def chunks():
        # Slice before conversion: temporary float64 arrays never contain a full
        # large tensor. Snapshots are saved in logical order, even for views.
        for offset in range(start, start + count, CHUNK_SIZE):
            yield np.asarray(
                flat[offset : min(offset + CHUNK_SIZE, start + count)], dtype=np.float64
            )

    with np.errstate(over="ignore", invalid="ignore"):
        origin = float(flat[start])
        means = []
        for chunk in chunks():
            if not np.isfinite(chunk).all():
                return PopulationStatistics(status="non_finite")
            # An offset keeps constant groups exact and preserves small
            # differences when every value has a large common magnitude.
            means.append(float(np.sum((chunk - origin) / count, dtype=np.float64)))
        try:
            mean = math.fsum([origin, *means])
            variance = math.fsum(
                float(np.sum(np.square(chunk - mean) / count, dtype=np.float64))
                for chunk in chunks()
            )
            denominator = math.sqrt(variance + eps)
        except (OverflowError, ValueError):
            return PopulationStatistics(status="overflow")
    if not all(math.isfinite(n) for n in (mean, variance, denominator)):
        return PopulationStatistics(status="overflow")
    return PopulationStatistics(status="ok", mean=mean, variance=variance, denominator=denominator)
