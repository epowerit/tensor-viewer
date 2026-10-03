"""Immutable dense snapshots; only requested logical elements cross the API."""

import math
from pathlib import Path

import numpy as np
import torch


def json_value(value):
    value = value.item() if hasattr(value, "item") else value
    if not math.isfinite(value) or isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    return value


def save_snapshot(directory: Path, tensor_id: str, tensor: torch.Tensor):
    directory.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.to(torch.float32)
    # Written now, so subsequent in-place operations cannot alter this state.
    np.save(directory / f"{tensor_id}.npy", value.numpy(), allow_pickle=False)


def read_snapshot(directory: Path, tensor_id: str, indices: list[int]):
    data = np.load(directory / f"{tensor_id}.npy", mmap_mode="r", allow_pickle=False)
    flat = data.reshape(-1)
    return [json_value(flat[index]) for index in indices]


def snapshot_array(directory: Path, tensor_id: str) -> np.ndarray:
    """A whole snapshot, memory-mapped rather than read into memory."""
    return np.load(directory / f"{tensor_id}.npy", mmap_mode="r", allow_pickle=False)
