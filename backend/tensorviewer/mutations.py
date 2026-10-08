"""Observe tensor writes separately from returned values and shared version counters."""

from typing import NamedTuple

import torch


# A plain tuple underneath: the recorder reads every live tensor around every
# call, so building and comparing these must be cheap.
class TensorObservation(NamedTuple):
    version: int
    storage: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    dtype: torch.dtype

    @classmethod
    def read(cls, tensor):
        return cls(
            tensor._version,
            tensor.untyped_storage()._cdata,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.dtype,
        )

    def layout(self):
        return self.storage, self.shape, self.strides, self.offset, self.dtype


# These operations can bump the shared version counter while changing only the
# receiving tensor's layout/binding. Other views retain their layout and values.
METADATA_OPERATIONS = {
    "transpose_",
    "t_",
    "squeeze_",
    "unsqueeze_",
    "as_strided_",
    "swapaxes_",
    "swapdims_",
    "set_",
    "resize_",
    "resize_as_",
    "detach_",
}


def affected_tensors(kind, before, after, operands, out=(), inplace=False):
    """Return tensor object IDs and effects; alias effects are storage-level, not cell claims."""
    changed = {key for key in before if before[key] != after[key]}
    if kind in METADATA_OPERATIONS:
        return {key: "metadata" for key in changed if before[key].layout() != after[key].layout()}
    targets = set(out)
    if (kind.endswith("_") and not kind.endswith("__")) or kind == "__setitem__" or inplace:
        targets.update(operands[:1])
    if not targets:
        targets.update(key for key in operands if key in changed)
    targets &= changed
    # A version counter may remain shared even after a tensor is rebound to
    # different storage. Only storage used by actual written operands spreads
    # the effect; a version bump on an unrelated storage is not a value write.
    stores = {before[key].storage for key in targets}
    return {
        key: "write" if key in targets else "alias"
        for key in before
        if key in targets or before[key].storage in stores
    }
