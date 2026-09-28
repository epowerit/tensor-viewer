from math import prod

import torch

from ..models import Lesson, TensorState


def describe_layout(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    before, after = inputs[0], outputs[0]
    rank = len(before.shape)
    order = None
    mapping = None
    if kind in {"reshape", "view", "flatten", "contiguous", "clone", "squeeze", "unsqueeze"}:
        if before.numel == after.numel:
            mapping = list(range(after.numel))
    if kind == "permute":
        order = [int(d) % rank for d in args["dims"]]
    elif kind in {"transpose", "t"}:
        order = list(range(rank))
        a, b = (0, 1) if kind == "t" else (args["dim0"] % rank, args["dim1"] % rank)
        if rank > 1:
            order[a], order[b] = order[b], order[a]
    if order is not None:
        mapping = torch.arange(prod(before.shape)).reshape(before.shape).permute(order).flatten().tolist()
    if kind == "unfold":
        mapping = torch.arange(before.numel).reshape(before.shape).unfold(
            args["dimension"], args["size"], args["step"]
        ).reshape(-1).tolist()
    shared = before.storage_id == after.storage_id
    storage_note = "This output shares storage with the input." if shared else "This output uses separate storage."
    if kind in {"permute", "transpose", "t"}:
        return Lesson(
            title="Reorder the axes", summary="Change the order of dimensions while keeping the same values.",
            detail=f"Output axes read input axes in the order {order}. Select an output element to find its original coordinates. {storage_note}",
            category="layout", interaction="mapping", mapping=mapping, axis_order=order,
        )
    if kind in {"contiguous", "clone"}:
        return Lesson(
            title="Make values contiguous" if kind == "contiguous" else "Copy the tensor",
            summary="The logical shape and values stay the same; inspect the storage and strides.",
            detail=("The requested memory layout is already satisfied; no copy was needed. " if shared else "PyTorch copied the values into new storage. ") + storage_note,
            category="memory", interaction="mapping", mapping=mapping,
        )
    if kind == "unfold":
        return Lesson(
            title="Extract sliding windows", summary="Expose each window as another dimension.",
            detail=f"A window of {args['size']} moves by {args['step']} along axis {args['dimension']}. Overlapping windows may reference the same input element. {storage_note}",
            category="layout", interaction="mapping", mapping=mapping,
        )
    return Lesson(
        title="Regroup the elements", summary="Keep the logical element sequence and change its dimension boundaries.",
        detail=f"{before.numel} elements are arranged from {before.shape} into {after.shape}. Select a cell to follow its exact position. {storage_note}",
        category="layout", interaction="mapping", mapping=mapping,
    )
