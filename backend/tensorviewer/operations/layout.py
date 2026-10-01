from math import prod

import torch

from ..models import Lesson, TensorState


def describe_layout(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    before, after = inputs[0], outputs[0]
    if before.dtype != after.dtype:
        return Lesson(
            title="Reinterpret the stored bits",
            summary="A dtype view interprets the same bytes using a different element type.",
            detail=f"The dtype changed from {before.dtype} to {after.dtype}. This is not a value-preserving reshape; inspect the recorded values and storage.",
            category="memory",
        )
    rank = len(before.shape)
    order = None
    mapping = None
    rule = None
    small = max(before.numel, after.numel) <= 4096
    if kind in {
        "reshape",
        "view",
        "flatten",
        "contiguous",
        "clone",
        "squeeze",
        "unsqueeze",
        "ravel",
        "view_as",
        "reshape_as",
        "unflatten",
    }:
        if before.numel == after.numel:
            rule = "identity"
            mapping = list(range(after.numel)) if small else None
    if kind == "permute":
        order = [int(d) % rank for d in args["dims"]]
    elif kind == "T":
        order = list(range(rank))[::-1]
    elif kind in {"movedim", "moveaxis"}:
        # Distinct sizes on a metadata tensor reveal where each axis lands.
        probe = torch.empty(tuple(range(2, 2 + rank)), device="meta")
        moved = probe.movedim(args["source"], args["destination"])
        order = [size - 2 for size in moved.shape]
    elif kind in {"transpose", "t", "swapaxes", "swapdims", "mT"}:
        order = list(range(rank))
        a, b = (
            (0, 1)
            if kind == "t"
            else (rank - 2, rank - 1)
            if kind == "mT"
            else (args["dim0"] % rank, args["dim1"] % rank)
        )
        if rank > 1:
            order[a], order[b] = order[b], order[a]
    if order is not None:
        rule = "permutation"
    if order is not None and small:
        mapping = (
            torch.arange(prod(before.shape), device="cpu")
            .reshape(before.shape)
            .permute(order)
            .flatten()
            .tolist()
        )
    if kind == "unfold":
        rule = "unfold"
    if kind == "unfold" and small:
        mapping = (
            torch.arange(before.numel, device="cpu")
            .reshape(before.shape)
            .unfold(args["dimension"], args["size"], args["step"])
            .reshape(-1)
            .tolist()
        )
    shared = before.storage_id == after.storage_id
    storage_note = (
        "This output shares storage with the input."
        if shared
        else "This output uses separate storage."
    )
    if kind == "roll" and before.shape == after.shape:
        shifts = args["shifts"]
        dims = args.get("dims")
        if small:
            mapping = (
                torch.roll(
                    torch.arange(before.numel, device="cpu").reshape(before.shape),
                    shifts=shifts if isinstance(shifts, int) else tuple(shifts),
                    dims=dims if isinstance(dims, int) or dims is None else tuple(dims),
                )
                .reshape(-1)
                .tolist()
            )
        return Lesson(
            title="Shift with wraparound",
            summary="Move values along the selected axes; values crossing an edge reappear at the other edge.",
            detail=f"Shifts {shifts} along {dims if dims is not None and dims != [] else 'the flattened tensor'}. This changes coordinates, not values. A cyclic shift alone does not mask attention. {storage_note}",
            category="layout",
            interaction="mapping",
            mapping=mapping,
            mapping_rule="roll",
        )
    if order is not None:
        return Lesson(
            title="Reorder the axes",
            summary="Change the order of dimensions while keeping the same values.",
            detail=f"Output axes read input axes in the order {order}. Select an output element to find its original coordinates. {storage_note}",
            category="layout",
            interaction="mapping",
            mapping=mapping,
            mapping_rule=rule,
            axis_order=order,
        )
    if kind in {"contiguous", "clone"}:
        return Lesson(
            title="Make values contiguous" if kind == "contiguous" else "Copy the tensor",
            summary="The logical shape and values stay the same; inspect the storage and strides.",
            detail=(
                "The requested memory layout is already satisfied; no copy was needed. "
                if shared
                else "PyTorch copied the values into new storage. "
            )
            + storage_note,
            category="memory",
            interaction="mapping",
            mapping=mapping,
            mapping_rule=rule,
        )
    if kind == "unfold":
        return Lesson(
            title="Extract sliding windows",
            summary="Expose each window as another dimension.",
            detail=f"A window of {args['size']} moves by {args['step']} along axis {args['dimension']}. Overlapping windows may reference the same input element. {storage_note}",
            category="layout",
            interaction="mapping",
            mapping=mapping,
            mapping_rule=rule,
        )
    return Lesson(
        title="Regroup the elements",
        summary="Keep the logical element sequence and change its dimension boundaries.",
        detail=f"{before.numel} elements are arranged from {before.shape} into {after.shape}. Select a cell to follow its exact position. {storage_note}",
        category="layout",
        interaction="mapping",
        mapping=mapping,
        mapping_rule=rule,
    )
