"""Coordinate-only descriptions for joins and their ordered partitions."""

from ..models import Lesson

JOIN_KINDS = {"cat", "concat", "concatenate", "stack"}
PARTITION_KINDS = {"split", "chunk", "unbind"}
ASSEMBLY_KINDS = JOIN_KINDS | PARTITION_KINDS


def assembly_spec(kind, arguments, inputs, outputs):
    if kind not in ASSEMBLY_KINDS:
        raise ValueError("Not a tensor assembly operation.")
    joining = kind in JOIN_KINDS
    wholes, parts = (outputs, inputs) if joining else (inputs, outputs)
    if len(wholes) != 1 or not parts:
        raise ValueError("Expected one whole tensor and an ordered list of parts.")
    whole = wholes[0]
    if not whole.numel or not whole.shape or any(p.dtype != whole.dtype for p in parts):
        raise ValueError("This lesson needs non-empty whole tensors with a preserved dtype.")
    axis = arguments.get("dim", 0)
    if (
        isinstance(axis, bool)
        or not isinstance(axis, int)
        or not -len(whole.shape) <= axis < len(whole.shape)
    ):
        raise ValueError("Invalid assembly axis.")
    axis %= len(whole.shape)
    inserted = kind in {"stack", "unbind"}
    if inserted:
        expected = whole.shape[:axis] + whole.shape[axis + 1 :]
        if len(parts) != whole.shape[axis] or any(p.shape != expected for p in parts):
            raise ValueError(
                "Stack/unbind must add/remove one axis without changing other dimensions."
            )
    else:
        if (
            any(
                len(p.shape) != len(whole.shape)
                or any(a != b for i, (a, b) in enumerate(zip(p.shape, whole.shape)) if i != axis)
                for p in parts
            )
            or sum(p.shape[axis] for p in parts) != whole.shape[axis]
        ):
            raise ValueError("The parts must cover the whole axis in order.")
    return joining, inserted, axis, whole, parts


def describe_assembly(kind, arguments, inputs, outputs):
    joining, inserted, axis, whole, parts = assembly_spec(kind, arguments, inputs, outputs)
    if joining:
        title = "Stack on a new axis" if inserted else "Join along an existing axis"
        summary = (
            f"Insert axis {axis} with one position per input tensor."
            if inserted
            else f"Extend axis {axis} by placing the input tensors end to end."
        )
        detail = "No arithmetic is performed. Select a result cell to identify its source tensor and original coordinate."
    else:
        title = "Unbind an axis" if inserted else "Split into ordered parts"
        summary = (
            f"Remove axis {axis}; each position becomes a separate tensor."
            if inserted
            else f"Divide axis {axis} into {len(parts)} recorded outputs, preserving their order."
        )
        detail = "Select a cell in any output to locate it in the original tensor. Recorded output sizes are authoritative; chunk can return fewer parts than requested."
    return Lesson(
        title=title,
        summary=summary,
        detail=detail,
        category="layout",
        interaction="tensor_assembly",
    )


def assembly_axes(kind, arguments, inputs, outputs):
    joining, inserted, axis, whole, parts = assembly_spec(kind, arguments, inputs, outputs)
    if joining:
        rank = len(parts[0].shape)
        axes = [
            parts[0].axes[i] if all(p.axes[i] == parts[0].axes[i] for p in parts) else f"axis {i}"
            for i in range(rank)
        ]
        if inserted:
            axes.insert(axis, "stack")
        return [axes]
    axes = whole.axes.copy()
    if inserted:
        axes.pop(axis)
    return [axes.copy() for _ in parts]
