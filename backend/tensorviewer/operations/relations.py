"""Compact cell-to-cell rules for selection, repetition, reduction, and elementwise work.

Each rule is validated against the recorded shapes. The frontend evaluates it per
selected coordinate, so no map proportional to tensor size is needed.
"""

import re

import torch

from ..models import Lesson, TensorState

SMALL = 4096

UNARY = {
    "relu": "Keep positive values and replace negative ones with zero",
    "gelu": "Weight each value with a smooth, nonlinear activation",
    "sigmoid": "Squash each value into the interval (0, 1)",
    "tanh": "Squash each value into the interval (-1, 1)",
    "silu": "Multiply each value by its own sigmoid",
    "leaky_relu": "Keep positive values and shrink negative ones",
    "exp": "Raise e to each value",
    "log": "Take the natural logarithm of each value",
    "sqrt": "Take the square root of each value",
    "rsqrt": "Take one over the square root of each value",
    "square": "Multiply each value by itself",
    "abs": "Drop the sign of each value",
    "neg": "Flip the sign of each value",
    "sin": "Take the sine of each value",
    "cos": "Take the cosine of each value",
    "reciprocal": "Take one over each value",
    "floor": "Round each value down",
    "ceil": "Round each value up",
    "round": "Round each value to the nearest integer",
    "sign": "Keep only the sign of each value",
    "clamp": "Limit each value to a range",
    "clip": "Limit each value to a range",
    "logical_not": "Flip each true/false value",
    "dropout": "Pass values through unchanged in evaluation mode",
}
BINARY = {
    "add": "Add",
    "sub": "Subtract",
    "rsub": "Subtract",
    "mul": "Multiply",
    "div": "Divide",
    "true_divide": "Divide",
    "floor_divide": "Divide and round down",
    "remainder": "Take the remainder of",
    "pow": "Raise to a power",
    "maximum": "Take the larger of",
    "minimum": "Take the smaller of",
    "eq": "Compare",
    "ne": "Compare",
    "gt": "Compare",
    "ge": "Compare",
    "lt": "Compare",
    "le": "Compare",
    "logical_and": "Combine",
    "logical_or": "Combine",
    "logical_xor": "Combine",
    "__rsub__": "Subtract",
    "__rdiv__": "Divide",
    "__rtruediv__": "Divide",
    "__rpow__": "Raise to a power",
    "__and__": "Combine",
    "__or__": "Combine",
    "__xor__": "Combine",
}
REDUCTIONS = {
    "sum": "adds",
    "mean": "averages",
    "prod": "multiplies",
    "amax": "takes the largest of",
    "amin": "takes the smallest of",
    "max": "takes the largest of",
    "min": "takes the smallest of",
    "argmax": "finds the position of the largest of",
    "argmin": "finds the position of the smallest of",
    "any": "checks whether any are true among",
    "all": "checks whether all are true among",
    "logsumexp": "takes the log of the summed exponentials of",
    "std": "takes the standard deviation of",
    "var": "takes the variance of",
}
PREFIX = {"cumsum": "sum", "cumprod": "product"}
ORDERING = {"sort", "topk"}
# Pure selections: every output element is one input element, so applying the
# same call to a tensor of positions yields the exact map.
REPLAYED = {
    "__getitem__": "Select elements by index",
    "masked_select": "Keep the elements where a mask is true",
    "take": "Pick elements by flat position",
    "take_along_dim": "Pick values by index along an axis",
    "diagonal": "Take a diagonal",
    "narrow": "Keep a range along one axis",
    "select": "Take one position along an axis",
    "flip": "Reverse the order along axes",
    "fliplr": "Reverse the columns",
    "flipud": "Reverse the rows",
    "rot90": "Rotate a plane by quarter turns",
    "pad": "Extend the edges with existing values",
    "repeat_interleave": "Repeat each element in place",
    "tile": "Repeat the whole tensor",
    "interpolate": "Resize by sampling",
    "upsample": "Resize by sampling",
    "pixel_shuffle": "Move channels into space",
    "pixel_unshuffle": "Move space into channels",
    "channel_shuffle": "Interleave channel groups",
}
# Products written as an einsum over index letters.
PRODUCTS = {"dot": "i,i->", "vdot": "i,i->", "outer": "i,j->ij", "mv": "ij,j->i"}
CLASS_LOSSES = {"cross_entropy", "nll_loss"}
ELEMENTWISE = {*UNARY, *BINARY, "where", "masked_fill", "tril", "triu"}
RELATION_KINDS = [
    "__getitem__",
    "expand",
    "expand_as",
    "broadcast_to",
    "repeat",
    "gather",
    "index_select",
    "embedding",
    *REDUCTIONS,
    *PREFIX,
    *ORDERING,
    "einsum",
    "pad",
    "one_hot",
    "batch_norm",
    *PRODUCTS,
    *CLASS_LOSSES,
    *(REPLAYED.keys() - {"__getitem__", "pad"}),
    *(ELEMENTWISE - {"add", "sub", "mul", "div", "gelu"}),
]


def broadcasts(shape: list[int], target: list[int]) -> bool:
    return len(shape) <= len(target) and all(
        a == b or a == 1 for a, b in zip(reversed(shape), reversed(target))
    )


def index_argument(value):
    """Record a subscript structurally; tuples and lists mean different things."""
    if isinstance(value, tuple):
        return [index_argument(item) for item in value]
    if value is None or value is Ellipsis:
        return None if value is None else "..."
    if isinstance(value, bool):
        return {"unsupported": "bool"}
    if isinstance(value, int):
        return value
    if isinstance(value, slice):
        parts = [value.start, value.stop, value.step]
        if all(p is None or (isinstance(p, int) and not isinstance(p, bool)) for p in parts):
            return {"slice": parts}
        return {"unsupported": "slice"}
    return {"unsupported": type(value).__name__}


def _index_relation(args: dict, before: TensorState, after: TensorState):
    index = args.get("index")
    items = index if isinstance(index, list) else [index]
    if any(isinstance(item, dict) and "unsupported" in item for item in items):
        return None
    if items.count("...") > 1:
        return None
    consuming = sum(1 for item in items if item is not None and item != "...")
    rank = len(before.shape)
    if consuming > rank:
        return None
    full = {"slice": [None, None, None]}
    if "..." in items:
        at = items.index("...")
        items = items[:at] + [full] * (rank - consuming) + items[at + 1 :]
    else:
        items = items + [full] * (rank - consuming)
    axes, shape, axis = [], [], 0
    for item in items:
        if item is None:
            shape.append(1)
            continue
        size = before.shape[axis]
        if isinstance(item, int):
            position = item + size if item < 0 else item
            if not 0 <= position < size:
                return None
            axes.append({"kind": "int", "index": position})
        else:
            start, stop, step = slice(*item["slice"]).indices(size)
            if step < 1:
                return None
            axes.append({"kind": "slice", "start": start, "step": step, "out": len(shape)})
            shape.append(len(range(start, stop, step)))
        axis += 1
    if shape != after.shape or not after.numel:
        return None
    return {"rule": "index", "operand": 0, "axes": axes}


def _tile_relation(before: TensorState, after: TensorState):
    offset = len(after.shape) - len(before.shape)
    if offset < 0 or not before.numel or not after.numel:
        return None
    if any(size % source for size, source in zip(after.shape[offset:], before.shape)):
        return None
    return {"rule": "tile", "operand": 0}


def _table(relation_operand: int, mapping: list[int]):
    return {"rule": "table", "operand": relation_operand}, mapping


def _integers(tensor: TensorState):
    values = tensor.values
    if len(values) != tensor.numel or any(isinstance(v, str) for v in values):
        return None
    return torch.tensor([int(v) for v in values], dtype=torch.int64).reshape(tensor.shape)


def _lookup(kind: str, args: dict, inputs: list[TensorState], after: TensorState):
    """Value-dependent selections get an exact map only while it stays small."""
    if len(inputs) < 2 or not after.numel or after.numel > SMALL:
        return None
    if kind == "embedding":
        ids, table = inputs[0], inputs[1]
        if len(table.shape) != 2 or after.shape != [*ids.shape, table.shape[1]]:
            return None
        rows = _integers(ids)
        if rows is None or table.numel > 2**40:
            return None
        width = table.shape[1]
        if rows.numel() and not (0 <= int(rows.min()) and int(rows.max()) < table.shape[0]):
            return None
        mapping = (rows.reshape(-1, 1) * width + torch.arange(width)).reshape(-1).tolist()
        return _table(1, mapping)
    source, index = inputs[0], inputs[1]
    dim = args.get("dim")
    if not isinstance(dim, int) or source.numel > SMALL or not source.shape:
        return None
    positions = _integers(index)
    if positions is None:
        return None
    labels = torch.arange(source.numel).reshape(source.shape)
    try:
        picked = (
            labels.gather(dim, positions)
            if kind == "gather"
            else labels.index_select(dim, positions)
        )
    except (RuntimeError, IndexError):
        return None
    if list(picked.shape) != after.shape:
        return None
    return _table(0, picked.reshape(-1).tolist())


def reduced_axes(args: dict, rank: int) -> list[int] | None:
    dim = args.get("dim")
    if dim is None:
        return list(range(rank))
    dims = dim if isinstance(dim, list) else [dim]
    if not rank or any(not isinstance(d, int) or isinstance(d, bool) for d in dims):
        return None
    if any(not -rank <= d < rank for d in dims):
        return None
    axes = sorted({d % rank for d in dims})
    return axes if len(axes) == len(dims) else None


def _reduce_relation(args: dict, before: TensorState, after: TensorState):
    rank = len(before.shape)
    axes = reduced_axes(args, rank)
    keep = args.get("keepdim", False)
    if axes is None or not isinstance(keep, bool) or not before.numel:
        return None
    expected = [
        1 if axis in axes else size
        for axis, size in enumerate(before.shape)
        if keep or axis not in axes
    ]
    if expected != after.shape:
        return None
    return {"rule": "reduce", "operand": 0, "axes": axes, "keepdim": keep}


def _prefix_relation(args: dict, before: TensorState, after: TensorState):
    dim = args.get("dim")
    rank = len(before.shape)
    if not isinstance(dim, int) or isinstance(dim, bool) or not rank or not before.numel:
        return None
    if not -rank <= dim < rank or before.shape != after.shape:
        return None
    return {"rule": "prefix", "operand": 0, "axis": dim % rank}


def _pad_relation(args: dict, before: TensorState, after: TensorState):
    pad = args.get("pad")
    rank = len(before.shape)
    if args.get("mode", "constant") != "constant" or not isinstance(pad, list):
        return None
    if len(pad) % 2 or len(pad) > 2 * rank or not before.numel or before.dtype != after.dtype:
        return None
    if any(not isinstance(p, int) or isinstance(p, bool) or p < 0 for p in pad):
        return None
    leading = [0] * rank
    shape = list(before.shape)
    # Pairs run from the last axis backwards: (left, right), (top, bottom), …
    for i in range(len(pad) // 2):
        axis = rank - 1 - i
        leading[axis] = pad[2 * i]
        shape[axis] += pad[2 * i] + pad[2 * i + 1]
    if shape != after.shape:
        return None
    return {"rule": "pad", "operand": 0, "before": leading}


def _einsum_relation(args: dict, inputs: list[TensorState], after: TensorState):
    equation = args.get("equation")
    if not isinstance(equation, str) or "." in equation or not after.numel:
        return None
    left, arrow, right = equation.replace(" ", "").partition("->")
    terms = left.split(",")
    if len(terms) != len(inputs) or not all(term.isalpha() or term == "" for term in terms):
        return None
    sizes: dict[str, int] = {}
    for term, tensor in zip(terms, inputs):
        if len(term) != len(tensor.shape) or not tensor.numel:
            return None
        for letter, size in zip(term, tensor.shape):
            if sizes.setdefault(letter, size) != size:
                return None
    if arrow:
        output = right
    else:
        # Implicit form: letters that appear once, in alphabetical order.
        joined = "".join(terms)
        output = "".join(sorted(c for c in set(joined) if joined.count(c) == 1))
    if not (output.isalpha() or output == "") or len(set(output)) != len(output):
        return None
    if any(c not in sizes for c in output) or [sizes[c] for c in output] != after.shape:
        return None
    return {"rule": "einsum", "operand": 0, "inputs": terms, "output": output, "sizes": sizes}


def _order_relation(args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    """sort and topk return the positions they took; those give the exact map."""
    if len(outputs) != 2 or outputs[0].shape != outputs[1].shape:
        return None
    source, after, picked = inputs[0], outputs[0], outputs[1]
    dim = args.get("dim", -1)
    rank = len(source.shape)
    if not isinstance(dim, int) or not rank or not -rank <= dim < rank:
        return None
    if not after.numel or max(source.numel, after.numel) > SMALL:
        return None
    positions = _integers(picked)
    if positions is None:
        return None
    try:
        labels = torch.arange(source.numel).reshape(source.shape)
        mapping = labels.gather(dim, positions).reshape(-1).tolist()
    except (RuntimeError, IndexError):
        return None
    return _table(0, mapping)


def replayed_selection(kind, func, args, kwargs, before: TensorState, after: TensorState):
    """Apply a pure selection to a tensor of positions to learn its exact map."""
    if kind not in REPLAYED or not after.numel or before.dtype != after.dtype:
        return None
    if max(before.numel, after.numel) > SMALL or not before.numel:
        return None
    labels = torch.arange(before.numel, dtype=torch.float64).reshape(before.shape)
    try:
        if kind == "pad":
            # A filled border would not be a copy of any input element.
            if kwargs.get("mode", args[2] if len(args) > 2 else "constant") == "constant":
                return None
        picked = func(labels, *args[1:], **{k: v for k, v in kwargs.items() if k != "out"})
    except Exception:
        return None
    if not isinstance(picked, torch.Tensor) or list(picked.shape) != after.shape:
        return None
    positions = picked.reshape(-1)
    if not bool((positions == positions.round()).all()):
        # A blend of positions, such as linear resizing, is not a selection.
        return None
    mapping = [int(v) for v in positions.tolist()]
    if any(not 0 <= v < before.numel for v in mapping):
        return None
    # Trust the map only if it reproduces every recorded output value.
    if len(before.values) != before.numel or len(after.values) != after.numel:
        return None
    if any(after.values[i] != before.values[source] for i, source in enumerate(mapping)):
        return None
    shared = before.storage_id == after.storage_id
    return Lesson(
        title=REPLAYED[kind],
        summary="Every output cell is a copy of one input cell; nothing is computed.",
        detail=f"{before.shape} becomes {after.shape}. Select an output cell to find the input cell it came from. "
        + (
            "The result is a view: it shares storage with the input."
            if shared
            else "The result has its own storage."
        ),
        category="layout",
        interaction="relation",
        relation={"rule": "table", "operand": 0},
        mapping=mapping,
    )


def _batch_norm_relation(args: dict, inputs: list[TensorState], after: TensorState):
    """Evaluation-mode batch normalization: one stored mean and variance per channel."""
    roles = [
        name
        for name in ("running_mean", "running_var", "weight", "bias")
        if args.get(name) == "tensor"
    ]
    eps = args.get("eps", 1e-5)
    source = inputs[0]
    if args.get("training", False) is not False or roles[:2] != ["running_mean", "running_var"]:
        # Training mode normalizes with statistics of the batch itself.
        return None
    if len(inputs) != 1 + len(roles) or len(source.shape) < 2 or source.shape != after.shape:
        return None
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or not after.numel:
        return None
    channels = source.shape[1]
    if any(t.shape != [channels] for t in inputs[1:]):
        return None
    return {
        "rule": "channel_affine",
        "operand": 0,
        "axis": 1,
        "roles": ["input", *roles],
        "eps": float(eps),
    }


def _one_hot_relation(inputs: list[TensorState], after: TensorState):
    indices = inputs[0]
    if indices.dtype != "int64" or not after.numel or len(after.shape) != len(indices.shape) + 1:
        return None
    if after.shape[:-1] != indices.shape:
        return None
    return {"rule": "one_hot", "operand": 0}


def _class_loss_relation(kind: str, args: dict, inputs: list[TensorState], after: TensorState):
    """Class-index targets over [N, C] or [C] scores, without weights or smoothing."""
    if len(inputs) != 2 or args.get("weight") is not None:
        return None
    scores, target = inputs
    reduction = args.get("reduction", "mean")
    if reduction not in {"mean", "sum", "none"} or args.get("label_smoothing", 0.0) != 0.0:
        return None
    if target.dtype != "int64" or len(scores.shape) not in (1, 2) or not scores.numel:
        return None
    if target.shape != scores.shape[:-1]:
        return None
    if after.shape != (target.shape if reduction == "none" else []):
        return None
    ignored = args.get("ignore_index", -100)
    if not isinstance(ignored, int) or isinstance(ignored, bool):
        return None
    return {
        "rule": "class_loss",
        "operand": 0,
        "reduction": reduction,
        "ignore_index": ignored,
        # nll_loss takes log-probabilities; cross_entropy normalizes raw scores first.
        "normalized": kind == "nll_loss",
    }


def _elementwise_relation(kind: str, args: dict, inputs: list[TensorState], after: TensorState):
    if not after.numel or not inputs:
        return None
    if kind == "where":
        order = ["condition", "input", "other"]
    elif kind == "masked_fill":
        order = ["input", "mask", "value"]
    elif kind in BINARY:
        order = ["input", "other"]
    else:
        order = ["input"]
    # A Python number takes the place of a tensor operand and is kept as an argument.
    roles = [role for role in order if role not in args or args[role] == "tensor"]
    roles = roles[: len(inputs)]
    if len(roles) != len(inputs):
        return None
    if kind == "where" and len(inputs) + sum(r in args for r in ("input", "other")) != 3:
        # torch.where(condition) alone returns positions, not a choice.
        return None
    if any(not broadcasts(t.shape, after.shape) for t in inputs):
        return None
    if kind in UNARY and inputs[0].shape != after.shape:
        return None
    relation = {"rule": "elementwise", "operand": 0, "roles": roles}
    if kind.startswith("__r"):
        # 2 - x calls x.__rsub__(2): the Python number is the left operand.
        relation["reversed"] = True
    if kind in {"tril", "triu"}:
        diagonal = args.get("diagonal", 0)
        if not isinstance(diagonal, int) or len(after.shape) < 2:
            return None
        relation["diagonal"] = diagonal
    return relation


def relation_for(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    """Return (relation, mapping) for a supported operation, or (None, None)."""
    before, after = inputs[0], outputs[0]
    mapping = None
    if kind == "__getitem__":
        relation = _index_relation(args, before, after) if before.dtype == after.dtype else None
    elif kind in {"expand", "expand_as", "broadcast_to", "repeat"}:
        relation = _tile_relation(before, after) if before.dtype == after.dtype else None
    elif kind in {"gather", "index_select", "embedding"}:
        found = _lookup(kind, args, inputs, after)
        relation, mapping = found if found else (None, None)
    elif kind in REDUCTIONS and len(inputs) == 1:
        relation = _reduce_relation(args, before, after)
    elif kind in PREFIX:
        relation = _prefix_relation(args, before, after)
    elif kind in ORDERING:
        found = _order_relation(args, inputs, outputs)
        relation, mapping = found if found else (None, None)
    elif kind == "einsum":
        relation = _einsum_relation(args, inputs, after)
    elif kind in PRODUCTS:
        relation = _einsum_relation({"equation": PRODUCTS[kind]}, inputs, after)
    elif kind == "one_hot":
        relation = _one_hot_relation(inputs, after)
    elif kind == "batch_norm":
        relation = _batch_norm_relation(args, inputs, after)
    elif kind in CLASS_LOSSES:
        relation = _class_loss_relation(kind, args, inputs, after)
    elif kind == "pad":
        relation = _pad_relation(args, before, after)
    elif kind in {"max", "min"} and len(inputs) == 2:
        relation = _elementwise_relation("maximum", args, inputs, after)
    elif kind in ELEMENTWISE:
        relation = _elementwise_relation(kind, args, inputs, after)
    else:
        relation = None
    return relation, mapping


def describe_relation(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    relation, mapping = relation_for(kind, args, inputs, outputs)
    before, after = inputs[0], outputs[0]
    interaction = "relation" if relation else "inspect"
    shared = before.storage_id == after.storage_id
    storage = (
        "The result is a view: it shares storage with the input."
        if shared
        else "The result has its own storage."
    )
    unavailable = " This call's arguments are recorded for inspection without a cell mapping."
    if kind == "__getitem__":
        return Lesson(
            title="Select part of a tensor",
            summary="An index picks one position on an axis; a slice keeps a range of positions.",
            detail=(
                f"{before.shape} becomes {after.shape}. An integer removes its axis, a slice keeps it, and None inserts an axis of size one. Select an output cell to find where it came from. {storage}"
                + ("" if relation else unavailable)
            ),
            category="layout",
            interaction=interaction,
            relation=relation,
        )
    if kind in {"expand", "expand_as", "broadcast_to", "repeat"}:
        copying = kind == "repeat"
        return Lesson(
            title="Repeat the values" if copying else "Broadcast to a larger shape",
            summary=(
                "Copy the tensor several times along each axis."
                if copying
                else "Reuse the same values along axes of size one, without copying them."
            ),
            detail=(
                f"{before.shape} becomes {after.shape}. Shapes align from the last axis. "
                + (
                    "Every output cell is a copy of one input cell. "
                    if copying
                    else "Many output cells read the same stored element. "
                )
                + storage
                + ("" if relation else unavailable)
            ),
            category="memory" if shared else "layout",
            interaction=interaction,
            relation=relation,
        )
    if kind == "embedding":
        return Lesson(
            title="Look up a row for each index",
            summary="Each integer selects one row of the table; nothing is computed.",
            detail="The output gains a final axis holding the selected row's features. Select an output cell to find the table entry it copies."
            + ("" if relation else " Larger lookups are recorded without a cell mapping."),
            category="layout",
            interaction=interaction,
            relation=relation,
            mapping=mapping,
        )
    if kind in {"gather", "index_select"}:
        return Lesson(
            title="Pick values by index",
            summary=(
                "Each output cell reads the input position named by the index tensor."
                if kind == "gather"
                else "Keep the listed positions along one axis, in the listed order."
            ),
            detail=f"Selection runs along axis {args.get('dim')}. Select an output cell to find the input cell it copies."
            + ("" if relation else " Larger selections are recorded without a cell mapping."),
            category="layout",
            interaction=interaction,
            relation=relation,
            mapping=mapping,
        )
    if kind in REDUCTIONS and len(inputs) == 1:
        axes = relation["axes"] if relation else None
        return Lesson(
            title="Reduce a dimension",
            summary=f"Compute the {kind} over the selected dimensions.",
            detail=(
                f"{kind} {REDUCTIONS[kind]} every cell that differs only along axes {axes}. "
                f"{before.shape} becomes {after.shape}. Select an output cell to see its group."
                if relation
                else f"Arguments: {args}. The output combines contributions from multiple input elements."
            ),
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind in PREFIX:
        return Lesson(
            title="Accumulate along an axis",
            summary=f"Each output is the running {PREFIX[kind]} of the inputs up to its position.",
            detail=f"Accumulation runs along axis {args.get('dim')}; the shape stays the same. Select an output cell to see every input it includes.",
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind in ORDERING:
        return Lesson(
            title="Order values along an axis" if kind == "sort" else "Keep the largest values",
            summary="Values move to new positions along one axis; the recorded positions say where each came from.",
            detail=f"The second output holds the original position of every value along axis {args.get('dim', -1)}. Select an output cell to find its source."
            + ("" if relation else " Larger tensors are recorded without a cell mapping."),
            category="layout",
            interaction=interaction,
            relation=relation,
            mapping=mapping,
        )
    if kind == "einsum":
        return Lesson(
            title="Multiply and sum by index letters",
            summary=f"{args.get('equation')}: letters missing from the output are summed over.",
            detail=(
                "Each letter names an axis. An output cell fixes the letters it keeps; every combination of the remaining letters contributes one product of operand cells."
                if relation
                else "This equation is recorded for inspection. Explicit letters without an ellipsis enable the contribution view."
            ),
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind in PRODUCTS:
        return Lesson(
            title="Multiply matching entries and add",
            summary=f"{kind} is the einsum {PRODUCTS[kind]}: letters missing from the output are summed over.",
            detail="Select an output cell to see every product that contributes to it.",
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind == "batch_norm":
        return Lesson(
            title="Normalize each channel",
            summary=(
                "Shift and scale every value using its channel's stored mean and variance."
                if relation
                else "Normalize every channel using statistics of the batch."
            ),
            detail=(
                "In evaluation mode the statistics are fixed numbers saved with the layer, so each output depends only on the input at the same position and on its channel. Select a cell to see its channel's mean, variance, scale, and shift."
                if relation
                else "In training mode each channel is normalized with the mean and variance of the current batch. This run records the operands and result for inspection."
            ),
            category="normalize",
            interaction=interaction,
            relation=relation,
        )
    if kind == "one_hot":
        return Lesson(
            title="Turn class numbers into indicator vectors",
            summary="Each index becomes a vector that is one at that position and zero elsewhere.",
            detail=f"{before.shape} becomes {after.shape}: the new last axis has one position per class. Select an output cell to see the index it is compared with.",
            category="layout",
            interaction=interaction,
            relation=relation,
        )
    if kind in CLASS_LOSSES:
        normalized = kind == "nll_loss"
        return Lesson(
            title="Score the true class",
            summary=(
                "Take the log-probability at each target class and negate it."
                if normalized
                else "Turn scores into log-probabilities, then take the negative one at each target class."
            ),
            detail=(
                f"Each sample contributes −log p(target). Reduction: {args.get('reduction', 'mean')}. Select the loss to see each sample's target score."
                if relation
                else "Class weights, label smoothing, probability targets, and extra spatial axes are recorded for inspection without the per-sample view."
            ),
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind == "pad":
        return Lesson(
            title="Add a border",
            summary="Surround the tensor with a constant value; the original cells keep their values.",
            detail=f"{before.shape} becomes {after.shape}. Padding amounts are listed from the last axis backwards. Select an output cell to see whether it is an original cell or padding."
            if relation
            else "This padding mode copies existing values into the border.",
            category="layout",
            interaction=interaction,
            relation=relation,
        )
    if kind in REPLAYED:
        # Shown when no verified cell map exists: larger tensors, shape-only
        # runs, or a mode that blends values instead of copying them.
        return Lesson(
            title=REPLAYED[kind],
            summary="Recorded from the real execution.",
            detail="When every output cell is a copy of one input cell, value runs of up to 4,096 elements link each cell to its source.",
            category="layout",
        )
    if kind in {"tril", "triu"}:
        return Lesson(
            title="Keep a triangle of each matrix",
            summary=f"Keep the {'lower' if kind == 'tril' else 'upper'} triangle of the last two axes and zero the rest.",
            detail="Whether a cell is kept depends only on its row and column, never on its value. This is how causal attention masks are built.",
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind == "where" and not relation and len(inputs) == 1:
        return Lesson(
            title="Find where a condition holds",
            summary="Return the coordinates of every true element, one tensor per axis.",
            detail="The number of results depends on the values, so there is no fixed cell mapping.",
            category="layout",
        )
    if kind == "where":
        return Lesson(
            title="Choose between two sources",
            summary="Where the condition is true take the first value, otherwise the second.",
            detail="All operands broadcast to the output shape. Select an output cell to see which source it used.",
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind == "masked_fill":
        return Lesson(
            title="Overwrite the masked cells",
            summary="Where the mask is true write the fill value; elsewhere keep the input.",
            detail="The mask broadcasts to the input's shape. Select an output cell to see whether it was filled.",
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    if kind in UNARY:
        summary = UNARY[kind]
        detail = "Each output depends only on the input at the same coordinate."
        if kind == "dropout":
            training, probability = args.get("training"), args.get("p", 0.5)
            if training is False or probability == 0:
                summary = "Pass values through unchanged"
            elif training is True:
                summary = (
                    "Replace every value with zero"
                    if probability == 1
                    else f"Zero values with probability {probability} and scale the kept values by 1 / (1 - {probability})"
                )
                detail = (
                    "Each output uses the input at the same coordinate and a sampled dropout mask. "
                    "The diagram shows the values recorded in this run."
                )
            else:
                summary = "Apply dropout according to this call's training mode and probability"
                detail = (
                    "Inspect the recorded arguments and values to see whether dropout was active."
                )
        elif kind == "round":
            decimals = args.get("decimals", 0)
            summary = (
                "Round each value to the nearest integer"
                if decimals == 0
                else f"Round each value to {decimals} decimal places"
                if isinstance(decimals, int) and decimals > 0
                else f"Round each value with decimals={decimals}"
            )
            detail += " Halfway values round to the nearest even result."
        return Lesson(
            title=f"Apply {kind} to each element",
            summary=summary + "; the shape stays the same.",
            detail=detail,
            category="compute",
            interaction=interaction,
            relation=relation,
        )
    return Lesson(
        title=f"{BINARY.get(kind, 'Combine')} element by element",
        summary="Apply the operation to matching cells, with broadcasting when needed.",
        detail="Shapes align from the last axis; an axis of size one is reused along the other operand's axis. Select an output cell to see both operands.",
        category="compute",
        interaction=interaction,
        relation=relation,
    )


def relation_axes(relation: dict, inputs: list[TensorState], output: TensorState) -> list[str]:
    """Carry axis names through rules that keep an axis's meaning."""
    before = inputs[relation.get("operand", 0)]
    rank = len(output.shape)
    neutral = [f"axis {i}" for i in range(rank)]

    def carried(names: list[str]) -> list[str]:
        # A placeholder such as "axis 2" is a position, not a meaning.
        return [
            neutral[i] if re.fullmatch(r"axis \d+", name) else name for i, name in enumerate(names)
        ]

    if relation["rule"] == "channel_affine":
        return before.axes.copy()
    if relation["rule"] == "prefix" or (relation["rule"] == "pad" and len(before.axes) == rank):
        return before.axes.copy()
    if relation["rule"] == "elementwise":
        for tensor in inputs:
            if tensor.shape == output.shape:
                return tensor.axes.copy()
        return neutral
    if relation["rule"] == "reduce":
        axes = relation["axes"]
        names = [
            name for axis, name in enumerate(before.axes) if relation["keepdim"] or axis not in axes
        ]
        return carried(names) if len(names) == rank else neutral
    if relation["rule"] == "index":
        names = neutral.copy()
        for axis, item in enumerate(relation["axes"]):
            if item["kind"] == "slice":
                names[item["out"]] = before.axes[axis]
        return carried(names)
    if relation["rule"] == "tile":
        offset = rank - len(before.shape)
        return carried(neutral[:offset] + before.axes)
    return neutral
