from math import isfinite

from ..models import Lesson, TensorState
from .relations import relation_for


def supports_addition(args, inputs, outputs):
    if len(inputs) != 2 or len(outputs) != 1 or any(t.numel == 0 for t in [*inputs, *outputs]):
        return False
    alpha = args.get("alpha", 1)
    if not isinstance(alpha, (int, float)) or not isfinite(alpha):
        return False
    rank = max(len(t.shape) for t in inputs)
    left, right = [[1] * (rank - len(t.shape)) + t.shape for t in inputs]
    return all(a == b or a == 1 or b == 1 for a, b in zip(left, right)) and outputs[0].shape == [
        max(a, b) for a, b in zip(left, right)
    ]


def describe_compute(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    relation = (
        relation_for(kind, args, inputs, outputs)[0]
        if kind in {"add", "sub", "mul", "div", "gelu"}
        else None
    )
    if kind == "add" and supports_addition(args, inputs, outputs):
        return Lesson(
            title="Add corresponding elements",
            summary="Match the operands from the last axis; size-one dimensions broadcast when needed.",
            detail="Select an output cell to follow both contributing coordinates. Broadcasting reuses the same operand coordinate along an expanded axis. The result is left + alpha × right, with alpha defaulting to 1.",
            category="compute",
            interaction="broadcast_add",
        )
    if kind in {"matmul", "bmm", "mm"}:
        supported = len(inputs) >= 2 and all(len(t.shape) >= 2 for t in inputs[:2])
        return Lesson(
            title="Combine rows and columns",
            summary="Each output cell is a dot product: multiply matching entries, then add them.",
            detail="Choose an output cell to inspect the contributing row from the left tensor and column from the right tensor. Leading dimensions select or broadcast batches.",
            category="compute",
            interaction="dot_product" if supported else "inspect",
        )
    if kind == "linear":
        supported = (
            len(inputs) in (2, 3)
            and len(outputs) == 1
            and bool(inputs[0].shape)
            and len(inputs[1].shape) == 2
            and all(t.numel > 0 for t in [*inputs, *outputs])
            and inputs[0].shape[-1] == inputs[1].shape[1]
            and outputs[0].shape == [*inputs[0].shape[:-1], inputs[1].shape[0]]
            and (len(inputs) == 2 or inputs[2].shape == [inputs[1].shape[0]])
        )
        return Lesson(
            title="Project into new features",
            summary="Apply the module's weights to the last dimension of every input vector.",
            detail=(
                "For each output feature, multiply the input features by one row of the weight matrix, add them, and include bias when present. Leading dimensions are preserved."
                if supported
                else "Inspect the recorded inputs and result. This operand layout does not support the interactive weight-row lesson."
            ),
            category="compute",
            interaction="linear_projection" if supported else "inspect",
        )
    if kind == "normalize":
        dim = args.get("dim", 1)
        p = args.get("p", 2)
        norm = (
            "length" if p in (2, 2.0) else f"L{p:g} norm" if isinstance(p, (int, float)) else "norm"
        )
        axis = (
            inputs[0].axes[dim]
            if isinstance(dim, int) and -len(inputs[0].axes) <= dim < len(inputs[0].axes)
            else None
        )
        named = f" ({axis})" if axis and not axis.startswith("axis ") else ""
        return Lesson(
            title="Scale each vector to unit length"
            if norm == "length"
            else "Scale each vector to unit norm",
            summary=f"Divide each vector along axis {dim}{named} by its {norm}, so every vector has {norm} 1.",
            detail=(
                "Each output cell is the input cell divided by the norm of the vector it belongs to; "
                "the other coordinates pick the vector. Direction is kept and size is discarded, "
                "so a dot product of two normalized vectors is their cosine similarity. "
                "eps keeps an all-zero vector from dividing by zero."
            ),
            category="normalize",
        )
    if kind == "log_softmax":
        dim = args.get("dim")
        if dim is None or not outputs[0].shape:
            return Lesson(
                title="Turn scores into log-probabilities",
                summary="Normalize a group of scores and take the logarithm.",
                detail="Inspect the recorded values. An explicit dimension on a non-scalar tensor enables the normalization-group interaction.",
                category="normalize",
            )
        return Lesson(
            title="Turn scores into log-probabilities",
            summary=f"Normalize scores along axis {dim} and take the logarithm; the exponentials of each group sum to one.",
            detail="Subtract the group maximum, then subtract the logarithm of the summed exponentials. Select an output to inspect its normalization group.",
            category="normalize",
            interaction="normalization",
        )
    return Lesson(
        title={
            "div": "Scale the values",
            "mul": "Multiply the values",
            "add": "Add the values",
            "sub": "Subtract values",
        }.get(kind, "Compute new values"),
        summary="Apply arithmetic element by element, with broadcasting when needed.",
        detail=f"PyTorch evaluated this operation with arguments {args}. Select an output cell to see the operand cells it combines.",
        category="compute",
        interaction="relation" if relation else "inspect",
        relation=relation,
    )
