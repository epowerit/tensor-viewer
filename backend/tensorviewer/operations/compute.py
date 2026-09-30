from math import isfinite

from ..models import Lesson, TensorState


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
    if kind == "add" and supports_addition(args, inputs, outputs):
        return Lesson(
            title="Add corresponding elements",
            summary="Match the operands from the last axis; size-one dimensions broadcast when needed.",
            detail="Select an output cell to follow both contributing coordinates. Broadcasting reuses the same operand coordinate along an expanded axis. The result is left + alpha × right, with alpha defaulting to 1.",
            category="compute",
            interaction="broadcast_add",
        )
    if kind == "gelu":
        return Lesson(
            title="Apply GELU to each feature",
            summary="Weight each value with a smooth, nonlinear activation; the shape stays the same.",
            detail=f"PyTorch evaluated GELU with approximation mode {args.get('approximate', 'none')}. Each output depends on the input at the same coordinate; negative values are smoothly attenuated.",
            category="compute",
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
    if kind == "softmax":
        dim = args.get("dim")
        if dim is None or not outputs[0].shape:
            return Lesson(
                title="Turn scores into weights",
                summary="Normalize a group of scores into weights.",
                detail="Inspect the recorded values. An explicit dimension on a non-scalar tensor enables the normalization-group interaction.",
                category="normalize",
            )
        return Lesson(
            title="Turn scores into weights",
            summary=f"Normalize scores along axis {dim}; each group sums to one.",
            detail="Subtract the group maximum for stability, exponentiate, then divide each exponential by their sum. Select an output to inspect its normalization group.",
            category="normalize",
            interaction="normalization",
        )
    if kind in {"mean", "sum"}:
        return Lesson(
            title="Reduce a dimension",
            summary=f"Compute the {kind} over the selected dimensions.",
            detail=f"Arguments: {args}. The output combines contributions from multiple input elements.",
            category="compute",
        )
    return Lesson(
        title={
            "div": "Scale the values",
            "mul": "Multiply the values",
            "add": "Add the values",
            "sub": "Subtract values",
        }.get(kind, "Compute new values"),
        summary="Apply arithmetic element by element, with broadcasting when needed.",
        detail=f"PyTorch evaluated this operation with arguments {args}. Select a tensor to inspect the actual values.",
        category="compute",
    )
