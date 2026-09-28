from ..models import Lesson, TensorState


def describe_compute(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    if kind in {"matmul", "bmm", "mm"}:
        supported = len(inputs) >= 2 and all(len(t.shape) >= 2 for t in inputs[:2])
        return Lesson(
            title="Combine rows and columns",
            summary="Each output cell is a dot product: multiply matching entries, then add them.",
            detail="Choose an output cell to inspect the contributing row from the left tensor and column from the right tensor. Leading dimensions select or broadcast batches.",
            category="compute", interaction="dot_product" if supported else "inspect",
        )
    if kind == "linear":
        return Lesson(
            title="Project into new features",
            summary="Apply learned weights to the last dimension of every input vector.",
            detail="For each output feature, multiply the input features by one row of the weight matrix, add them, and include bias when present. Leading dimensions are preserved.",
            category="compute",
        )
    if kind == "softmax":
        dim = args.get("dim", -1)
        return Lesson(
            title="Turn scores into weights",
            summary=f"Normalize scores along axis {dim}; each group sums to one.",
            detail="Subtract the group maximum for stability, exponentiate, then divide each exponential by their sum. Select an output to inspect its normalization group.",
            category="normalize", interaction="normalization",
        )
    if kind in {"mean", "sum"}:
        return Lesson(
            title="Reduce a dimension", summary=f"Compute the {kind} over the selected dimensions.",
            detail=f"Arguments: {args}. The output combines contributions from multiple input elements.",
            category="compute",
        )
    return Lesson(
        title={"div": "Scale the values", "mul": "Multiply the values", "add": "Add the values", "sub": "Subtract values"}.get(kind, "Compute new values"),
        summary="Apply arithmetic element by element, with broadcasting when needed.",
        detail=f"PyTorch evaluated this operation with arguments {args}. Select a tensor to inspect the actual values.",
        category="compute",
    )
