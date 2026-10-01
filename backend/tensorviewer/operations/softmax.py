"""Explicit-axis softmax geometry, independent of physical tensor storage."""

from math import prod

from ..models import Lesson


def softmax_spec(args, inputs, outputs):
    if len(inputs) != 1 or len(outputs) != 1 or args.get("out") is not None:
        raise ValueError("Expected one softmax input and one output.")
    x, y = inputs[0], outputs[0]
    dim = args.get("dim")
    rank = max(1, len(x.shape))
    dtype = args.get("dtype")
    if (
        type(dim) is not int
        or not -rank <= dim < rank
        or x.shape != y.shape
        or x.dtype not in {"float16", "bfloat16", "float32", "float64"}
        or y.dtype != x.dtype
        or dtype not in (None, f"torch.{x.dtype}")
        or any(
            t.numel < 1 or prod(t.shape) != t.numel or any(n < 1 for n in t.shape) for t in (x, y)
        )
    ):
        raise ValueError("Unsupported softmax axis, geometry, or dtype conversion.")
    axis = dim % rank
    size = x.shape[axis] if x.shape else 1
    stride = prod(x.shape[axis + 1 :])
    return {"axis": axis, "size": size, "groups": x.numel // size, "stride": stride}


def describe_softmax(kind, args, inputs, outputs):
    spec = softmax_spec(args, inputs, outputs)
    return Lesson(
        title="Turn scores into weights",
        summary="Shift scores by their group maximum, exponentiate, then divide by the shared total.",
        detail=f"Each group spans {spec['size']} cells along axis {spec['axis']}. "
        "The other coordinates stay fixed. Negative-infinity scores receive zero weight when "
        "a finite score exists; all-masked and other non-finite groups have no finite reference.",
        category="normalize",
        interaction="normalization",
    )
