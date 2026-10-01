"""Reduction groups described by axes, without materializing contributor maps."""

from math import prod

from ..models import Lesson

FLOATS = {"float16", "bfloat16", "float32", "float64"}
INTEGERS = {"uint8", "int8", "int16", "int32", "int64"}


def reduction_spec(kind, args, inputs, outputs):
    if kind not in {"mean", "sum"} or len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one reduction input and one output.")
    x, y = inputs[0], outputs[0]
    if any(t.numel < 1 or prod(t.shape) != t.numel or any(n < 1 for n in t.shape) for t in (x, y)):
        raise ValueError("Reduction lessons require nonempty tensors.")
    dtype = args.get("dtype")
    if dtype is None:
        dtype = "int64" if kind == "sum" and x.dtype in INTEGERS | {"bool"} else x.dtype
    elif isinstance(dtype, str) and dtype.startswith("torch."):
        dtype = dtype.removeprefix("torch.")
    else:
        raise ValueError("Unknown reduction dtype.")
    if (
        x.dtype not in FLOATS | INTEGERS | {"bool"}
        or dtype not in (FLOATS if kind == "mean" else FLOATS | INTEGERS)
        or y.dtype != dtype
        or args.get("out") is not None
    ):
        raise ValueError("Unsupported reduction dtype or output mutation.")
    keepdim = args.get("keepdim", False)
    if type(keepdim) is not bool:
        raise ValueError("keepdim must be a boolean.")
    rank = len(x.shape)
    dim = args.get("dim")
    if dim is None or isinstance(dim, (list, tuple)) and not dim:
        axes = list(range(rank))
    else:
        dims = list(dim) if isinstance(dim, (list, tuple)) else [dim]
        virtual_rank = max(1, rank)  # PyTorch permits dim=0 or -1 on a scalar.
        if any(type(d) is not int or not -virtual_rank <= d < virtual_rank for d in dims):
            raise ValueError("Invalid reduction axis.")
        normalized = [d % virtual_rank for d in dims]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Repeated reduction axis.")
        axes = sorted(normalized) if rank else []
    shape = (
        [1 if i in axes else n for i, n in enumerate(x.shape)]
        if keepdim
        else [n for i, n in enumerate(x.shape) if i not in axes]
    )
    if shape != y.shape:
        raise ValueError("Recorded shape does not match the reduction axes.")
    return {"axes": axes, "keepdim": keepdim, "size": prod(x.shape[i] for i in axes)}


def reduction_axes(kind, args, inputs, outputs):
    spec = reduction_spec(kind, args, inputs, outputs)
    return [
        name for i, name in enumerate(inputs[0].axes) if spec["keepdim"] or i not in spec["axes"]
    ]


def describe_reduction(kind, args, inputs, outputs):
    spec = reduction_spec(kind, args, inputs, outputs)
    return Lesson(
        title="Average each group" if kind == "mean" else "Sum each group",
        summary="Hold the unreduced coordinates fixed and combine all cells along the reduced axes.",
        detail=f"Each output uses {spec['size']} input cells over axes {spec['axes']}. "
        + (
            "Mean divides their sum by the full group size. "
            if kind == "mean"
            else "Sum adds their values. "
        )
        + (
            "Reduced axes remain with size one." if spec["keepdim"] else "Reduced axes are removed."
        ),
        category="compute",
        interaction="reduction",
    )
