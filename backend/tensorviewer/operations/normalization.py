"""Layer normalization groups and unambiguous affine operand roles."""

from math import isfinite, prod

from ..models import Lesson


def affine_roles(args, count):
    roles = []
    for name in ("weight", "bias"):
        if name not in args:
            roles.append(None)
        elif args[name] == "tensor":
            roles.append(True)
        elif args[name] is None:
            roles.append(False)
        else:
            raise ValueError("Unknown affine operand role.")
    missing = roles.count(None)
    remaining = count - roles.count(True)
    if remaining < 0 or remaining > missing or (missing == 2 and remaining == 1):
        raise ValueError("Ambiguous affine operands.")
    return [remaining > 0 if role is None else role for role in roles]


def layer_normalization_spec(args, inputs, outputs):
    if not inputs or len(inputs) > 3 or len(outputs) != 1:
        raise ValueError("Expected input, optional scale/bias, and one output.")
    x, y = inputs[0], outputs[0]
    shape, eps = args.get("normalized_shape"), args.get("eps", 1e-5)
    if (
        not isinstance(shape, (list, tuple))
        or not shape
        or any(type(n) is not int or n < 1 for n in shape)
        or len(shape) > len(x.shape)
        or list(shape) != x.shape[-len(shape) :]
        or x.shape != y.shape
        or x.dtype not in {"float16", "bfloat16", "float32", "float64"}
        or any(not t.numel or t.dtype != x.dtype for t in [*inputs, y])
        or type(eps) not in (int, float)
        or not isfinite(eps)
        or eps < 0
    ):
        raise ValueError("Unsupported layer normalization geometry or dtype.")
    weight, bias = affine_roles(args, len(inputs) - 1)
    if any(t.shape != list(shape) for t in inputs[1:]):
        raise ValueError("Affine operands must match normalized_shape exactly.")
    return {"shape": list(shape), "size": prod(shape), "eps": eps, "weight": weight, "bias": bias}


def describe_layer_normalization(kind, args, inputs, outputs):
    spec = layer_normalization_spec(args, inputs, outputs)
    return Lesson(
        title="Normalize each feature group",
        summary="Center each trailing group, normalize its variance, then apply scale and bias when present.",
        detail=f"Each group contains {spec['size']} elements over trailing shape {spec['shape']}. Variance divides by the group size, not size minus one. Epsilon {spec['eps']} is added inside the square root. Affine parameters are per normalized coordinate and shared across leading groups. Statistics come from the current input, including in evaluation mode.",
        category="normalize",
        interaction="layer_normalization",
    )
