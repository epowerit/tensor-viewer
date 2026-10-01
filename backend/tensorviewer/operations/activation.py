"""Elementwise activation lessons; mutation effects keep their own inspector."""

from math import prod

from ..models import Lesson

ACTIVATIONS = {"relu": "ReLU", "gelu": "GELU", "sigmoid": "Sigmoid", "tanh": "Tanh"}
FLOATS = {"float16", "bfloat16", "float32", "float64"}
INTEGERS = {"uint8", "int8", "int16", "int32", "int64"}


def activation_spec(kind, args, inputs, outputs):
    if kind not in ACTIVATIONS or len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected one input and one output for a supported activation.")
    x, y = inputs[0], outputs[0]
    if (
        x.shape != y.shape
        or any(
            t.numel < 1 or prod(t.shape) != t.numel or any(n < 1 for n in t.shape) for t in (x, y)
        )
        or x.dtype != y.dtype
        or x.dtype not in (FLOATS | INTEGERS if kind == "relu" else FLOATS)
        or args.get("inplace", False)
        or args.get("out") is not None
    ):
        raise ValueError("Unsupported activation shape, dtype, or mutation.")
    approximate = args.get("approximate", "none") if kind == "gelu" else "none"
    if approximate not in {"none", "tanh"}:
        raise ValueError("Unknown GELU approximation.")
    return {"kind": kind, "approximate": approximate}


def describe_activation(kind, args, inputs, outputs):
    spec = activation_spec(kind, args, inputs, outputs)
    details = {
        "relu": "ReLU computes max(0, x): negative values become zero and positive values remain unchanged.",
        "gelu": (
            "GELU multiplies x by the standard normal cumulative distribution Φ(x)."
            if spec["approximate"] == "none"
            else "This GELU call uses the tanh approximation: 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))."
        ),
        "sigmoid": "Sigmoid computes 1 / (1 + exp(−x)). Large magnitudes approach zero or one; floating-point results may round to those endpoints.",
        "tanh": "Tanh computes the hyperbolic tangent. Large magnitudes approach −1 or 1; floating-point results may round to those endpoints.",
    }
    return Lesson(
        title=f"Apply {ACTIVATIONS[kind]} to each element",
        summary="Transform each value independently while keeping its shape and coordinate.",
        detail=details[kind]
        + " Select a captured input/output pair to follow the same coordinate. The curve illustrates the function; the recorded output remains authoritative.",
        category="compute",
        interaction="activation",
    )
