"""Tensors made from nothing but arguments: ranges, constants, random draws."""

from ..models import Lesson, TensorState

FILLED = {"ones": "ones", "zeros": "zeros", "empty": "uninitialized memory"}
RANDOM = {
    "rand": "uniform random values in [0, 1)",
    "randn": "random values from a standard normal distribution",
    "randint": "random integers",
}
CREATION = {
    "arange",
    "linspace",
    "eye",
    "full",
    "tensor",
    *FILLED,
    *RANDOM,
    *(f"{name}_like" for name in [*FILLED, *RANDOM, "full"]),
}


def _shape(tensor: TensorState) -> str:
    return f"[{', '.join(map(str, tensor.shape))}]" if tensor.shape else "scalar"


def _number(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def describe_creation(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    output = outputs[0]
    shape = _shape(output)
    numbers = [value for value in args.get("args", []) if isinstance(value, (int, float))]
    like = kind.endswith("_like")
    base = kind.removesuffix("_like")
    # "a [4, 4] tensor" or "a tensor shaped like x".
    sized = (
        f"{output.dtype} tensor shaped like {inputs[0].name}"
        if like and inputs
        else f"{shape} {output.dtype} tensor"
    )
    if kind == "arange" and numbers:
        start, end, step = (
            (0, numbers[0], 1)
            if len(numbers) == 1
            else (numbers[0], numbers[1], numbers[2] if len(numbers) > 2 else 1)
        )
        summary = f"Count from {_number(start)} up to, but not including, {_number(end)}" + (
            f", in steps of {_number(step)}" if step != 1 else ""
        )
        detail = f"Cell i holds {_number(start)} + i × {_number(step)}. The result has {output.numel} values; it is a common source of positions and indices."
        title = "Count along an axis"
    elif kind == "linspace":
        title = "Space values evenly"
        summary = (
            f"Make {output.numel} evenly spaced values between the two endpoints, both included"
        )
        detail = "Cell i holds start + i × (end − start) / (steps − 1)."
    elif kind == "eye":
        title = "Make an identity matrix"
        summary = f"Make a {shape} matrix with ones on the diagonal and zeros elsewhere"
        detail = "Multiplying by an identity matrix leaves a vector unchanged."
    elif base in FILLED:
        title = f"Make a tensor of {FILLED[base]}"
        summary = f"Make a {sized} filled with {FILLED[base]}"
        detail = (
            "Its values are whatever was in memory; code is expected to overwrite them."
            if base == "empty"
            else "Constant tensors often start masks, accumulators, and padding."
        )
    elif base == "full":
        fill = args.get("fill_value", numbers[-1] if numbers else None)
        title = "Make a constant tensor"
        summary = f"Make a {sized} where every value is {_number(fill) if fill is not None else 'the fill value'}"
        detail = "Every cell holds the same value."
    elif base in RANDOM:
        title = "Draw random values"
        summary = f"Fill a {sized} with {RANDOM[base]}"
        detail = (
            "The values depend on the random seed. Set the input's seed to repeat a run exactly."
        )
    else:
        title = "Make a tensor from data"
        summary = f"Copy the given values into a new {shape} tensor"
        detail = "The values come straight from the Python data in the call."
    return Lesson(
        title=title,
        summary=summary + ".",
        detail=detail,
        category="creation",
    )
