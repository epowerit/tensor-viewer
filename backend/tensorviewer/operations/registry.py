from collections.abc import Callable

from ..models import Lesson, TensorState
from .compute import describe_compute
from .layout import describe_layout

Adapter = Callable[[str, dict, list[TensorState], list[TensorState]], Lesson]
ADAPTERS: dict[str, Adapter] = {}


def register(names: list[str], adapter: Adapter):
    for name in names:
        ADAPTERS[name] = adapter


register(["reshape", "view", "flatten", "permute", "transpose", "t", "contiguous", "clone", "squeeze", "unsqueeze", "unfold"], describe_layout)
register(["matmul", "mm", "bmm", "linear", "softmax", "div", "mul", "add", "sub", "mean", "sum"], describe_compute)


def describe_operation(kind: str, arguments: dict, inputs: list[TensorState], outputs: list[TensorState]):
    adapter = ADAPTERS.get(kind)
    if adapter and inputs and outputs:
        try:
            return adapter(kind, arguments, inputs, outputs)
        except (KeyError, ValueError, TypeError, IndexError, RuntimeError):
            # A valid execution can always be inspected, even without a matching lesson.
            pass
    return Lesson(
        title=f"Inspect {kind}", summary="Recorded from the real PyTorch execution.",
        detail="A specialized lesson is not available for this operation. Its inputs, outputs, source, and tensor metadata remain available.",
        category="generic",
    )
