from collections.abc import Callable

from ..models import Lesson, TensorState
from .assembly import ASSEMBLY_KINDS, describe_assembly
from .compute import describe_compute
from .convolution import describe_convolution
from .layout import describe_layout
from .normalization import describe_layer_normalization
from .pooling import POOL_KINDS, describe_pooling
from .relations import RELATION_KINDS, describe_relation
from .spatial import describe_spatial

Adapter = Callable[[str, dict, list[TensorState], list[TensorState]], Lesson]
ADAPTERS: dict[str, Adapter] = {}


def register(names: list[str], adapter: Adapter):
    for name in names:
        ADAPTERS[name] = adapter


register(
    [
        "reshape",
        "view",
        "flatten",
        "permute",
        "transpose",
        "t",
        "contiguous",
        "clone",
        "squeeze",
        "unsqueeze",
        "unfold",
        "roll",
        "swapaxes",
        "swapdims",
        "movedim",
        "moveaxis",
        "T",
        "mT",
        "ravel",
        "view_as",
        "reshape_as",
        "unflatten",
    ],
    describe_layout,
)
register(
    [
        "matmul",
        "mm",
        "bmm",
        "linear",
        "softmax",
        "log_softmax",
        "gelu",
        "div",
        "mul",
        "add",
        "sub",
    ],
    describe_compute,
)
register(RELATION_KINDS, describe_relation)
register(["conv2d"], describe_spatial)
register(["conv1d"], describe_convolution)
register(list(ASSEMBLY_KINDS), describe_assembly)
register(list(POOL_KINDS), describe_pooling)
register(["layer_norm"], describe_layer_normalization)


def describe_operation(
    kind: str, arguments: dict, inputs: list[TensorState], outputs: list[TensorState]
):
    adapter = ADAPTERS.get(kind)
    if adapter and inputs and outputs:
        try:
            return adapter(kind, arguments, inputs, outputs)
        except (KeyError, ValueError, TypeError, IndexError, RuntimeError):
            # A valid execution can always be inspected, even without a matching lesson.
            pass
    return Lesson(
        title=f"Inspect {kind}",
        summary="Recorded from the real PyTorch execution.",
        detail="A specialized lesson is not available for this operation. Its inputs, outputs, source, and tensor metadata remain available.",
        category="generic",
    )
