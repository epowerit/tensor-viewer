from collections.abc import Callable

from ..models import Lesson, TensorState
from .activation import ACTIVATIONS, describe_activation
from .assembly import ASSEMBLY_KINDS, describe_assembly
from .compute import describe_compute
from .convolution import describe_convolution
from .layout import describe_layout
from .normalization import describe_layer_normalization
from .pooling import POOL_KINDS, describe_pooling
from .reduction import describe_reduction
from .relations import RELATION_KINDS, describe_relation, relation_for
from .softmax import describe_softmax
from .spatial import describe_spatial

Adapter = Callable[[str, dict, list[TensorState], list[TensorState]], Lesson]
ADAPTERS: dict[str, Adapter] = {}


def register(names: list[str], adapter: Adapter):
    # A later registration replaces an earlier one: dedicated lessons below
    # take precedence over the general cell rules for the same function.
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
        "log_softmax",
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
register(list(ACTIVATIONS), describe_activation)
register(["mean", "sum"], describe_reduction)
register(["softmax"], describe_softmax)


def describe_operation(
    kind: str, arguments: dict, inputs: list[TensorState], outputs: list[TensorState]
):
    adapter = ADAPTERS.get(kind)
    if adapter and inputs and outputs:
        try:
            lesson = adapter(kind, arguments, inputs, outputs)
        except (KeyError, ValueError, TypeError, IndexError, RuntimeError):
            # A valid execution can always be inspected, even without a matching lesson.
            lesson = None
        if lesson and lesson.relation is None and kind in RELATION_KINDS:
            # A dedicated view keeps its interaction; the cell rule rides along
            # for features that read rules directly, such as axis lineage.
            try:
                lesson.relation, mapping = relation_for(kind, arguments, inputs, outputs)
                lesson.mapping = lesson.mapping or mapping
            except (KeyError, ValueError, TypeError, IndexError, RuntimeError):
                pass
        if lesson:
            return lesson
    return Lesson(
        title=f"Inspect {kind}",
        summary="Recorded from the real PyTorch execution.",
        detail="A specialized lesson is not available for this operation. Its inputs, outputs, source, and tensor metadata remain available.",
        category="generic",
    )
