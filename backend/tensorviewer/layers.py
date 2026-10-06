"""A model's stack of repeated blocks, and the states that flow through it."""

from .models import ModuleCall, Trace


def layer_stack(trace: Trace) -> list[ModuleCall]:
    """The model's stack of repeated blocks: the longest run of its top-level
    calls of one module type at numbered paths (blocks.0, blocks.1, …)."""
    root = next((call for call in trace.module_calls if call.parent_id is None), None)
    if root is None:
        return []
    children = [call for call in trace.module_calls if call.parent_id == root.id]
    best: list[ModuleCall] = []
    run: list[ModuleCall] = []
    for call in children:
        numbered = call.path.rsplit(".", 1)[-1].isdigit()
        if run and numbered and call.module_type == run[-1].module_type:
            run.append(call)
        else:
            run = [call] if numbered else []
        if len(run) > len(best):
            best = list(run)
    return best if len(best) >= 2 else []


def layer_states(trace: Trace) -> list[tuple[str, str]]:
    """The state entering the stack and each block's result, as (name,
    tensor id): "before blocks.0", "blocks.0", "blocks.1", …"""
    stack = layer_stack(trace)
    if not stack or not stack[0].inputs or not all(call.outputs for call in stack):
        return []
    return [(f"before {stack[0].path}", stack[0].inputs[0])] + [
        (call.path, call.outputs[0]) for call in stack
    ]


def maker_of(trace: Trace, tensor_id: str) -> tuple[int, int] | None:
    """The step that made a tensor, and which of its results it is."""
    for op in trace.operations:
        if tensor_id in op.outputs:
            return op.index, op.outputs.index(tensor_id)
    return None


def readout(trace: Trace) -> tuple[int, int, str] | None:
    """The model's scores over its vocabulary at each position, as (step,
    which of its results, tensor id): the model's own result when it is such
    scores, or else, for a model that goes on to pick a token (an argmax),
    the last floating-point tensor made after the block stack that keeps the
    stack's leading axes, such as its logits or their softmax."""
    stack = layer_stack(trace)
    if not stack or not stack[-1].outputs or not trace.output_ids:
        return None
    state = trace.tensors.get(stack[-1].outputs[0])
    if state is None:
        return None

    def scores(tensor_id: str) -> bool:
        tensor = trace.tensors.get(tensor_id)
        return (
            tensor is not None
            and tensor.dtype.startswith(("float", "bfloat"))
            and len(tensor.shape) == len(state.shape)
            and tensor.shape[:-1] == state.shape[:-1]
        )

    output = trace.output_ids[0]
    if scores(output) and (made := maker_of(trace, output)):
        return made[0], made[1], output
    for op in reversed(trace.operations):
        if op.index < stack[-1].end_index:
            break
        for at, tensor_id in enumerate(op.outputs):
            if scores(tensor_id):
                return op.index, at, tensor_id
    return None
