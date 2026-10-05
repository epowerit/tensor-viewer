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
