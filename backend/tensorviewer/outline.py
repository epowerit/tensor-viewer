"""Which module calls a large model's run records step by step.

A run records at most `MAX_OPERATIONS` steps. Past that, an outline names the
calls recorded step by step; every other call is one summarized step with its
inputs and outputs. The outline is chosen from a counting pass that recorded
only the model's own steps and counted the steps every call ran inside.
"""

from collections.abc import Iterable

from .tracing import MAX_OPERATIONS, OperationLimitError


def choose_outline(
    base: int,
    steps: dict[str, int],
    parents: dict[str, str | None],
    focus: Iterable[str] = (),
    preferred: Iterable[str] = (),
    budget: int = MAX_OPERATIONS,
) -> list[str]:
    """The calls to record step by step, in calling order.

    `base` is how many steps the counting pass recorded: the model's own
    steps and one per call it made. `steps` gives each call (`path#n`) the
    steps it ran inside, recorded or not, and `parents` the call it ran in, in
    calling order. Calls made by no other call are always open. Then `focus`
    is opened (with the calls around it), then `preferred` (the last outline)
    where it fits, then every call level by level in calling order while the
    run stays within `budget` steps.
    """
    children: dict[str, list[str]] = {key: [] for key in parents}
    for key, parent in parents.items():
        if parent is not None:
            children[parent].append(key)
    opened = {key for key, parent in parents.items() if parent is None}
    total = base

    def cost(key: str) -> int:
        # Opening a call replaces its one step with the steps it runs itself
        # and one for each call it makes, all summarized at first.
        own = steps[key] - sum(steps[child] for child in children[key])
        return own + len(children[key]) - 1

    def open_call(key: str) -> bool:
        nonlocal total
        if key in opened:
            return True
        parent = parents[key]
        if parent is not None and not open_call(parent):
            return False
        if total + cost(key) > budget:
            return False
        opened.add(key)
        total += cost(key)
        return True

    for key in focus:
        if key not in parents:
            raise ValueError(f"This run made no call {key}.")
        if not open_call(key):
            raise OperationLimitError(
                f"Recording every step inside {key.split('#')[0]} takes more than {budget} steps."
            )
    for key in preferred:
        if key in parents:
            open_call(key)
    level = [child for key in sorted(opened, key=list(parents).index) for child in children[key]]
    while level:
        deeper = []
        for key in level:
            if open_call(key):
                deeper.extend(children[key])
        level = deeper
    return [key for key in parents if key in opened]
