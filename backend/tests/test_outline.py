import pytest
import torch

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.outline import choose_outline
from tensorviewer.tracing import MAX_OPERATIONS, OperationLimitError
from tensorviewer.worker import execute

# Twelve blocks of two six-pass mixes: about 300 steps, past the budget.
DEEP = """from torch import nn
class Mix(nn.Module):
    def forward(self, x):
        for _ in range(6):
            x = x * 1.01 + 0.1
        return x
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Mix()
        self.second = Mix()
    def forward(self, x):
        return self.second(self.first(x)) + x
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(12))
    def forward(self, x):
        x = x - 1
        for block in self.blocks:
            x = block(x)
        return x.relu()
"""


def run(code=DEEP, **input):
    return execute(
        ProjectDraft(
            name="Deep",
            code=code,
            class_name="Model",
            constructor={},
            input=InputSpec(shape=[2, 3], axis_names=["batch", "features"], **input),
        )
    )


def expected():
    namespace = {}
    exec(DEEP, namespace)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return namespace["Model"]().eval()(x)


def summarized(trace):
    return [op for op in trace.operations if op.call is not None]


def test_a_run_past_the_budget_summarizes_calls_level_by_level():
    trace = run()
    assert trace.error is None
    assert len(trace.operations) <= MAX_OPERATIONS
    # The model, every block and as many mixes as fit are opened, in order.
    assert trace.outline[0] == "Model#0"
    assert all(f"blocks.{i}#0" in trace.outline for i in range(12))
    calls = summarized(trace)
    assert calls and all(op.kind == "Mix" for op in calls)
    assert all(op.call.module_type == "Mix" and op.call.steps == 12 for op in calls)
    # Opened first, the early mixes are recorded; the last ones summarized.
    assert calls[-1].call.key == "blocks.11.second#0"
    assert "blocks.0.first#0" in trace.outline
    # A summarized call keeps what it took and returned, with true values.
    final = trace.tensors[trace.output_ids[0]]
    assert torch.allclose(torch.tensor(final.values).reshape(2, 3), expected())
    for op in calls:
        assert op.inputs and op.outputs
        assert op.source and "self." in op.source.text
        # Shaped like what it took, its result keeps those axes' names.
        taken = trace.tensors[op.inputs[0]]
        assert trace.tensors[op.outputs[0]].axes == taken.axes


def test_opening_a_summarized_call_records_the_steps_inside():
    first = run()
    key = summarized(first)[-1].call.key
    opened = run(outline=first.outline, focus=key)
    assert opened.error is None
    assert key in opened.outline
    assert key not in [op.call.key for op in summarized(opened)]
    assert len(opened.operations) <= MAX_OPERATIONS
    # Room is made by summarizing others, latest first.
    assert summarized(opened)


def test_a_run_records_by_the_outline_it_is_given():
    first = run()
    again = run(outline=first.outline)
    assert again.outline == first.outline
    assert [(op.kind, op.call and op.call.key) for op in again.operations] == [
        (op.kind, op.call and op.call.key) for op in first.operations
    ]


def test_a_model_within_the_budget_records_every_step():
    small = DEEP.replace("range(12)", "range(2)")
    trace = run(small)
    assert trace.error is None
    assert trace.outline is None
    assert not summarized(trace)
    assert len(trace.operations) == 2 * 25 + 2


def test_choosing_opens_the_focus_first_and_then_fills_level_by_level():
    # root -> a (5 own steps, children a.x, a.y of 10 each) and b (30 own).
    parents = {"root#0": None, "a#0": "root#0", "a.x#0": "a#0", "a.y#0": "a#0", "b#0": "root#0"}
    steps = {"root#0": 60, "a#0": 25, "a.x#0": 10, "a.y#0": 10, "b#0": 30}
    base = 2 + 2  # the root's own steps and one per call it makes
    assert choose_outline(base, steps, parents, budget=100) == list(parents)
    # Tight: a opens (cost 5 + 2 - 1), b does not fit (cost 29).
    assert choose_outline(base, steps, parents, budget=20) == ["root#0", "a#0", "a.x#0"]
    # The focus is opened first, even past what level order would choose.
    assert choose_outline(base, steps, parents, focus=["b#0"], budget=40) == [
        "root#0",
        "a#0",
        "b#0",
    ]
    with pytest.raises(OperationLimitError):
        choose_outline(base, steps, parents, focus=["b#0"], budget=10)


def test_summarized_calls_fold_as_a_loop_of_repeated_blocks():
    trace = run()
    calls = summarized(trace)
    # Each summarized mix sits in the loop over blocks, so the canvas folds
    # repeated passes as it does recorded ones.
    assert all(op.loops for op in calls)


def test_a_module_returning_several_tensors_keeps_each_output():
    code = """import torch
from torch import nn
class Split(nn.Module):
    def forward(self, x):
        for _ in range(140):
            x = x + 0.0
        return x.chunk(2, dim=0)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.split = Split()
        self.again = Split()
    def forward(self, x):
        a, b = self.split(x)
        c, d = self.again(torch.cat([a, b]))
        return c + d
"""
    trace = run(code)
    assert trace.error is None
    # The first split fits and is opened; the second is summarized, its two
    # results named by the assignment that takes them.
    [call] = summarized(trace)
    assert call.call.key == "again#0"
    assert [trace.tensors[t].name for t in call.outputs] == ["c", "d"]
