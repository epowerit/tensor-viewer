import sys

from tensorviewer.declarations import read_project
from tensorviewer.worker import execute

MODEL = """
import torch
from torch import nn

# input x: batch=2, features=4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))


class Looped(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(3))

    def forward(self, x):
        x = x * 2
        for block in self.blocks:
            x = block(x)
        for part in x.split(2, dim=-1):
            for _ in range(2):
                x = x + part.sum()
        steps = 0
        while steps < 2:
            x = x.tanh()
            steps += 1
        for _ in range(2): x = x - 1
        return x
"""


def loops_of(trace):
    return [[(step.line, step.iteration) for step in op.loops] for op in trace.operations]


def test_operations_know_their_loop_iteration():
    trace = execute(read_project(MODEL).draft)
    assert trace.error is None
    ops = trace.operations
    tags = loops_of(trace)
    # Before any loop.
    assert ops[0].kind == "mul" and tags[0] == []
    # Each block's operations, inside Block.forward, belong to the caller's loop.
    blocks = [tag for op, tag in zip(ops, tags) if op.module.endswith("linear")]
    assert [tag[0] for tag in blocks] == [(24, 1), (24, 2), (24, 3)]
    # The iterable is evaluated on the header line, before the loop runs.
    split = next(i for i, op in enumerate(ops) if op.kind == "split")
    assert tags[split] == []
    # Nested loops: the inner count restarts in every outer pass.
    nested = [tag for tag in tags if len(tag) == 2]
    assert [tuple(tag) for tag in nested[::2]] == [
        ((26, 1), (27, 1)),
        ((26, 1), (27, 2)),
        ((26, 2), (27, 1)),
        ((26, 2), (27, 2)),
    ]
    # Each pass of the outer loop is a different run of the inner loop.
    inner_runs = {op.loops[1].id for op in ops if len(op.loops) == 2}
    assert len(inner_runs) == 2
    # A while loop and a loop written on one line.
    assert [tag for op, tag in zip(ops, tags) if op.kind == "tanh"] == [[(30, 1)], [(30, 2)]]
    assert [tag for op, tag in zip(ops, tags) if op.kind == "sub"] == [[(33, 1)], [(33, 2)]]
    assert ops[0].loops == [] and ops[-1].loops[0].text == "for _ in range(2)"


def test_loop_text_is_the_header():
    trace = execute(read_project(MODEL).draft)
    step = next(op for op in trace.operations if op.loops).loops[0]
    assert step.text == "for block in self.blocks"
    assert step.file is None


HOOK = """
import sys
import torch
from torch import nn

# input x: batch=2, features=4


class Hooked(nn.Module):
    def forward(self, x):
        for _ in range(2):
            x = x + 1
            # The recorder pauses its hook only inside its own work.
            print("traced" if sys.gettrace() is not None else "untraced")
        return x
"""


def test_the_loop_hook_is_on_in_user_code_and_gone_after_the_run():
    draft = read_project(HOOK).draft
    trace = execute(draft)
    assert trace.error is None
    assert trace.stdout.split() == ["traced", "traced"]
    header = draft.code.splitlines().index("        for _ in range(2):") + 1
    assert loops_of(trace) == [[(header, 1)], [(header, 2)]]
    assert sys.gettrace() is None
