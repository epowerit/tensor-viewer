from tensorviewer.declarations import read_project
from tensorviewer.worker import execute

MODEL = """
import math
import torch
from torch import nn

# input x: batch=1, tokens=5, features=8


class Model(nn.Module):
    def forward(self, x):
        q = x.reshape(1, 5, 2, 4).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        scores = q @ q.transpose(-2, -1) / math.sqrt(4)  # axes: batch, heads, queries, keys
        allowed = torch.ones(5, 5).tril().bool()  # axes: queries, keys
        return scores.masked_fill(~allowed, 0.0)
"""


def test_annotations_name_the_statement_value_and_follow_it_back():
    trace = execute(read_project(MODEL).draft)
    axes = [(op.kind, trace.tensors[op.outputs[0]].axes) for op in trace.operations]
    assert axes == [
        # Before the transpose, heads and tokens are the other way round.
        ("reshape", ["batch", "tokens", "heads", "head_features"]),
        ("transpose", ["batch", "heads", "tokens", "head_features"]),
        ("transpose", ["batch", "heads", "head_features", "tokens"]),
        # The product feeds only the division, so it already has the axes.
        ("matmul", ["batch", "heads", "queries", "keys"]),
        ("div", ["batch", "heads", "queries", "keys"]),
        ("ones", ["queries", "keys"]),
        ("tril", ["queries", "keys"]),
        ("bool", ["queries", "keys"]),
        ("__invert__", ["queries", "keys"]),
        ("masked_fill", ["batch", "heads", "queries", "keys"]),
    ]


LAYERS = """
import torch
from torch import nn

# input ids: batch=2, tokens=5 | int64


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)
        self.up = nn.Linear(8, 12)

    def forward(self, ids):
        return self.up(self.embed(ids))
"""


def test_embedding_and_linear_keep_the_input_axes():
    trace = execute(read_project(LAYERS).draft)
    # Declared input axes come from the `# input` line.
    assert [trace.tensors[op.outputs[0]].axes for op in trace.operations] == [
        ["batch", "tokens", "features"],
        ["batch", "tokens", "features"],
    ]


GATHER = """
import torch
from torch import nn

# input x: tokens=3, experts=4, features=2


class Model(nn.Module):
    def forward(self, x):
        index = torch.zeros(3, 1, 2, dtype=torch.long)
        return x.gather(1, index)
"""


def test_gather_keeps_the_source_axes():
    trace = execute(read_project(GATHER).draft)
    gather = next(op for op in trace.operations if op.kind == "gather")
    assert trace.tensors[gather.outputs[0]].axes == ["tokens", "experts", "features"]
