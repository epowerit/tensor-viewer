import math

import pytest
import torch

from tensorviewer.tracing import Recorder, values_are_plain

FILENAME = "<tensorviewer-project>"
CODE = """import torch
from torch import nn


class Tiny(nn.Module):
    def forward(self, x):
        doubled = x * 2
        return doubled + 1
"""


def recorded(values: bool):
    namespace: dict = {}
    exec(compile(CODE, FILENAME, "exec"), namespace)
    model = namespace["Tiny"]()
    recorder = Recorder(CODE, FILENAME, model, values=values)
    x = torch.arange(6.0).reshape(2, 3)
    recorder.capture(x, "x", role="input")
    with recorder:
        recorder.results = [model(x)]
    return recorder


def test_an_analysis_pass_records_the_steps_without_their_values():
    full, bare = recorded(True), recorded(False)
    # The same steps and tensors, by id, shape and name.
    assert [op.kind for op in bare.trace.operations] == [op.kind for op in full.trace.operations]
    assert len(bare.trace.operations) == 2
    for tensor_id, tensor in full.trace.tensors.items():
        other = bare.trace.tensors[tensor_id]
        assert (other.name, other.shape, other.dtype) == (tensor.name, tensor.shape, tensor.dtype)
    # A full run keeps the values; an analysis pass keeps none.
    assert all(t.value_source == "inline" and t.values for t in full.trace.tensors.values())
    assert all(
        t.value_source == "shape" and not t.values and t.histogram is None
        for t in bare.trace.tensors.values()
    )
    # The live tensors, which the analyses read, are there in both.
    assert torch.equal(bare.results[0], full.results[0])


@pytest.mark.parametrize(
    ("tensor", "plain"),
    [
        (torch.tensor([1.5, -2.0]), True),
        (torch.tensor([1.0, float("nan")]), False),
        (torch.tensor([float("-inf")]).bfloat16(), False),
        (torch.tensor([True, False]), True),
        (torch.tensor([2**53 - 1, -(2**53 - 1)]), True),
        (torch.tensor([2**53]), False),
        (torch.tensor([-(2**63), 1]), False),
        (torch.tensor([-128], dtype=torch.int8), True),
    ],
)
def test_values_needing_no_per_value_check_are_found_from_the_whole_tensor(tensor, plain):
    # Plain: every value is finite and exact in JavaScript, so it is sent as is.
    assert values_are_plain(tensor) is plain
    values = tensor.reshape(-1).tolist()
    exact = all(
        math.isfinite(v) and not (isinstance(v, int) and abs(v) > 2**53 - 1) for v in values
    )
    assert exact is plain
