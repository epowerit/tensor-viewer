import pytest
import torch

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.worker import execute


def trace_addition(shape, other, expression, mode="values", snapshot_dir=None):
    return execute(
        ProjectDraft(
            name="Addition",
            class_name="Add",
            constructor={},
            input=InputSpec(shape=shape, axis_names=[]),
            capture_mode=mode,
            code=f"""import torch
from torch import nn
class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.right = nn.Parameter(torch.ones({other!r}))
    def forward(self, x):
        return {expression}
""",
        ),
        snapshot_dir,
    )


@pytest.mark.parametrize(
    "shape,other",
    [
        ([2, 3, 4], [1, 3, 4]),
        ([2, 3, 4], [4]),
        ([2, 1, 4], [3, 1]),
        ([2, 3, 4], []),
        ([2, 3, 4], [2, 3, 4]),
    ],
)
@pytest.mark.parametrize("alpha", [1, -2, 0])
def test_addition_operands_and_broadcast_values_are_exact(shape, other, alpha):
    trace = trace_addition(shape, other, f"torch.add(other=self.right, alpha={alpha}, input=x)")
    assert trace.error is None, trace.error
    op = trace.operations[-1]
    assert op.lesson.interaction == "broadcast_add"
    left, right = [trace.tensors[i] for i in op.inputs]
    assert left.shape == shape and right.shape == other
    expected = torch.add(
        torch.tensor(left.values).reshape(shape),
        torch.tensor(right.values).reshape(other),
        alpha=alpha,
    )
    output = trace.tensors[op.outputs[0]]
    torch.testing.assert_close(torch.tensor(output.values).reshape(output.shape), expected)


def test_duplicate_operands_keep_both_contributions():
    trace = trace_addition([2, 3], [3], "x + x")
    op = trace.operations[0]
    assert op.inputs[0] == op.inputs[1]
    assert op.lesson.interaction == "broadcast_add"


def test_scalar_python_operand_empty_output_and_mutation_do_not_claim_addition_lesson():
    for expression in ["x + 3", "x[:0] + self.right", "torch.add(x, self.right, out=x)"]:
        trace = trace_addition([2, 3], [3], expression)
        assert trace.error is None, trace.error
        # A scalar operand uses the general elementwise lesson instead.
        assert trace.operations[-1].lesson.interaction != "broadcast_add"


def test_large_addition_records_shapes_without_materializing_values():
    trace = trace_addition([1024, 1024, 768], [1, 1024, 768], "x + self.right", mode="shapes")
    assert trace.error is None, trace.error
    assert trace.operations[0].lesson.interaction == "broadcast_add"
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 10_000
