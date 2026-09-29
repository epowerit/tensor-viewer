import numpy as np
import pytest
import torch

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.worker import execute


def project(shape=(2, 3, 4), bias=True, mode="values", forward=None):
    return ProjectDraft(
        name="Linear study",
        class_name="Projection",
        constructor={},
        code=f"""import torch
from torch import nn
class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear({shape[-1]}, 5, bias={bias})
    def forward(self, x):
        {forward or "return self.projection(x)"}
""",
        input=InputSpec(shape=list(shape), axis_names=[]),
        capture_mode=mode,
    )


@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4), (2, 3, 2, 4)])
@pytest.mark.parametrize("bias", [True, False])
def test_each_output_uses_one_weight_row_and_preserves_leading_coordinates(shape, bias):
    trace = execute(project(shape, bias))
    assert trace.error is None
    op = trace.operations[0]
    assert op.kind == "linear"
    assert op.lesson.interaction == "linear_projection"
    x, w = [trace.tensors[i] for i in op.inputs[:2]]
    y = trace.tensors[op.outputs[0]]
    assert y.shape == [*shape[:-1], 5]
    assert w.role == "parameter"
    input_tensor = torch.tensor(x.values).reshape(x.shape)
    weight = torch.tensor(w.values).reshape(w.shape)
    offset = torch.tensor(trace.tensors[op.inputs[2]].values) if bias else None
    expected = torch.nn.functional.linear(input_tensor, weight, offset)
    actual = torch.tensor(y.values).reshape(y.shape)
    torch.testing.assert_close(actual, expected)
    prefix = tuple(s - 1 for s in shape[:-1])
    feature = 3
    selected = (input_tensor[prefix] * weight[feature]).sum()
    if bias:
        selected += offset[feature]
    torch.testing.assert_close(selected, actual[(*prefix, feature)])


def test_keyword_operands_remain_input_weight_bias():
    draft = project(
        forward="return torch.nn.functional.linear(bias=self.projection.bias, weight=self.projection.weight, input=x)"
    )
    trace = execute(draft)
    assert trace.error is None
    op = trace.operations[0]
    assert [trace.tensors[i].shape for i in op.inputs] == [[2, 3, 4], [5, 4], [5]]
    assert op.lesson.interaction == "linear_projection"


def test_noncontiguous_input_keeps_logical_snapshot_order():
    trace = execute(project(forward="return self.projection(x.transpose(0, 1))"))
    assert trace.error is None
    op = trace.operations[-1]
    x, w, b = [trace.tensors[i] for i in op.inputs]
    assert not x.contiguous
    assert op.lesson.interaction == "linear_projection"
    expected = torch.nn.functional.linear(
        torch.tensor(x.values).reshape(x.shape),
        torch.tensor(w.values).reshape(w.shape),
        torch.tensor(b.values),
    )
    actual = trace.tensors[op.outputs[0]]
    torch.testing.assert_close(torch.tensor(actual.values).reshape(actual.shape), expected)


@pytest.mark.parametrize(
    "forward",
    [
        "return torch.nn.functional.linear(x, self.projection.weight[0])",
        "return torch.nn.functional.linear(x, self.projection.weight, torch.tensor(1.0))",
        "return self.projection(x[:0])",
    ],
)
def test_unhandled_variants_remain_inspectable(forward):
    trace = execute(project(forward=forward))
    assert trace.error is None
    op = trace.operations[-1]
    assert op.kind == "linear"
    assert op.lesson.interaction == "inspect"


def test_large_paged_projection_keeps_actual_values(tmp_path):
    trace = execute(project((2, 4, 1024)), tmp_path)
    assert trace.error is None
    op = trace.operations[0]
    assert op.lesson.interaction == "linear_projection"
    x, w, b = [trace.tensors[i] for i in op.inputs]
    assert x.value_source == w.value_source == "paged"
    input_values = np.load(tmp_path / f"{x.id}.npy").reshape(x.shape)
    weight_values = np.load(tmp_path / f"{w.id}.npy").reshape(w.shape)
    y = trace.tensors[op.outputs[0]]
    expected = (input_values[1, 3] * weight_values[4]).sum() + b.values[4]
    assert np.array(y.values).reshape(y.shape)[1, 3, 4] == pytest.approx(expected, rel=1e-5)


def test_huge_shape_projection_has_no_fabricated_values():
    trace = execute(project((1024, 1024, 768), mode="shapes"))
    assert trace.error is None
    assert trace.operations[0].lesson.interaction == "linear_projection"
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 10000
