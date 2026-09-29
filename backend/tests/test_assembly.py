import pytest
import torch
from test_hierarchy import draft, specification

from tensorviewer.composer import compose
from tensorviewer.models import ProjectDraft
from tensorviewer.worker import execute


def run(expression, shape=(2, 6, 4), *, metadata=False, prelude=""):
    project = ProjectDraft(
        name="Join and split",
        class_name="Example",
        constructor={},
        code="import torch\nfrom torch import nn\nclass Example(nn.Module):\n def forward(self,x):\n"
        + prelude
        + f"  result = {expression}\n  return result\n",
        input={
            "shape": shape,
            "axis_names": ["batch", "tokens", "features"] if len(shape) == 3 else [],
        },
        capture_mode="shapes" if metadata else "values",
    )
    trace = execute(project)
    assert trace.error is None, trace.error
    return trace, trace.operations[-1]


@pytest.mark.parametrize(
    "expression,reference",
    [
        ("torch.cat([x, x * 10], 1)", lambda x: torch.cat([x, x * 10], 1)),
        ("torch.concat(dim=-1, tensors=[x, x])", lambda x: torch.concat([x, x], -1)),
        ("torch.concatenate([x, x], dim=0)", lambda x: torch.cat([x, x], 0)),
        ("torch.stack([x, x * 2], -2)", lambda x: torch.stack([x, x * 2], -2)),
        ("torch.stack(dim=0, tensors=[x, x])", lambda x: torch.stack([x, x], 0)),
        ("torch.split(x, [1, 0, 5], dim=1)", lambda x: torch.split(x, [1, 0, 5], 1)),
        ("x.split(4, 1)", lambda x: x.split(4, 1)),
        ("torch.split(tensor=x, split_size_or_sections=2, dim=-1)", lambda x: x.split(2, -1)),
        ("torch.chunk(x, 4, 1)", lambda x: x.chunk(4, 1)),
        ("x.unbind(-1)", lambda x: x.unbind(-1)),
    ],
)
def test_assembly_values_operands_and_explicit_axis(expression, reference):
    trace, op = run(expression)
    assert op.lesson.interaction == "tensor_assembly", (op.kind, op.arguments)
    assert "dim" in op.arguments
    expected = reference(torch.arange(48, dtype=torch.float32).reshape(2, 6, 4))
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(op.outputs) == len(expected)
    for tensor_id, value in zip(op.outputs, expected):
        recorded = trace.tensors[tensor_id]
        torch.testing.assert_close(torch.tensor(recorded.values).reshape(recorded.shape), value)
    if op.kind in {"chunk", "split", "unbind"}:
        assert op.inputs == trace.input_ids
        assert all(
            trace.tensors[i].storage_id == trace.tensors[op.inputs[0]].storage_id
            for i in op.outputs
        )
    if op.kind == "chunk":
        assert op.arguments["chunks"] == 4 and len(op.outputs) == 3


def test_axes_follow_actual_inserted_and_removed_dimensions():
    trace, op = run("torch.stack([x, x], 1)")
    assert trace.tensors[op.outputs[0]].axes == ["batch", "stack", "tokens", "features"]
    trace, op = run("x.unbind(1)")
    assert all(trace.tensors[i].axes == ["batch", "features"] for i in op.outputs)
    trace, op = run("x.split(2, 1)")
    assert all(trace.tensors[i].axes == ["batch", "tokens", "features"] for i in op.outputs)
    trace, op = run("torch.cat([x, x], -1)")
    assert trace.tensors[op.outputs[0]].axes == ["batch", "tokens", "features"]


def test_repeated_operands_noncontiguous_parts_and_scalar_stacking():
    trace, op = run("torch.cat([x, x], 1)", prelude="  x = x.transpose(1, 2)\n")
    assert op.inputs[0] == op.inputs[1]
    assert not trace.tensors[op.inputs[0]].contiguous
    expected = torch.cat([torch.arange(48).reshape(2, 6, 4).transpose(1, 2)] * 2, 1)
    output = trace.tensors[op.outputs[0]]
    assert output.values == expected.flatten().tolist()
    trace, op = run("torch.stack([x.sum(), x.sum()], dim=0)")
    assert [trace.tensors[i].shape for i in op.inputs] == [[], []]
    assert trace.tensors[op.outputs[0]].values == [1128, 1128]


def test_large_shape_only_assembly_keeps_descriptions_and_payload_bounded():
    trace, op = run("torch.stack([x, x], dim=1)", (1024, 1024, 768), metadata=True)
    assert op.lesson.interaction == "tensor_assembly"
    assert trace.tensors[op.outputs[0]].shape == [1024, 2, 1024, 768]
    assert not trace.tensors[op.outputs[0]].values
    assert len(trace.model_dump_json()) < 16000


def test_dtype_promotion_and_out_mutations_keep_general_inspection():
    trace, op = run("torch.cat([x, x.to(torch.float64)], 0)")
    assert op.lesson.interaction != "tensor_assembly"
    trace, op = run("torch.cat([x, x], 0, out=dest)", prelude="  dest = torch.zeros(4,6,4)\n")
    assert op.mutations and op.lesson.interaction != "tensor_assembly"


@pytest.mark.parametrize("axis", [0, 1, 3, -1, -4])
def test_stack_branches_shape_and_values(axis):
    spec = specification((2, 3, 4), [("stack_join", {"axis": axis})])
    plan = compose(spec)
    assert plan.valid
    assert plan.stages[0].shape == list(torch.stack([torch.empty(2, 3, 4)] * 2, dim=axis).shape)
    trace = execute(draft(spec))
    assert trace.error is None
    assert trace.operations[-1].lesson.interaction == "tensor_assembly"
    assert trace.operations[-1].inputs[0] == trace.operations[-1].inputs[1]
    output = trace.tensors[trace.output_ids[0]]
    before = trace.tensors[trace.input_ids[0]]
    original = torch.tensor(before.values).reshape(before.shape)
    torch.testing.assert_close(
        torch.tensor(output.values).reshape(output.shape), torch.stack([original, original], axis)
    )


def test_stack_validation_checks_both_inputs_new_rank_and_numeric_budget():
    assert (
        "identical"
        in compose(
            specification((2, 3, 4), [("linear", {"features": 6}), ("stack_join", {})])
        ).error
    )
    assert not compose(specification((2, 3, 4), [("stack_join", {"axis": 4})])).valid
    assert not compose(specification((1, 1, 1, 1, 1, 1), [("stack_join", {})])).valid
    spec = specification((1024, 4096), [("stack_join", {}), ("stack_join", {})])
    spec.blueprint.components[-1].sources = ["s-0", "s-0"]
    assert not compose(spec).valid
    spec.capture_mode = "shapes"
    assert compose(spec).valid
