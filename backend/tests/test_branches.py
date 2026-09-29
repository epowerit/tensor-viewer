import pytest
import torch
from pydantic import ValidationError
from test_hierarchy import draft, model_for, specification

from tensorviewer.composer import compose
from tensorviewer.models import Blueprint, ComponentSpec
from tensorviewer.worker import execute


def branch_spec(join="add_join"):
    spec = specification(
        (2, 3, 4),
        [("linear", {"features": 4}), ("gelu", {}), ("linear", {"features": 4}), (join, {})],
    )
    spec.blueprint.components[2].sources = ["input"]
    spec.blueprint.components[3].sources = ["s-1", "s-2"]
    return spec


@pytest.mark.parametrize("join", ["add_join", "concat_join"])
def test_branch_values_shapes_and_recorded_dependencies(join):
    spec = branch_spec(join)
    model, _ = model_for(spec)
    x = torch.randn(spec.input.shape)
    a = torch.nn.functional.gelu(model.stage_0(x))
    b = model.stage_2(x)
    expected = a + b if join == "add_join" else torch.cat((a, b), dim=-1)
    torch.testing.assert_close(model(x), expected)
    trace = execute(draft(spec))
    assert trace.error is None, trace.error
    output = trace.tensors[trace.output_ids[0]]
    torch.testing.assert_close(torch.tensor(output.values).reshape(output.shape), expected)
    plan = compose(spec)
    assert plan.stages[2].input_shape == [2, 3, 4]
    assert plan.stages[3].source_shapes == [[2, 3, 4], [2, 3, 4]]
    assert len(trace.operations[-1].inputs) == 2
    assert trace.operations[2].inputs[0] == trace.input_ids[0]


def test_residual_join_and_shape_changing_parallel_branches():
    spec = branch_spec("concat_join")
    spec.blueprint.components[0].parameters["features"] = 6
    assert compose(spec).stages[-1].shape == [2, 3, 10]
    spec.blueprint.components[-1].kind = "add_join"
    assert "identical" in compose(spec).error
    spec = specification((2, 3, 4), [("linear", {"features": 4}), ("add_join", {})])
    assert compose(spec).valid
    model, _ = model_for(spec)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(model(x), model.stage_0(x) + x)


@pytest.mark.parametrize("sources", [["missing"], ["s-2"], ["s-0"], ["input", "input"]])
def test_invalid_edges_are_rejected_before_execution(sources):
    spec = branch_spec()
    spec.blueprint.components[0].sources = sources
    plan = compose(spec)
    assert not plan.valid
    assert plan.stages[0].error
    assert "raise ValueError" in plan.code


def test_concat_checks_axes_and_output_budget():
    spec = branch_spec("concat_join")
    spec.blueprint.components[-1].parameters = {"axis": 6}
    assert not compose(spec).valid
    spec = specification((2, 3, 4), [("linear", {"features": 8}), ("concat_join", {"axis": 1})])
    assert "Concatenation" in compose(spec).error
    spec = specification((1024, 4096), [("concat_join", {}), ("concat_join", {})])
    spec.blueprint.components[-1].sources = ["s-0", "s-0"]
    assert "limit" in compose(spec).error
    spec.capture_mode = "shapes"
    assert compose(spec).stages[-1].shape == [1024, 16384]


def test_component_id_cannot_shadow_project_input():
    with pytest.raises(ValidationError, match="reserved"):
        Blueprint(has_input=True, components=[ComponentSpec(id="input", kind="relu")])
