import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.composer import canonical_project, compose
from tensorviewer.models import (
    Blueprint,
    ComponentSpec,
    CompositionRequest,
    InputSpec,
    ProjectDraft,
)
from tensorviewer.worker import execute


def request(shape, kinds, **kwargs):
    return CompositionRequest(
        blueprint=Blueprint(
            has_input=True,
            components=[
                ComponentSpec(id=f"node-{i}", kind=kind, parameters=params)
                for i, (kind, params) in enumerate(kinds)
            ],
        ),
        input=InputSpec(shape=shape, axis_names=[], generator="random"),
        **kwargs,
    )


@pytest.mark.parametrize(
    "shape,kinds",
    [
        ([2, 4, 8], [("attention", {"heads": 2})]),
        ([2, 3, 4], [("transformer", {"heads": 2, "blocks": 2})]),
        ([2, 3, 4], [("rnn", {"hidden": 6}), ("linear", {"features": 3})]),
        (
            [2, 3, 5, 6],
            [
                ("conv2d", {"channels": 4, "kernel": 2, "stride": 2}),
                ("tokens", {}),
                ("attention", {}),
            ],
        ),
        (
            [2, 3, 4, 4],
            [
                ("conv2d", {"channels": 2}),
                ("relu", {}),
                ("flatten", {}),
                ("linear", {"features": 5}),
            ],
        ),
    ],
)
def test_composed_shapes_and_values_match_uninstrumented_pytorch(shape, kinds):
    spec = request(shape, kinds)
    plan = compose(spec)
    assert plan.valid, plan.error
    draft = canonical_project(
        ProjectDraft(
            name="Composition", code="placeholder", input=spec.input, blueprint=spec.blueprint
        )
    )
    trace = execute(draft)
    assert trace.error is None, trace.error
    output = trace.tensors[trace.output_ids[0]]
    assert output.shape == plan.stages[-1].shape
    namespace = {}
    exec(draft.code, namespace)
    torch.manual_seed(draft.input.seed)
    model = namespace["ComposedModel"]().eval()
    x = torch.randn(shape)
    with torch.no_grad():
        expected = model(x)
    torch.testing.assert_close(torch.tensor(output.values).reshape(output.shape), expected)


def test_invalid_connections_propagate_without_fake_output_shapes():
    spec = request([2, 3, 8, 8], [("attention", {}), ("linear", {})])
    plan = compose(spec)
    assert not plan.valid
    assert "Needs [batch, tokens, features]" in plan.stages[0].error
    assert plan.stages[0].shape is None and plan.stages[1].shape is None
    spec.blueprint.components.insert(0, ComponentSpec(id="adapter", kind="tokens"))
    spec.blueprint.components[1].parameters = {"heads": 1}
    assert compose(spec).valid


def test_parameter_and_intermediate_budgets_are_checked_before_allocation():
    attention = request([32, 784, 8], [("attention", {})])
    assert "intermediate" in compose(attention).error
    attention.capture_mode = "shapes"
    assert compose(attention).valid
    huge_weights = request([1, 8192], [("linear", {"features": 4096})])
    assert "weights" in compose(huge_weights).error


def test_blank_canvas_and_blueprint_persist_and_code_is_canonical(tmp_path):
    client = TestClient(create_app(tmp_path))
    spec = request([2, 3, 4], [("linear", {"features": 6})])
    draft = ProjectDraft(
        name="Canvas", code="out of date", input=spec.input, blueprint=spec.blueprint
    )
    saved = client.post("/api/v1/projects", json=draft.model_dump()).json()
    assert "nn.Linear(4, 6)" in saved["code"]
    assert saved["class_name"] == "ComposedModel"
    restored = TestClient(create_app(tmp_path)).get(f"/api/v1/projects/{saved['id']}").json()
    assert restored["blueprint"] == spec.blueprint.model_dump()
    assert client.post("/api/v1/compose", json=spec.model_dump()).json()["valid"]
    assert len(client.get("/api/v1/toolbox").json()) >= 8
    spec.blueprint = Blueprint()
    blank = compose(spec)
    assert not blank.valid and blank.stages == []


def test_axis_labels_cannot_escape_generated_comment():
    spec = request([2, 3], [("relu", {})])
    spec.input.axis_names = ["batch\nraise RuntimeError('injected')", "features"]
    plan = compose(spec)
    namespace = {}
    exec(plan.code, namespace)
    assert namespace["ComposedModel"]()(torch.ones(2, 3)).shape == (2, 3)


@pytest.mark.parametrize(
    "shape,kinds,expected_shape",
    [
        (
            [2, 3, 8, 8],
            [("patch_embedding", {"patch": 4, "features": 6}), ("transformer", {"heads": 2})],
            [2, 4, 6],
        ),
        ([2, 3, 4], [("mlp", {"hidden": 6, "features": 2})], [2, 3, 2]),
        ([2, 3, 7], [("conv1d", {"channels": 4, "kernel": 2, "stride": 2})], [2, 4, 4]),
        (
            [2, 3, 5, 7],
            [("maxpool2d", {}), ("avgpool2d", {}), ("adaptiveavgpool2d", {})],
            [2, 3, 1, 1],
        ),
        ([2, 3, 4, 4], [("batchnorm2d", {}), ("layernorm", {})], [2, 3, 4, 4]),
        (
            [2, 3, 4],
            [("gelu", {}), ("sigmoid", {}), ("tanh", {}), ("softmax", {"axis": 1})],
            [2, 3, 4],
        ),
        (
            [2, 3, 8],
            [
                ("split_axis", {"factor": 2}),
                ("transpose", {}),
                ("contiguous", {}),
                ("merge_axes", {}),
            ],
            [2, 3, 8],
        ),
        (
            [2, 3, 4],
            [
                ("unsqueeze", {"axis": -1}),
                ("squeeze", {"axis": -1}),
                ("mean", {"axis": 1, "keepdim": 1}),
            ],
            [2, 1, 4],
        ),
        ([2], [("mean", {}), ("unsqueeze", {"axis": 0})], [1]),
        ([2, 3, 8], [("unfold", {"window": 3, "step": 2})], [2, 3, 3, 3]),
    ],
)
def test_expanded_library_executes_with_expected_shapes(shape, kinds, expected_shape):
    spec = request(shape, kinds)
    plan = compose(spec)
    assert plan.valid, plan.error
    assert plan.stages[-1].shape == expected_shape
    test_composed_shapes_and_values_match_uninstrumented_pytorch(shape, kinds)


@pytest.mark.parametrize(
    "shape,kind,params,message",
    [
        ([2, 3, 7, 8], "patch_embedding", {}, "divisible"),
        ([2, 3, 1, 1], "maxpool2d", {}, "window"),
        ([2, 3, 4], "transpose", {"axis_a": 3}, "Axis"),
        ([2, 3, 7], "split_axis", {"factor": 2}, "evenly"),
        ([2, 3, 4], "merge_axes", {"axis": -1}, "immediately after"),
        ([2, 3, 4], "squeeze", {}, "size one"),
        ([1, 1, 1, 1, 1, 1], "unsqueeze", {}, "six"),
        ([2, 3], "unfold", {"window": 4}, "window"),
    ],
)
def test_invalid_new_component_settings_are_explained(shape, kind, params, message):
    plan = compose(request(shape, [(kind, params)]))
    assert not plan.valid
    assert message in plan.error
    assert plan.stages[0].shape is None


def test_reduction_to_scalar_and_integer_layer_mismatches_are_validated():
    scalar = compose(request([2], [("mean", {}), ("linear", {})]))
    assert not scalar.valid and "feature axis" in scalar.error
    spec = request([2, 4], [("layernorm", {})])
    spec.input.dtype, spec.input.generator = "int64", "arange"
    assert not compose(spec).valid
    large = request([2, 4, 4096], [("mlp", {"hidden": 4096, "features": 4096})])
    assert "weights" in compose(large).error


def test_all_catalog_components_have_an_executable_default():
    from tensorviewer.composer import CATALOG

    for item in CATALOG:
        kind = item["kind"]
        shape = [2, 4, 8]
        if item["group"] == "Spatial" or kind in {
            "patch_embedding",
            "tokens",
            "batchnorm2d",
            "vit",
        }:
            shape = [2, 3, 8, 8]
        elif kind == "squeeze":
            shape = [2, 1, 8]
        elif kind == "hierarchical_vit":
            shape = [2, 3, 8, 8]
        elif kind == "window_reverse":
            shape = [8, 4, 8]
        spec = request(shape, [(kind, {})])
        plan = compose(spec)
        assert plan.valid, (kind, plan.error)
        namespace = {}
        exec(plan.code, namespace)
        output = namespace["ComposedModel"]().eval()(torch.randn(shape))
        assert list(output.shape) == plan.stages[0].shape, kind
