import pytest
import torch
from torch.nn import functional as F

from tensorviewer.composer import canonical_project, compose
from tensorviewer.models import (
    Blueprint,
    ComponentSpec,
    CompositionRequest,
    InputSpec,
    ProjectDraft,
)
from tensorviewer.vision import vision_budget
from tensorviewer.worker import execute


def specification(shape=(2, 3, 8, 12), components=None, mode="values", dtype="float32"):
    return CompositionRequest(
        input=InputSpec(
            shape=list(shape),
            axis_names=[],
            generator="arange" if dtype == "int64" else "random",
            dtype=dtype,
        ),
        capture_mode=mode,
        blueprint=Blueprint(
            has_input=True,
            components=[
                ComponentSpec(id=f"stage-{i}", kind=kind, parameters=parameters)
                for i, (kind, parameters) in enumerate(components or [("vit", {})])
            ],
        ),
    )


def project(spec):
    return canonical_project(
        ProjectDraft(
            name="ViT study",
            code="placeholder",
            input=spec.input,
            blueprint=spec.blueprint,
            capture_mode=spec.capture_mode,
        )
    )


def instantiate(code, seed=42):
    namespace = {}
    exec(code, namespace)
    torch.manual_seed(seed)
    return namespace["ComposedModel"]().eval()


def reference(model, x):
    """Independent functional path using recorded model weights, not its forward methods."""
    projection = model.patches.projection
    patches = F.conv2d(x, projection.weight, projection.bias, stride=projection.stride)
    patches = patches.flatten(2).transpose(1, 2)
    x = torch.cat([model.tokens.class_token.expand(x.shape[0], -1, -1), patches], 1)
    x = x + model.tokens.positions
    for block in model.encoder:
        a = block.attention
        normalized = F.layer_norm(x, (x.shape[-1],), block.norm1.weight, block.norm1.bias)
        batch, tokens, features = normalized.shape
        q, k, v = [
            F.linear(normalized, layer.weight)
            .reshape(batch, tokens, a.num_heads, a.head_dim)
            .transpose(1, 2)
            for layer in [a.query, a.key, a.value]
        ]
        attended = torch.softmax((q @ k.transpose(-1, -2)) / a.head_dim**0.5, -1) @ v
        x = x + F.linear(
            attended.transpose(1, 2).reshape(batch, tokens, features), a.projection.weight
        )
        n = F.layer_norm(x, (features,), block.norm2.weight, block.norm2.bias)
        first, last = block.feedforward[0], block.feedforward[2]
        x = x + F.linear(F.gelu(F.linear(n, first.weight, first.bias)), last.weight, last.bias)
    readout = model.readout
    normalized = F.layer_norm(x, (x.shape[-1],), readout.norm.weight, readout.norm.bias)
    return F.linear(normalized[:, 0, :], readout.classifier.weight, readout.classifier.bias)


@pytest.mark.parametrize("blocks", [1, 4])
def test_full_vit_matches_independent_pytorch_and_keeps_module_boundaries(blocks):
    spec = specification(components=[("vit", {"blocks": blocks})])
    draft = project(spec)
    trace = execute(draft)
    assert trace.error is None, trace.error
    model = instantiate(draft.code, draft.input.seed)
    x = torch.randn(spec.input.shape)
    with torch.no_grad():
        expected = reference(model.stage_0, x)
    out = trace.tensors[trace.output_ids[0]]
    assert out.shape == [2, 10]
    torch.testing.assert_close(torch.tensor(out.values).reshape(out.shape), expected)
    outputs = {
        call.path: trace.tensors[call.outputs[0]].shape
        for call in trace.module_calls
        if call.outputs
    }
    assert outputs["stage_0.patches"] == [2, 6, 8]
    assert outputs["stage_0.tokens"] == [2, 7, 8]
    assert outputs["stage_0.encoder"] == [2, 7, 8]
    assert outputs["stage_0.readout"] == [2, 10]
    assert (
        sum(op.lesson.interaction == "broadcast_add" for op in trace.operations) == 1 + blocks * 2
    )
    norms = [op for op in trace.operations if op.kind == "layer_norm"]
    assert len(norms) == blocks * 2 + 1
    assert all(op.arguments["normalized_shape"] == [8] for op in norms)
    assert len(trace.operations) < 256


def test_token_preparation_preserves_patch_order_and_broadcasts_positions():
    spec = specification((3, 4, 8), [("token_preparation", {})])
    model = instantiate(compose(spec).code)
    prep = model.stage_0
    x = torch.arange(96).reshape(3, 4, 8).float()
    actual = prep(x)
    torch.testing.assert_close(
        actual[:, 0, :], (prep.class_token + prep.positions[:, :1, :]).expand(3, -1, -1)[:, 0]
    )
    torch.testing.assert_close(actual[:, 1:, :], x + prep.positions[:, 1:, :])
    assert torch.count_nonzero(prep.positions) > 0
    assert torch.count_nonzero(prep.class_token) > 0
    with pytest.raises(ValueError, match="configured"):
        prep(torch.zeros(3, 5, 8))


def test_modular_sequence_matches_complete_model_with_the_same_weights():
    full = instantiate(compose(specification()).code).stage_0
    modular_spec = specification(
        components=[
            ("patch_embedding", {"patch": 4, "features": 8}),
            ("token_preparation", {}),
            ("transformer", {}),
            ("class_readout", {}),
        ]
    )
    plan = compose(modular_spec)
    assert [stage.shape for stage in plan.stages] == [[2, 6, 8], [2, 7, 8], [2, 7, 8], [2, 10]]
    modular = instantiate(plan.code)
    for target, source in zip(
        modular.children(), [full.patches, full.tokens, full.encoder, full.readout]
    ):
        target.load_state_dict(source.state_dict())
    x = torch.randn(2, 3, 8, 12)
    torch.testing.assert_close(modular(x), full(x))


def test_budget_matches_actual_parameters_and_includes_cls_in_attention_scores():
    spec = specification(components=[("vit", {"blocks": 2})])
    model = instantiate(compose(spec).code)
    parameters, _ = vision_budget(
        spec.input.shape, patch=4, features=8, heads=2, blocks=2, expansion=2, classes=10
    )
    assert parameters == sum(p.numel() for p in model.parameters())
    # 64 image tokens fit exactly; 65 tokens including CLS exceed the values budget.
    large = specification((1024, 3, 32, 32))
    assert "intermediate" in compose(large).error
    large.capture_mode = "shapes"
    assert compose(large).valid
    trace = execute(project(large))
    assert trace.error is None, trace.error
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 150_000
    assert trace.tensors[trace.output_ids[0]].shape == [1024, 10]
    assert (
        "weights"
        in compose(specification(components=[("vit", {"features": 1024, "blocks": 4})])).error
    )


@pytest.mark.parametrize(
    "shape,parameters,dtype,message",
    [
        ([2, 4, 8], {}, "float32", "image"),
        ([2, 3, 9, 8], {}, "float32", "divisible"),
        ([2, 3, 8, 8], {"heads": 3}, "float32", "heads"),
        ([2, 3, 8, 8], {}, "int64", "floating-point"),
    ],
)
def test_invalid_vit_inputs_are_rejected_before_execution(shape, parameters, dtype, message):
    assert message in compose(specification(shape, [("vit", parameters)], dtype=dtype)).error


def test_fixed_spatial_positions_allow_different_batches_but_not_resolutions():
    model = instantiate(compose(specification()).code)
    assert model(torch.randn(5, 3, 8, 12)).shape == (5, 10)
    with pytest.raises(ValueError, match="configured"):
        model(torch.randn(2, 3, 12, 8))
