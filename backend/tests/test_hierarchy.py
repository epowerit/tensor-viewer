import pytest
import torch
from torch.nn import functional as F

from tensorviewer.composer import canonical_project, compose
from tensorviewer.hierarchy import hierarchy_budget, merge_parameters, window_budget
from tensorviewer.models import (
    Blueprint,
    ComponentSpec,
    CompositionRequest,
    InputSpec,
    ProjectDraft,
)
from tensorviewer.worker import execute


def specification(shape=(2, 3, 8, 16), kinds=None, mode="values", dtype="float32"):
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
                ComponentSpec(id=f"s-{i}", kind=k, parameters=p)
                for i, (k, p) in enumerate(kinds or [("hierarchical_vit", {})])
            ],
        ),
    )


def draft(spec):
    return canonical_project(
        ProjectDraft(
            name="Hierarchy",
            code="placeholder",
            input=spec.input,
            capture_mode=spec.capture_mode,
            blueprint=spec.blueprint,
        )
    )


def model_for(spec):
    plan = compose(spec)
    assert plan.valid, plan.error
    namespace = {}
    exec(plan.code, namespace)
    torch.manual_seed(spec.input.seed)
    model = namespace["ComposedModel"]().eval()
    if spec.input.dtype == "float64":
        model.double()
    return model, namespace


def reference_block(block, tokens):
    features = tokens.shape[-1]
    normalized = F.layer_norm(tokens, (features,), block.norm1.weight, block.norm1.bias)
    attention = block.attention
    q, k, v = [
        F.linear(normalized, layer.weight)
        .reshape(1, tokens.shape[1], attention.num_heads, attention.head_dim)
        .transpose(1, 2)
        for layer in [attention.query, attention.key, attention.value]
    ]
    attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(tokens)
    residual = tokens + F.linear(attended, attention.projection.weight)
    n = F.layer_norm(residual, (features,), block.norm2.weight, block.norm2.bias)
    first, last = block.feedforward[0], block.feedforward[2]
    return residual + F.linear(
        F.gelu(F.linear(n, first.weight, first.bias)), last.weight, last.bias
    )


def reference_windows(module, x):
    """Slice windows directly and run them separately; no partition/restore helper calls."""
    out = torch.empty_like(x)
    size = module.partition.window
    for b in range(x.shape[0]):
        for row in range(0, x.shape[2], size):
            for col in range(0, x.shape[3], size):
                local = x[b, :, row : row + size, col : col + size]
                tokens = local.flatten(1).t().unsqueeze(0)
                attended = reference_block(module.block, tokens)
                out[b, :, row : row + size, col : col + size] = attended[0].t().reshape_as(local)
    return out


def reference_merge(module, x):
    grid = x.permute(0, 2, 3, 1)
    joined = torch.cat(
        [grid[:, 0::2, 0::2], grid[:, 1::2, 0::2], grid[:, 0::2, 1::2], grid[:, 1::2, 1::2]], dim=-1
    )
    normalized = F.layer_norm(joined, (joined.shape[-1],), module.norm.weight, module.norm.bias)
    return F.linear(normalized, module.reduction.weight).permute(0, 3, 1, 2)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_partition_and_restore_preserve_every_cell_for_rectangular_noncontiguous_inputs(dtype):
    spec = specification((2, 3, 4, 6), [("window_partition", {})], dtype=dtype)
    model, ns = model_for(spec)
    x = torch.arange(144, dtype=getattr(torch, dtype)).reshape(2, 3, 6, 4).transpose(2, 3)
    windows = model(x)
    assert not x.is_contiguous()
    assert windows.shape == (12, 4, 3)
    for b in range(2):
        for r in range(2):
            for c in range(3):
                expected = x[b, :, 2 * r : 2 * r + 2, 2 * c : 2 * c + 2].flatten(1).t()
                torch.testing.assert_close(windows[b * 6 + r * 3 + c], expected)
    torch.testing.assert_close(ns["WindowReverse"](4, 6, 2)(windows), x)


def test_grouping_preserves_neighbor_order_and_merge_matches_slicing_reference():
    spec = specification((2, 3, 4, 6), [("patch_merging", {})])
    model, ns = model_for(spec)
    x = torch.arange(144).reshape(2, 3, 4, 6).float()
    grouped = ns["PatchGrouping"]()(x)
    assert grouped.shape == (2, 2, 3, 12)
    assert grouped[1, 1, 1].tolist() == [
        x[1, c, y, z].item() for y, z in [(2, 2), (3, 2), (2, 3), (3, 3)] for c in range(3)
    ]
    torch.testing.assert_close(model(x), reference_merge(model.stage_0, x))
    assert sum(p.numel() for p in model.parameters()) == merge_parameters(3)


def test_attention_is_independent_across_windows_and_batches():
    spec = specification((2, 4, 4, 6), [("window_attention", {"heads": 2})])
    model, _ = model_for(spec)
    x = torch.randn(spec.input.shape)
    expected = reference_windows(model.stage_0, x)
    torch.testing.assert_close(model(x), expected)
    changed = x.clone()
    changed[1, 0, 0, 0] += 10
    actual = model(changed)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1, :, 2:], expected[1, :, 2:])
    torch.testing.assert_close(actual[1, :, :2, 2:], expected[1, :, :2, 2:])
    assert not torch.allclose(actual[1, :, :2, :2], expected[1, :, :2, :2])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complete_hierarchy_matches_independent_functional_reference_and_records_stages(dtype):
    spec = specification(dtype=dtype)
    model, _ = model_for(spec)
    x = torch.randn(spec.input.shape, dtype=getattr(torch, dtype))
    hierarchy = model.stage_0
    with torch.no_grad():
        conv = hierarchy.embedding.projection
        grid = F.conv2d(x, conv.weight, conv.bias, stride=2)
        fine = reference_windows(hierarchy.fine, grid)
        merged = reference_merge(hierarchy.merge, fine)
        coarse = reference_windows(hierarchy.coarse, merged).permute(0, 2, 3, 1)
        normalized = F.layer_norm(
            coarse, (16,), hierarchy.readout.norm.weight, hierarchy.readout.norm.bias
        )
        expected = F.linear(
            normalized.mean((1, 2)),
            hierarchy.readout.classifier.weight,
            hierarchy.readout.classifier.bias,
        )
    trace = execute(draft(spec))
    assert trace.error is None, trace.error
    out = trace.tensors[trace.output_ids[0]]
    torch.testing.assert_close(
        torch.tensor(out.values, dtype=getattr(torch, dtype)).reshape(out.shape), expected
    )
    shapes = {
        call.path: trace.tensors[call.outputs[0]].shape
        for call in trace.module_calls
        if call.outputs
    }
    assert shapes["stage_0.embedding"] == [2, 8, 4, 8]
    assert shapes["stage_0.fine"] == [2, 8, 4, 8]
    assert shapes["stage_0.fine.partition"] == [16, 4, 8]
    assert shapes["stage_0.merge.group"] == [2, 2, 4, 32]
    assert shapes["stage_0.merge"] == [2, 16, 2, 4]
    assert shapes["stage_0.coarse"] == [2, 16, 2, 4]
    assert shapes["stage_0.readout"] == [2, 10]
    assert len(trace.operations) < 100
    assert not trace.warnings


def test_reusable_sequence_matches_full_hierarchy_with_identical_weights():
    full, _ = model_for(specification())
    spec = specification(
        kinds=[
            ("spatial_embedding", {}),
            ("window_attention", {"heads": 2}),
            ("patch_merging", {}),
            ("window_attention", {"heads": 2}),
            ("spatial_readout", {}),
        ]
    )
    modular, _ = model_for(spec)
    for target, source in zip(modular.children(), full.stage_0.children()):
        target.load_state_dict(source.state_dict())
    x = torch.randn(spec.input.shape)
    torch.testing.assert_close(modular(x), full(x))


def test_large_shape_capture_and_preallocation_limits_count_local_attention_correctly():
    spec = specification((1024, 3, 128, 128), mode="shapes")
    trace = execute(draft(spec))
    assert trace.error is None, trace.error
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 250_000
    assert trace.tensors[trace.output_ids[0]].shape == [1024, 10]
    assert (
        "intermediate"
        in compose(specification((512, 3, 64, 64), [("hierarchical_vit", {"window": 4})])).error
    )
    assert (
        "weights" in compose(specification(kinds=[("hierarchical_vit", {"features": 512})])).error
    )
    model, _ = model_for(specification())
    parameters, _ = hierarchy_budget(
        [2, 3, 8, 16], patch=2, features=8, window=2, heads=2, expansion=2, classes=10
    )
    assert parameters == sum(p.numel() for p in model.parameters())
    _, peak = window_budget([128, 8, 64, 64], window=4, heads=2, expansion=2)
    assert peak == 128 * 256 * 2 * 16 * 16


@pytest.mark.parametrize(
    "shape,kind,params,contains",
    [
        ([2, 3, 8], "hierarchical_vit", {}, "image"),
        ([2, 3, 8, 12], "hierarchical_vit", {}, "both stages"),
        ([2, 3, 8, 8], "hierarchical_vit", {"heads": 3}, "heads"),
        ([2, 3, 5, 6], "window_partition", {}, "divisible"),
        ([2, 3, 4, 6], "window_attention", {"heads": 2}, "heads"),
        ([2, 3, 5, 6], "patch_merging", {}, "even"),
        ([2, 3, 5, 6], "spatial_embedding", {}, "patch"),
        ([7, 4, 3], "window_reverse", {}, "windows"),
        ([8, 3, 3], "window_reverse", {}, "windows"),
        ([8, 4, 3], "window_reverse", {"height": 5}, "divisible"),
    ],
)
def test_invalid_connections_are_explained_before_execution(shape, kind, params, contains):
    plan = compose(specification(shape, [(kind, params)]))
    assert not plan.valid and contains in plan.error


def test_integer_layout_components_remain_reusable_and_learned_components_reject_integers():
    spec = specification(
        (2, 3, 4, 6),
        [("window_partition", {}), ("window_reverse", {"height": 4, "width": 6})],
        dtype="int64",
    )
    trace = execute(draft(spec))
    assert trace.error is None, trace.error
    assert trace.tensors[trace.output_ids[0]].values == list(range(144))
    for kind in [
        "hierarchical_vit",
        "window_attention",
        "patch_merging",
        "spatial_embedding",
        "spatial_readout",
    ]:
        assert (
            "floating-point"
            in compose(specification((2, 3, 8, 8), [(kind, {})], dtype="int64")).error
        )


def test_runtime_checks_remain_when_generated_code_is_used_separately():
    model, ns = model_for(specification())
    assert model(torch.zeros(1, 3, 8, 16)).shape == (1, 10)
    with pytest.raises(ValueError, match="configured"):
        model(torch.zeros(2, 3, 16, 8))
    with pytest.raises(ValueError, match="even"):
        ns["PatchGrouping"]()(torch.zeros(1, 3, 3, 4))
    with pytest.raises(ValueError, match="Window reverse"):
        ns["WindowReverse"](4, 4, 2)(torch.zeros(7, 4, 3))
