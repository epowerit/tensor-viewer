import json

import pytest
import torch
from test_hierarchy import draft, model_for, specification
from torch.nn import functional as F

from tensorviewer.composer import compose
from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.window_attention import compose_window
from tensorviewer.worker import execute


def reference(module, x):
    """Gather original coordinates explicitly; no rolls, partition helpers, or model mask."""
    b, c, h, w = x.shape
    m, sy, sx = module.partition.window, module.shift_y, module.shift_x
    attention = module.attention
    result = torch.empty_like(x)
    for batch in range(b):
        for wy in range(h // m):
            for wx in range(w // m):
                coords = [
                    ((wy * m + y + sy) % h, (wx * m + z + sx) % w)
                    for y in range(m)
                    for z in range(m)
                ]
                tokens = torch.stack([x[batch, :, y, z] for y, z in coords]).unsqueeze(0)
                normalized = F.layer_norm(tokens, (c,), module.norm1.weight, module.norm1.bias)
                q, k, v = [
                    F.linear(normalized, layer.weight)
                    .reshape(1, m * m, attention.num_heads, attention.head_dim)
                    .transpose(1, 2)
                    for layer in [attention.query, attention.key, attention.value]
                ]
                bias = torch.empty(attention.num_heads, m * m, m * m, dtype=x.dtype)
                for i, (yi, xi) in enumerate(coords):
                    for j, (yj, xj) in enumerate(coords):
                        if (yi < sy) != (yj < sy) or (xi < sx) != (xj < sx):
                            bias[:, i, j] = float("-inf")
                        else:
                            idx = (i // m - j // m + m - 1) * (2 * m - 1) + (i % m - j % m + m - 1)
                            bias[:, i, j] = attention.adjust.relative_bias_table[idx]
                out = (
                    F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                    .transpose(1, 2)
                    .reshape_as(tokens)
                )
                residual = tokens + F.linear(out, attention.projection.weight)
                norm = F.layer_norm(residual, (c,), module.norm2.weight, module.norm2.bias)
                first, last = module.feedforward[0], module.feedforward[2]
                out = residual + F.linear(
                    F.gelu(F.linear(norm, first.weight, first.bias)), last.weight, last.bias
                )
                for i, (y, z) in enumerate(coords):
                    result[batch, :, y, z] = out[0, i]
    return result


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "shape,window,shift",
    [
        ((2, 4, 4, 6), 2, 1),
        ((1, 4, 6, 9), 3, 2),
        ((1, 4, 2, 6), 2, 1),
        ((1, 4, 2, 2), 2, 1),
        ((1, 4, 4, 4), 2, 0),
    ],
)
def test_shifted_blocks_match_independent_original_coordinate_attention(
    dtype, shape, window, shift
):
    spec = specification(
        shape, [("shifted_window", dict(window=window, shift=shift, heads=2))], dtype=dtype
    )
    model, _ = model_for(spec)
    x = torch.randn(shape, dtype=getattr(torch, dtype)).transpose(2, 3).contiguous().transpose(2, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(x), reference(model.stage_0, x), rtol=1e-5, atol=1e-6)
    t = execute(draft(spec))
    assert t.error is None and not t.warnings
    json.dumps(t.model_dump(), allow_nan=False)
    adjustment = next(c for c in t.module_calls if c.module_type == "WindowScoreAdjustment")
    ops = t.operations[adjustment.start_index : adjustment.end_index]
    weights = t.tensors[ops[-1].outputs[0]]
    mask = t.tensors[ops[4].inputs[1]]
    values = torch.tensor(weights.values).reshape(weights.shape)
    blocked = torch.tensor(mask.values, dtype=torch.bool).reshape(mask.shape)
    grouped = values.reshape(shape[0], blocked.shape[0], 2, window**2, window**2)
    assert (grouped.masked_select(blocked) == 0).all()
    torch.testing.assert_close(values.sum(-1), torch.ones_like(values.sum(-1)))


def test_relative_offsets_share_entries_and_batches_never_mix():
    spec = specification((2, 4, 4, 6), [("window_pair", {"heads": 2})])
    model, _ = model_for(spec)
    x = torch.randn(spec.input.shape)
    with torch.no_grad():
        expected = reference(model.stage_0.shifted, reference(model.stage_0.regular, x))
        torch.testing.assert_close(model(x), expected)
        changed = x.clone()
        changed[1, :, 1, 1] += 10
        torch.testing.assert_close(model(changed)[0], expected[0])
    idx = model.stage_0.shifted.attention.adjust.relative_index
    assert idx[0, 1] == idx[2, 3] and idx[1, 0] != idx[0, 1]
    _, _, _, parameters, _ = compose_window(
        "window_pair", spec.input.shape, dict(window=2, shift=1, heads=2, expansion=2), "float32"
    )
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_mask_stops_wraparound_but_keeps_cross_window_neighbors():
    spec = specification((1, 4, 4, 4), [("shifted_window", {"heads": 2})])
    model, _ = model_for(spec)
    mask = model.stage_0.attention.adjust.blocked
    # First shifted window joins cells from four original regular windows.
    assert not mask[0].any()
    # Last window contains all four corners; only self-pairs remain.
    assert torch.equal(mask[3, 0], ~torch.eye(4, dtype=torch.bool))
    x = torch.randn(spec.input.shape)
    baseline = model(x)
    changed = x.clone()
    changed[0, 0, 0, 0] += 30
    actual = model(changed)
    torch.testing.assert_close(actual[0, :, 3, 3], baseline[0, :, 3, 3])
    changed = x.clone()
    changed[0, 0, 1, 1] += 30
    assert not torch.allclose(model(changed)[0, :, 2, 2], baseline[0, :, 2, 2])


def test_large_meta_pair_is_bounded_and_invalid_allocations_fail_early():
    spec = specification((1024, 8, 64, 64), [("window_pair", {"heads": 2})], mode="shapes")
    trace = execute(draft(spec))
    assert trace.error is None, trace.error
    assert len(trace.operations) == 80
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 300_000
    spec.capture_mode = "values"
    assert not compose(spec).valid
    for shape, p in [
        ((1, 4, 5, 6), {}),
        ((1, 4, 4, 6), {"shift": 2}),
        ((1, 4, 4, 6), {"heads": 3}),
    ]:
        assert not compose(specification(shape, [("shifted_window", p)])).valid


@pytest.mark.parametrize(
    "shifts,dims", [((1, -2), (1, 2)), ((5, -1), (1, 1)), (7, None), ((-99,), (2,))]
)
def test_roll_records_exact_mapping_and_nonfinite_arguments_are_json_safe(shifts, dims):
    code = f"""import torch\nfrom torch import nn\nclass Shift(nn.Module):\n    def forward(self,x):\n        return torch.roll(x, shifts={shifts!r}, dims={dims!r})\n"""
    trace = execute(
        ProjectDraft(
            name="roll",
            code=code,
            class_name="Shift",
            constructor={},
            input=InputSpec(shape=[2, 3, 4], axis_names=[]),
        )
    )
    assert trace.error is None
    op = trace.operations[0]
    expected = (
        torch.roll(torch.arange(24).reshape(2, 3, 4), shifts=shifts, dims=dims).flatten().tolist()
    )
    assert op.lesson.mapping_rule == "roll" and op.lesson.mapping == expected
