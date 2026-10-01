from itertools import product
from math import prod

import pytest
import torch

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.worker import execute


def run(script, shape=(2, 3, 4), mode="values", generator="arange"):
    trace = execute(
        ProjectDraft(
            name="relations",
            code="pass",
            script=script,
            capture_mode=mode,
            input=InputSpec(shape=list(shape), axis_names=[], generator=generator),
        )
    )
    assert trace.error is None, trace.error
    return trace


def last(trace, kind=None):
    op = next(o for o in reversed(trace.operations) if kind is None or o.kind == kind)
    return op, [trace.tensors[i] for i in op.inputs], trace.tensors[op.outputs[0]]


def unravel(index, shape):
    coords = []
    for size in reversed(shape):
        coords.append(index % size)
        index //= size
    return coords[::-1]


def ravel(coords, shape):
    index = 0
    for c, size in zip(coords, shape):
        index = index * size + c
    return index


def source(op, before, after, target):
    """An independent evaluator of the recorded rule, as the viewer applies it."""
    relation = op.lesson.relation
    out = unravel(target, after.shape)
    if relation["rule"] == "table":
        return op.lesson.mapping[target]
    if relation["rule"] == "index":
        coords = [
            item["index"]
            if item["kind"] == "int"
            else item["start"] + item["step"] * out[item["out"]]
            for item in relation["axes"]
        ]
    else:
        offset = len(after.shape) - len(before.shape)
        coords = [out[offset + axis] % size for axis, size in enumerate(before.shape)]
    return ravel(coords, before.shape)


def assert_copies(script, shape=(2, 3, 4), kind=None):
    trace = run(script, shape)
    op, inputs, after = last(trace, kind)
    assert op.lesson.interaction == "relation", op.lesson
    before = inputs[op.lesson.relation["operand"]]
    assert after.numel > 0
    for target in range(after.numel):
        assert after.values[target] == before.values[source(op, before, after, target)]
    return op, before, after


def test_basic_indexing_maps_every_output_cell_to_its_source():
    for script in [
        "x[0]",
        "x[-1, 1:, ::2]",
        "x[:, None, ..., -1]",
        "x[..., 1:3]",
        "x[1:, :, 1:4][:, 0:2]",
        "x[None, None, 1, 2, 3]",
        "x[:, 1:3, None, ::3]",
    ]:
        assert_copies(script)
    op, before, after = assert_copies("x[1, ::2]", shape=(3, 5))
    assert after.shape == [3] and after.storage_id == before.storage_id
    assert op.lesson.relation["axes"] == [
        {"kind": "int", "index": 1},
        {"kind": "slice", "start": 0, "step": 2, "out": 0},
    ]


def test_subscripts_without_a_closed_form_have_no_index_rule():
    # Value-dependent subscripts are mapped by replay instead; see the
    # selection test below. They never claim the slice rule.
    for script in ["x[[0, 1]]", "x[torch.tensor([1, 0])]", "x[x > 3]", "x[True]"]:
        op, _, _ = last(run(script), "__getitem__")
        assert (op.lesson.relation or {}).get("rule") != "index", script
    # An empty selection has no cell to follow.
    op, _, after = last(run("x[:, 3:]"), "__getitem__")
    assert after.numel == 0 and op.lesson.relation is None


def test_expand_and_repeat_reuse_input_cells():
    op, before, after = assert_copies("x[:, :1].expand(2, 3, 4)", kind="expand")
    assert after.storage_id == before.storage_id and op.lesson.category == "memory"
    assert_copies("x[0, 0].expand(5, 2, 4)", kind="expand")
    assert_copies("x[0].broadcast_to(3, 3, 4)", kind="broadcast_to")
    op, before, after = assert_copies("x.repeat(2, 1, 3)", kind="repeat")
    assert after.shape == [4, 3, 12] and after.storage_id != before.storage_id
    assert_copies("x[0, 0].repeat(3, 2)", kind="repeat")


def test_value_dependent_lookups_have_exact_small_maps():
    op, _, after = assert_copies(
        "x.gather(2, torch.tensor([[[3, 0], [1, 1], [2, 0]], [[0, 0], [3, 3], [2, 1]]]))",
        kind="gather",
    )
    assert after.shape == [2, 3, 2]
    assert_copies("x.index_select(1, torch.tensor([2, 0, 2]))", kind="index_select")
    op, table, after = assert_copies(
        "table = x.reshape(6, 4)\nF.embedding(torch.tensor([[5, 0], [2, 2]]), table)",
        kind="embedding",
    )
    assert op.lesson.relation["operand"] == 1 and after.shape == [2, 2, 4]
    # Shape-only runs have no index values, so no map is claimed.
    trace = run("x.gather(2, torch.zeros(2, 3, 1, dtype=torch.int64))", mode="shapes")
    assert last(trace, "gather")[0].lesson.relation is None


def test_reduction_groups_match_pytorch():
    cases = {
        "x.sum(dim=-1)": torch.sum,
        "x.mean(dim=(0, 2), keepdim=True)": torch.mean,
        "x.sum()": torch.sum,
        "x.prod(1)": torch.prod,
        "x.amax(dim=(1, 2))": torch.amax,
    }
    for script, function in cases.items():
        trace = run(script)
        op, (before,), after = last(trace)
        relation = op.lesson.relation
        assert relation["rule"] == "reduce", script
        axes, keep = relation["axes"], relation["keepdim"]
        values = torch.tensor(before.values).reshape(before.shape)
        for target in range(after.numel):
            out = unravel(target, after.shape)
            kept = iter(out if not keep else [c for a, c in enumerate(out) if a not in axes])
            fixed = [None if a in axes else next(kept) for a in range(len(before.shape))]
            group = [
                values[tuple(next(free) if c is None else c for c in fixed)]
                for combo in product(*[range(before.shape[a]) for a in axes])
                for free in [iter(combo)]
            ]
            assert len(group) == prod(before.shape[a] for a in axes)
            expected = function(torch.stack(group))
            assert abs(float(expected) - after.values[target]) < 1e-4, script
    op, _, after = last(run("x.max(dim=1)"), "max")
    assert op.lesson.relation["axes"] == [1] and len(op.outputs) == 2
    op, _, after = last(run("x.argmax(-1)"))
    assert op.lesson.relation == {"rule": "reduce", "operand": 0, "axes": [2], "keepdim": False}
    assert after.dtype == "int64"
    # Repeated or out-of-range axes never reach a lesson.
    from tensorviewer.operations.relations import reduced_axes

    assert reduced_axes({"dim": [0, -3]}, 3) is None and reduced_axes({"dim": 3}, 3) is None
    assert reduced_axes({}, 0) == []


def test_elementwise_roles_cover_scalars_broadcasting_and_masks():
    trace = run(
        "m = x > 5\n"
        "a = torch.where(m, x, 0.0)\n"
        "b = x.masked_fill(m, -1.0)\n"
        "c = 2 - x\n"
        "d = x * x[:, :, :1]\n"
        "e = torch.tril(torch.ones(4, 4), diagonal=-1)\n"
        "f = torch.relu(x - 10)\n"
        "g = torch.maximum(x, x.flip(0))"
    )
    roles = {
        op.kind: op.lesson.relation
        for op in trace.operations
        if op.lesson.interaction == "relation" and op.lesson.relation["rule"] == "elementwise"
    }
    assert roles["gt"]["roles"] == ["input"]
    assert roles["where"]["roles"] == ["condition", "input"]
    assert roles["masked_fill"]["roles"] == ["input", "mask"]
    assert roles["__rsub__"] == {
        "rule": "elementwise",
        "operand": 0,
        "roles": ["input"],
        "reversed": True,
    }
    assert roles["mul"]["roles"] == ["input", "other"]
    assert roles["tril"]["diagonal"] == -1
    assert roles["relu"]["roles"] == ["input"] and roles["maximum"]["roles"] == ["input", "other"]
    assert next(op for op in trace.operations if op.kind == "__rsub__").arguments == {"other": 2}
    # Tensor–tensor addition keeps its dedicated lesson.
    add = last(run("x + x[:, :1]"), "add")[0]
    assert add.lesson.interaction == "broadcast_add"
    assert last(run("x + 1"), "add")[0].lesson.relation["roles"] == ["input"]


@pytest.mark.parametrize(
    "script,summary,zeros",
    [
        ("F.dropout(x, p=1, training=True)", "Replace every value with zero", True),
        ("nn.Dropout(1)(x)", "Replace every value with zero", True),
        ("F.dropout(x, p=0.5, training=True)", "Zero values with probability 0.5", False),
        ("F.dropout(x, p=1, training=False)", "Pass values through unchanged", False),
        ("F.dropout(x, p=0, training=True)", "Pass values through unchanged", False),
    ],
)
def test_dropout_lesson_follows_recorded_training_mode(script, summary, zeros):
    op, (before,), after = last(run(script, generator="ones"))
    assert summary in op.lesson.summary
    if zeros:
        assert after.values == [0] * after.numel
        assert "sampled dropout mask" in op.lesson.detail
    elif "unchanged" in summary:
        assert after.values == before.values
    else:
        assert set(after.values) <= {0, 2}
        assert "sampled dropout mask" in op.lesson.detail


@pytest.mark.parametrize("decimals", [None, 0, 2, -1])
def test_round_lesson_uses_recorded_decimal_precision(decimals):
    argument = "" if decimals is None else f", decimals={decimals}"
    op, (before,), after = last(run(f"torch.round(x / 3{argument})"))
    expected = torch.tensor(before.values).round(decimals=decimals or 0).tolist()
    assert after.values == expected
    assert "nearest even" in op.lesson.detail
    if not decimals:
        assert "nearest integer" in op.lesson.summary
    else:
        assert "nearest integer" not in op.lesson.summary
        assert ("2 decimal places" if decimals > 0 else "decimals=-1") in op.lesson.summary


def test_axis_names_follow_selection_and_reduction():
    trace = execute(
        ProjectDraft(
            name="axes",
            code="pass",
            script="a = x[0, :, 1:]\nb = x.sum(dim=1)\nc = x[:, None]\nd = x[0].expand(5, 3, 4)\ne = x > 0",
            input=InputSpec(shape=[2, 3, 4], axis_names=["batch", "tokens", "features"]),
        )
    )
    names = {
        trace.tensors[op.outputs[0]].name: trace.tensors[op.outputs[0]].axes
        for op in trace.operations
    }
    assert names["a"] == ["tokens", "features"]
    assert names["b"] == ["batch", "features"]
    assert names["c"] == ["batch", "axis 1", "tokens", "features"]
    assert names["d"] == ["axis 0", "tokens", "features"]
    assert names["e"] == ["batch", "tokens", "features"]


def test_large_shape_only_relations_stay_constant_size():
    trace = run(
        "a = x[:, 5:900:3]\nb = a.sum(dim=(0, 2))\nc = x[:, :1].expand(1024, 1024, 1024)\nd = c * 2",
        shape=(1024, 1024, 1024),
        mode="shapes",
    )
    for op in trace.operations:
        assert op.lesson.relation is not None and op.lesson.mapping is None, op.kind
    assert trace.tensors[trace.operations[1].outputs[0]].shape == [299]


def test_layout_aliases_share_the_verified_layout_rules():
    trace = run(
        "a = x.swapaxes(0, 1)\nb = x.movedim(0, -1)\nc = x[0].T\nd = x.mT\n"
        "e = x.ravel()\nf = x.unflatten(2, (2, 2))\ng = x.view_as(x)"
    )
    by_name = {trace.tensors[op.outputs[0]].name: op for op in trace.operations}
    assert by_name["a"].lesson.axis_order == [1, 0, 2]
    assert by_name["b"].lesson.axis_order == [1, 2, 0]
    assert by_name["c"].kind == "T" and by_name["c"].lesson.axis_order == [1, 0]
    assert by_name["d"].kind == "mT" and by_name["d"].lesson.axis_order == [0, 2, 1]
    for name in "abcd":
        op = by_name[name]
        before, after = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
        values = torch.tensor(before.values).reshape(before.shape)
        assert after.values == values.permute(op.lesson.axis_order).reshape(-1).tolist()
        assert [after.values[i] for i in range(after.numel)] == [
            before.values[j] for j in op.lesson.mapping
        ]
    for name in "efg":
        assert by_name[name].lesson.mapping_rule == "identity"
        assert by_name[name].lesson.mapping == list(range(24))


def test_selections_by_value_replay_to_an_exact_map():
    for script, kind in [
        ("x[[1, 0]]", "__getitem__"),
        ("x[x > 12]", "__getitem__"),
        ("x[torch.tensor([0, 1]), :, torch.tensor([3, 0])]", "__getitem__"),
        ("x.masked_select(x > 20)", "masked_select"),
        ("x.diagonal(dim1=1, dim2=2)", "diagonal"),
        ("x.flip(0, 2)", "flip"),
        ("x.narrow(2, 1, 2)", "narrow"),
        ("F.pad(x, (2, 1), mode='reflect')", "pad"),
        ("torch.rot90(x, 1, (1, 2))", "rot90"),
    ]:
        op, before, after = assert_copies(script, kind=kind)
        assert op.lesson.relation == {"rule": "table", "operand": 0}, script
    # Nothing is replayed without values, or when a selection is empty.
    assert last(run("x[[1, 0]]", mode="shapes"), "__getitem__")[0].lesson.relation is None
    assert last(run("x[x > 99]"), "__getitem__")[0].lesson.relation is None
    assert last(run("torch.where(x[0, 0] > 1)"), "where")[0].lesson.relation is None


def test_sort_and_topk_follow_their_recorded_positions():
    op, before, after = assert_copies("x.flip(2).sort(dim=-1)", kind="sort")
    assert after.values == sorted(before.values[:4]) + after.values[4:]
    op, _, after = assert_copies("x.topk(2, dim=1)", kind="topk")
    assert after.shape == [2, 2, 4]
    assert last(run("x.sort()", mode="shapes"), "sort")[0].lesson.relation is None


def test_running_totals_and_constant_padding_have_closed_form_rules():
    op, (before,), after = last(run("x.cumsum(1)"))
    assert op.lesson.relation == {"rule": "prefix", "operand": 0, "axis": 1}
    values = torch.tensor(before.values).reshape(before.shape)
    assert after.values == values.cumsum(1).reshape(-1).tolist()
    assert last(run("x.cumsum(-1)"))[0].lesson.relation["axis"] == 2
    op, (before,), after = last(run("F.pad(x, (1, 2, 0, 3), value=7.0)"))
    assert op.lesson.relation == {"rule": "pad", "operand": 0, "before": [0, 0, 1]}
    assert after.shape == [2, 6, 7]
    leading = op.lesson.relation["before"]
    for target in range(after.numel):
        coords = [c - p for c, p in zip(unravel(target, after.shape), leading)]
        inside = all(0 <= c < size for c, size in zip(coords, before.shape))
        expected = before.values[ravel(coords, before.shape)] if inside else 7.0
        assert after.values[target] == expected
    # Cropping with negative padding is not a border.
    assert last(run("F.pad(x, (-1, 0))"), "pad")[0].lesson.relation is None


def test_einsum_letters_are_validated_against_the_operands():
    op, inputs, after = last(run("torch.einsum('bij,bkj->bik', x, x)"))
    assert op.arguments["equation"] == "bij,bkj->bik"
    assert op.lesson.relation == {
        "rule": "einsum",
        "operand": 0,
        "inputs": ["bij", "bkj"],
        "output": "bik",
        "sizes": {"b": 2, "i": 3, "j": 4, "k": 3},
    }
    left = torch.tensor(inputs[0].values).reshape(2, 3, 4)
    for target in range(after.numel):
        b, i, k = unravel(target, after.shape)
        assert abs(float((left[b, i] * left[b, k]).sum()) - after.values[target]) < 1e-3
    # Implicit output, a repeated letter, and a full contraction.
    assert last(run("torch.einsum('ij,jk', x[0], x[0].T)"))[0].lesson.relation["output"] == "ik"
    trace_relation = last(run("torch.einsum('ii', x[0, :, :3])"))[0].lesson.relation
    assert trace_relation["output"] == "" and trace_relation["sizes"] == {"i": 3}
    assert last(run("torch.einsum('...ij->...ji', x)"))[0].lesson.relation is None


def test_replayed_maps_must_reproduce_the_recorded_values():
    for script, kind in [
        ("x.repeat_interleave(2, dim=1)", "repeat_interleave"),
        ("x.tile(1, 1, 2)", "tile"),
        ("F.interpolate(x, scale_factor=2)", "interpolate"),
        ("F.pixel_shuffle(x.reshape(1, 4, 3, 2), 2)", "pixel_shuffle"),
    ]:
        op, _, _ = assert_copies(script, kind=kind)
        assert op.lesson.relation == {"rule": "table", "operand": 0}, script
    # Linear resizing blends neighbors: no output is a copy, so no map is claimed.
    blended = last(run("F.interpolate(x, scale_factor=2, mode='linear')"), "interpolate")[0]
    assert blended.lesson.relation is None and blended.lesson.mapping is None
    assert "copy" not in blended.lesson.summary
    # Equal neighbors would make a blend look like a selection by position alone;
    # the value check is what each map has to pass.
    from tensorviewer.models import TensorState
    from tensorviewer.operations.relations import replayed_selection

    def state(values, shape):
        return TensorState(
            id="t",
            name="t",
            shape=shape,
            axes=[],
            dtype="float32",
            strides=[1] * len(shape),
            storage_id="s",
            storage_offset=0,
            contiguous=True,
            numel=len(values),
            values=values,
        )

    before = state([1.0, 2.0, 3.0], [3])
    flipped = (torch.flip, (torch.zeros(3), [0]), {})
    assert replayed_selection("flip", *flipped, before, state([3.0, 2.0, 1.0], [3])) is not None
    assert replayed_selection("flip", *flipped, before, state([3.0, 2.0, 9.0], [3])) is None


def test_products_one_hot_and_class_losses_have_rules():
    op, _, _ = last(run("torch.dot(x[0, 0], x[0, 1])"))
    assert op.lesson.relation["inputs"] == ["i", "i"] and op.lesson.relation["output"] == ""
    op, _, after = last(run("torch.outer(x[0, 0], x[0, 1])"))
    assert op.lesson.relation["output"] == "ij" and after.shape == [4, 4]
    assert last(run("torch.mv(x[0], x[0, 0])"))[0].lesson.relation["inputs"] == ["ij", "j"]
    op, inputs, after = last(run("F.one_hot(torch.tensor([2, 0, 1]), 4)"))
    assert op.lesson.relation == {"rule": "one_hot", "operand": 0}
    assert op.arguments["num_classes"] == 4 and after.shape == [3, 4]
    assert after.values == [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]

    scores = "x[:, :, 0] / 10"
    op, inputs, after = last(run(f"F.cross_entropy({scores}, torch.tensor([1, 2]))"))
    assert op.lesson.relation == {
        "rule": "class_loss",
        "operand": 0,
        "reduction": "mean",
        "ignore_index": -100,
        "normalized": False,
    }
    logits = torch.tensor(inputs[0].values).reshape(2, 3)
    target = torch.tensor(inputs[1].values)
    expected = -torch.log_softmax(logits, -1)[torch.arange(2), target].mean()
    assert abs(float(expected) - after.values[0]) < 1e-5
    op, _, after = last(
        run(f"F.nll_loss(F.log_softmax({scores}, -1), torch.tensor([1, 2]), reduction='none')")
    )
    assert op.lesson.relation["normalized"] and op.lesson.relation["reduction"] == "none"
    assert after.shape == [2]
    # Weighted, smoothed, and probability-target losses keep general inspection.
    for extra in ["weight=torch.ones(3)", "label_smoothing=0.1"]:
        op = last(run(f"F.cross_entropy({scores}, torch.tensor([1, 2]), {extra})"))[0]
        assert op.lesson.relation is None
    op = last(run(f"F.cross_entropy({scores}, torch.softmax({scores}, -1))"))[0]
    assert op.lesson.relation is None
    logsm = last(run("F.log_softmax(x, dim=-1)"))[0]
    assert logsm.lesson.interaction == "normalization" and "log" in logsm.lesson.title


def test_evaluation_batch_norm_uses_each_channels_stored_statistics():
    script = (
        "mean = torch.tensor([1.0, 2.0, 3.0])\n"
        "var = torch.tensor([4.0, 1.0, 0.25])\n"
        "scale = torch.tensor([1.0, 2.0, 3.0])\n"
        "shift = torch.tensor([0.0, 10.0, 20.0])\n"
        "F.batch_norm(x, mean, var, scale, shift, eps=0.0)"
    )
    op, inputs, after = last(run(script))
    assert op.lesson.relation == {
        "rule": "channel_affine",
        "operand": 0,
        "axis": 1,
        "roles": ["input", "running_mean", "running_var", "weight", "bias"],
        "eps": 0.0,
    }
    source, mean, var, scale, shift = inputs
    for target in range(after.numel):
        channel = unravel(target, after.shape)[1]
        expected = (source.values[target] - mean.values[channel]) / var.values[channel] ** 0.5
        expected = expected * scale.values[channel] + shift.values[channel]
        assert abs(expected - after.values[target]) < 1e-4
    # Without scale and shift only the statistics are operands.
    op, inputs, _ = last(run("F.batch_norm(x, torch.zeros(3), torch.ones(3))"))
    assert op.lesson.relation["roles"] == ["input", "running_mean", "running_var"]
    assert len(inputs) == 3
    # A module in evaluation mode goes through the same call.
    op, _, _ = last(run("layer = nn.BatchNorm1d(3).eval()\nlayer(x)"), "batch_norm")
    assert op.lesson.relation["roles"][-2:] == ["weight", "bias"]
    # Training mode depends on the whole batch, not on stored numbers.
    trained = last(run("F.batch_norm(x, torch.zeros(3), torch.ones(3), training=True)"))[0]
    assert trained.lesson.relation is None and "batch" in trained.lesson.summary
