import json
from pathlib import Path

import pytest
import torch

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.operations.reduction import reduction_spec
from tensorviewer.operations.registry import describe_operation
from tensorviewer.snapshots import json_value, read_snapshot
from tensorviewer.worker import execute


def run(expression, *, shape=(2, 3, 4), dtype="float64", mode="values", snapshot_dir=None):
    trace = execute(
        ProjectDraft(
            name="Reduction study",
            class_name="Example",
            constructor={},
            capture_mode=mode,
            input=InputSpec(
                shape=list(shape), dtype=dtype, axis_names=[f"axis{i}" for i in range(len(shape))]
            ),
            code=f"""import torch
from torch import nn
class Example(nn.Module):
    def forward(self, x):
        return {expression}
""",
        ),
        snapshot_dir=snapshot_dir,
    )
    assert trace.error is None
    return trace, trace.operations[-1]


@pytest.mark.parametrize(
    "expression,axes,keepdim",
    [
        ("x.mean(1)", [1], False),
        ("torch.mean(keepdim=True, dim=(-1, 0), input=x)", [0, 2], True),
        ("x.sum((2, 0))", [0, 2], False),
        ("x.mean(dim=None, keepdim=True)", [0, 1, 2], True),
        ("x.sum()", [0, 1, 2], False),
        ("x.sum(dim=())", [0, 1, 2], False),
        ("x.mean(dim=[], keepdim=True)", [0, 1, 2], True),
        ("x[0, 0, 0].sum(-1, True)", [], True),
        ("x[0, 0, 0].mean(0)", [], False),
        ("x.permute(2, 0, 1).mean((0, -1))", [0, 2], False),
    ],
)
def test_real_reductions_preserve_values_and_unreduced_axis_labels(expression, axes, keepdim):
    trace, op = run(expression)
    assert op.lesson.interaction == "reduction"
    x, y = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
    spec = reduction_spec(op.kind, op.arguments, [x], [y])
    assert spec["axes"] == axes and spec["keepdim"] == keepdim
    assert x.numel == y.numel * spec["size"]
    assert y.axes == [name for i, name in enumerate(x.axes) if keepdim or i not in axes]
    data = torch.tensor(x.values, dtype=torch.float64).reshape(x.shape)
    expected = getattr(data, op.kind)(dim=axes or None, keepdim=keepdim)
    torch.testing.assert_close(
        torch.tensor(y.values, dtype=torch.float64).reshape(y.shape), expected
    )


@pytest.mark.parametrize(
    "expression,dtype,expected_dtype",
    [
        ("x.sum(1)", "int64", "int64"),
        ("x.to(torch.int32).sum(1)", "int64", "int64"),
        ("(x > 3).sum((0, 2))", "float32", "int64"),
        ("x.mean(1, dtype=torch.float64)", "int64", "float64"),
        ("x.sum(1, dtype=torch.float64)", "float32", "float64"),
        ("x.to(torch.bfloat16).mean(1)", "float32", "bfloat16"),
    ],
)
def test_dtype_casts_promotions_and_boolean_counts(expression, dtype, expected_dtype):
    trace, op = run(expression, dtype=dtype)
    assert op.lesson.interaction == "reduction"
    assert trace.tensors[op.outputs[0]].dtype == expected_dtype


def test_empty_and_mutating_reductions_keep_general_inspection():
    trace, op = run("x[:0].mean(0)")
    assert op.lesson.interaction == "inspect"
    assert all(v == "nan" for v in trace.tensors[op.outputs[0]].values)
    trace, op = run("torch.sum(x, dim=1, keepdim=True, out=x[:, :1, :])")
    assert op.mutations and op.lesson.category == "memory"
    assert op.lesson.interaction == "inspect"


def test_explicit_axis_annotations_take_priority():
    trace, op = run("x.mean(1) # axes: images, embedding")
    assert trace.tensors[op.outputs[0]].axes == ["images", "embedding"]


def test_nonfinite_inputs_are_preserved_without_fabricated_statistics():
    trace, op = run("(x / 0).sum(1)")
    assert op.lesson.interaction == "reduction"
    assert trace.tensors[op.outputs[0]].values == [
        "nan",
        "inf",
        "inf",
        "inf",
        "inf",
        "inf",
        "inf",
        "inf",
    ]


def test_large_paged_and_billion_element_shapes_remain_bounded(tmp_path):
    trace, op = run("x.mean(1)", shape=(128, 16, 768), snapshot_dir=tmp_path)
    x, y = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
    assert x.value_source == y.value_source == "paged"
    assert read_snapshot(tmp_path, y.id, [0, y.numel - 1]) == [
        7.5 * 768,
        127 * 16 * 768 + 7.5 * 768 + 767,
    ]
    assert op.lesson.interaction == "reduction" and op.lesson.mapping is None
    trace, op = run("x.sum((0, 2), keepdim=True)", shape=(1024, 1024, 1024), mode="shapes")
    assert op.lesson.interaction == "reduction"
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert trace.tensors[op.outputs[0]].shape == [1, 1024, 1]
    assert len(trace.model_dump_json()) < 10000


def test_invalid_arguments_or_recorded_geometry_do_not_claim_a_lesson():
    trace, op = run("x.sum(1)")
    inputs, outputs = [trace.tensors[op.inputs[0]]], [trace.tensors[op.outputs[0]]]
    for args in [
        {"dim": [1, -2]},
        {"dim": 3},
        {"dim": True},
        {"dim": "tokens"},
        {"dim": 1, "keepdim": None},
        {"dim": 1, "keepdim": 1},
        {"dim": 1, "dtype": "float64"},
        {"dim": 1, "dtype": "torch.int64"},
        {"dim": 1, "out": "tensor"},
        {"dim": 0},
    ]:
        assert describe_operation("sum", args, inputs, outputs).interaction == "inspect"
    for key, value in [("numel", 0), ("shape", [4, 2]), ("dtype", "complex64")]:
        broken = outputs[0].model_copy(update={key: value})
        assert describe_operation("sum", {"dim": 1}, inputs, [broken]).interaction == "inspect"
    assert describe_operation("sum", {"dim": 1}, inputs * 2, outputs).interaction == "inspect"


def test_frontend_reduction_fixtures_match_native_pytorch():
    path = Path(__file__).parents[2] / "frontend/src/operations/fixtures/reductions.json"
    for case in json.loads(path.read_text()):
        dtype = getattr(torch, case["dtype"])
        values = [int(v) if isinstance(v, str) else v for v in case["input"]]
        x = torch.tensor(values, dtype=dtype).reshape(case["shape"])
        args = case["arguments"].copy()
        if args.get("dtype"):
            args["dtype"] = getattr(torch, args["dtype"].removeprefix("torch."))
        y = getattr(x, case["kind"])(**args)
        assert case["output_shape"] == list(y.shape)
        assert case["output_dtype"] == str(y.dtype).removeprefix("torch.")
        assert case["output"] == [json_value(v) for v in y.flatten()]
