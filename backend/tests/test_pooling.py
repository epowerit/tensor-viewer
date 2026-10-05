import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.operations.pooling import pooling_spec
from tensorviewer.operations.registry import describe_operation
from tensorviewer.worker import execute

CASES = [
    (
        "max_pool2d",
        [2, 3, 5, 7],
        {"kernel_size": [3, 2], "stride": [2, 2], "padding": [1, 0], "ceil_mode": True},
    ),
    (
        "max_pool2d_with_indices",
        [2, 2, 5, 6],
        {"kernel_size": 2, "stride": 2, "padding": 1, "dilation": 2, "ceil_mode": True},
    ),
    (
        "avg_pool2d",
        [1, 2, 4, 4],
        {"kernel_size": 3, "stride": 2, "padding": 1, "ceil_mode": True, "count_include_pad": True},
    ),
    (
        "avg_pool2d",
        [1, 2, 4, 4],
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": True,
            "count_include_pad": False,
        },
    ),
    (
        "avg_pool2d",
        [2, 4, 5],
        {
            "kernel_size": [2, 3],
            "stride": [2, 2],
            "padding": [1, 1],
            "ceil_mode": True,
            "divisor_override": 7,
        },
    ),
    ("avg_pool2d", [1, 1, 5, 5], {"kernel_size": [2], "divisor_override": -2}),
    ("adaptive_avg_pool2d", [2, 2, 5, 7], {"output_size": [3, 2]}),
    ("adaptive_avg_pool2d", [2, 3, 4], {"output_size": [8, None]}),
]


def project(kind, shape, args, mode="values", expression=None):
    settings = ", ".join(f"{k}={v!r}" for k, v in args.items())
    expression = expression or f"torch.nn.functional.{kind}(x, {settings})"
    return ProjectDraft(
        name="Pooling study",
        class_name="Example",
        constructor={},
        input=InputSpec(shape=shape, axis_names=[], dtype="float64"),
        capture_mode=mode,
        code=f"import torch\nfrom torch import nn\nclass Example(nn.Module):\n def forward(self,x):\n  return {expression}\n",
    )


@pytest.mark.parametrize("kind,shape,args", CASES)
def test_pooling_capture_and_values_match_native_pytorch(kind, shape, args):
    trace = execute(project(kind, shape, args))
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "pooling"
    assert all(k in op.arguments for k in args)
    inputs, outputs = [trace.tensors[i] for i in op.inputs], [trace.tensors[i] for i in op.outputs]
    assert pooling_spec(op.kind, op.arguments, inputs, outputs)["mode"] in {
        "max",
        "average",
        "adaptive",
    }
    x = torch.tensor(inputs[0].values, dtype=torch.float64).reshape(shape)
    expected = getattr(F, kind)(x, **args)
    for actual, reference in zip(outputs, expected if isinstance(expected, tuple) else [expected]):
        value = torch.tensor(actual.values, dtype=getattr(torch, actual.dtype)).reshape(
            actual.shape
        )
        torch.testing.assert_close(value, reference)
    if kind == "max_pool2d_with_indices":
        assert outputs[1].dtype == "int64"
        flat_spatial = x.reshape(*shape[:-2], -1)
        indices = expected[1].flatten(-2)
        selected = flat_spatial.gather(-1, indices).reshape(expected[0].shape)
        torch.testing.assert_close(selected, expected[0])


def test_positional_arguments_and_keyword_input_are_named():
    draft = project(
        "avg_pool2d",
        [1, 2, 5, 7],
        {},
        expression="torch.nn.functional.avg_pool2d(x, (3,2), (2,1), (1,0), True, False, 7)",
    )
    op = execute(draft).operations[-1]
    assert op.lesson.interaction == "pooling"
    assert op.arguments == {
        "kernel_size": [3, 2],
        "stride": [2, 1],
        "padding": [1, 0],
        "ceil_mode": True,
        "count_include_pad": False,
        "divisor_override": 7,
    }
    draft.code = draft.code.replace(
        "torch.nn.functional.avg_pool2d(x, (3,2), (2,1), (1,0), True, False, 7)",
        "torch.nn.functional.max_pool2d(kernel_size=2, input=x, return_indices=True)",
    )
    trace = execute(draft)
    assert trace.error is None
    assert trace.operations[-1].inputs == trace.input_ids
    assert trace.operations[-1].lesson.interaction == "pooling"


def test_ceil_average_counts_only_declared_padding():
    x = torch.ones(1, 1, 4, 4, dtype=torch.float64)
    with_padding = F.avg_pool2d(x, 3, 2, 1, ceil_mode=True, count_include_pad=True)
    without = F.avg_pool2d(x, 3, 2, 1, ceil_mode=True, count_include_pad=False)
    # Last window starts at (3,3), ends at (5,5) including declared padding,
    # and has one real cell. Its divisor is 4, not the untruncated 3x3 size.
    assert with_padding[0, 0, -1, -1].item() == 1 / 4
    assert without[0, 0, -1, -1].item() == 1
    assert with_padding[0, 0, 0, 0].item() == 4 / 9


def test_adaptive_regions_overlap_and_keep_none_dimensions():
    x = torch.arange(15, dtype=torch.float64).reshape(1, 1, 5, 3)
    y = F.adaptive_avg_pool2d(x, (3, None))
    for row, (start, end) in enumerate([(0, 2), (1, 4), (3, 5)]):
        torch.testing.assert_close(y[:, :, row], x[:, :, start:end].mean(2))


def test_max_padding_ties_and_nonfinite_values_are_recorded_without_invented_indices():
    draft = project(
        "max_pool2d_with_indices",
        [1, 1, 3, 3],
        {},
        expression="torch.nn.functional.max_pool2d(x * 0 - 2, 3, 2, 1, return_indices=True)",
    )
    trace = execute(draft)
    op = trace.operations[-1]
    assert op.lesson.interaction == "pooling"
    assert trace.tensors[op.outputs[0]].values == [-2] * 4
    assert trace.tensors[op.outputs[1]].values == [0, 1, 3, 4]
    draft.code = draft.code.replace("x * 0 - 2", "x * float('nan')")
    trace = execute(draft)
    assert trace.error is None
    assert all(v == "nan" for v in trace.tensors[trace.operations[-1].outputs[0]].values)
    draft = project(
        "max_pool2d_with_indices",
        [1, 1, 1, 1],
        {"kernel_size": 2, "padding": 1, "dilation": 2, "stride": 1},
    )
    trace = execute(draft)
    assert trace.error is None
    assert trace.tensors[trace.operations[-1].outputs[0]].values == ["-inf"]


def test_metadata_windows_and_noncontiguous_tensors():
    trace = execute(
        project(
            "adaptive_avg_pool2d", [1024, 4, 1024, 1024], {"output_size": [7, 5]}, mode="shapes"
        )
    )
    assert trace.error is None
    assert trace.operations[-1].lesson.interaction == "pooling"
    assert all(t.value_source == "shape" for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 8000
    trace = execute(
        project(
            "avg_pool2d",
            [2, 3, 5, 7],
            {},
            expression="torch.nn.functional.avg_pool2d(x.transpose(-1,-2), 2)",
        )
    )
    assert trace.error is None
    assert not trace.tensors[trace.operations[-1].inputs[0]].contiguous
    assert trace.operations[-1].lesson.interaction == "pooling"


def test_invalid_geometry_uses_general_inspection():
    trace = execute(project("avg_pool2d", [1, 1, 4, 4], {"kernel_size": 2}))
    op = trace.operations[-1]
    inputs, outputs = [trace.tensors[i] for i in op.inputs], [trace.tensors[i] for i in op.outputs]
    for args in [
        {"kernel_size": 0},
        {"kernel_size": 2, "stride": 0},
        {"kernel_size": 2, "divisor_override": 0},
        {"kernel_size": 2, "ceil_mode": 1},
        {"kernel_size": 3},
    ]:
        assert describe_operation(op.kind, args, inputs, outputs).interaction == "inspect"


def test_browser_pooling_references_match_native_pytorch():
    path = Path(__file__).parents[2] / "frontend/src/operations/fixtures/pooling.json"
    for case in json.loads(path.read_text()):
        x = torch.tensor(case["input"]["values"], dtype=torch.float64).reshape(
            case["input"]["shape"]
        )
        expected = getattr(F, case["kind"])(x, **case["arguments"])
        expected = expected if isinstance(expected, tuple) else [expected]
        for actual, ref in zip(case["outputs"], expected):
            torch.testing.assert_close(
                torch.tensor(actual["values"], dtype=ref.dtype).reshape(actual["shape"]),
                ref,
                # Indices match exactly; an average to float64 rounding, since
                # another CPU may sum its window in another order.
                rtol=0 if not ref.dtype.is_floating_point else 1e-12,
                atol=0 if not ref.dtype.is_floating_point else 1e-12,
            )


def test_pooling_preserves_axis_roles_for_values_and_indices_and_respects_annotations():
    draft = project("max_pool2d_with_indices", [2, 3, 4, 4], {"kernel_size": 2})
    draft.input.axis_names = ["batch", "channels", "height", "width"]
    trace = execute(draft)
    assert trace.error is None
    assert all(
        trace.tensors[i].axes == draft.input.axis_names for i in trace.operations[-1].outputs
    )
    draft.code = draft.code.rstrip() + " # axes: samples, maps, rows, columns\n"
    trace = execute(draft)
    assert trace.error is None
    assert all(
        trace.tensors[i].axes == ["samples", "maps", "rows", "columns"]
        for i in trace.operations[-1].outputs
    )
