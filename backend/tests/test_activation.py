import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.operations.registry import describe_operation
from tensorviewer.snapshots import json_value, read_snapshot
from tensorviewer.worker import execute


def run(expression, *, shape=(2, 3, 4), dtype="float64", mode="values", snapshot_dir=None):
    draft = ProjectDraft(
        name="Activation study",
        class_name="Example",
        constructor={},
        capture_mode=mode,
        input=InputSpec(
            shape=list(shape), dtype=dtype, axis_names=[f"axis{i}" for i in range(len(shape))]
        ),
        code=f"""import torch
from torch import nn
from torch.nn import functional as F
class Example(nn.Module):
    def forward(self, x):
        x = x - 5
        return {expression}
""",
    )
    trace = execute(draft, snapshot_dir=snapshot_dir)
    assert trace.error is None
    return trace, trace.operations[-1]


@pytest.mark.parametrize(
    "expression,kind,approximate",
    [
        ("nn.ReLU()(x)", "relu", "none"),
        ("F.relu(input=x)", "relu", "none"),
        ("x.relu()", "relu", "none"),
        ("nn.GELU()(x)", "gelu", "none"),
        ("F.gelu(input=x, approximate='tanh')", "gelu", "tanh"),
        ("nn.GELU(approximate='tanh')(x)", "gelu", "tanh"),
        ("nn.Sigmoid()(x)", "sigmoid", "none"),
        ("x.sigmoid()", "sigmoid", "none"),
        ("nn.Tanh()(x)", "tanh", "none"),
        ("torch.tanh(input=x)", "tanh", "none"),
    ],
)
def test_module_function_and_method_preserve_values_coordinates_and_mode(
    expression, kind, approximate
):
    trace, op = run(expression)
    assert op.kind == kind and op.lesson.interaction == "activation"
    x, y = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
    assert x.axes == y.axes and x.shape == y.shape
    data = torch.tensor(x.values, dtype=torch.float64)
    reference = (
        F.gelu(data, approximate=approximate) if kind == "gelu" else getattr(torch, kind)(data)
    )
    torch.testing.assert_close(torch.tensor(y.values, dtype=torch.float64), reference)
    if kind == "gelu":
        assert op.arguments.get("approximate", "none") == approximate


def test_noncontiguous_scalar_and_integer_relu():
    for expression in ("torch.sigmoid(x.transpose(0, 2))", "torch.tanh(x[0, 0, 0])"):
        trace, op = run(expression)
        assert op.lesson.interaction == "activation"
        x, y = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
        assert x.shape == y.shape and x.numel == y.numel
    trace, op = run("x.relu()", dtype="int64")
    assert op.lesson.interaction == "activation"
    assert trace.tensors[op.outputs[0]].values[:7] == [0, 0, 0, 0, 0, 0, 1]


def test_inplace_and_out_variants_keep_mutation_inspection():
    for expression in ("F.relu(x, inplace=True)", "x.relu_()", "torch.sigmoid(x, out=x)"):
        trace, op = run(expression)
        assert op.mutations
        assert op.lesson.interaction == "inspect" and op.lesson.category == "memory"


@pytest.mark.parametrize("kind", ["relu", "gelu", "sigmoid", "tanh"])
def test_nonfinite_inputs_keep_the_actual_native_results(kind):
    trace, op = run(f"F.{kind}(x / 0)")
    assert op.lesson.interaction == "activation"
    x, y = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
    assert [x.values[i] for i in (0, 5, 6)] == ["-inf", "nan", "inf"]
    expected = getattr(F, kind)(
        torch.tensor([-float("inf"), float("nan"), float("inf")], dtype=torch.float64)
    )
    assert [y.values[i] for i in (0, 5, 6)] == [json_value(value) for value in expected]


def test_large_paged_and_metadata_runs_keep_bounded_lessons(tmp_path):
    trace, op = run("F.gelu(x, approximate='tanh')", shape=(128, 16, 768), snapshot_dir=tmp_path)
    assert op.lesson.interaction == "activation"
    output = trace.tensors[op.outputs[0]]
    assert output.value_source == "paged" and output.values == []
    assert read_snapshot(tmp_path, output.id, [0, output.numel - 1]) == pytest.approx(
        [
            F.gelu(torch.tensor(-5.0, dtype=torch.float64), approximate="tanh").item(),
            output.numel - 6,
        ]
    )
    trace, op = run("torch.tanh(x)", shape=(1024, 1024, 1024), mode="shapes")
    assert op.lesson.interaction == "activation"
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 10000


def test_invalid_geometry_and_unknown_approximation_use_general_inspection():
    trace, op = run("F.gelu(x)")
    inputs = [trace.tensors[i] for i in op.inputs]
    outputs = [trace.tensors[i] for i in op.outputs]
    for args in (
        {"approximate": "unknown"},
        {"approximate": None},
        {"inplace": True},
        {"out": "tensor"},
    ):
        assert describe_operation("gelu", args, inputs, outputs).interaction == "inspect"
    outputs[0].shape = [3, 2, 4]
    assert describe_operation("gelu", {}, inputs, outputs).interaction == "inspect"
    outputs[0].shape = inputs[0].shape
    outputs[0].dtype = "float32"
    assert describe_operation("gelu", {}, inputs, outputs).interaction == "inspect"
    inputs[0].numel = 0
    assert describe_operation("gelu", {}, inputs, outputs).interaction == "inspect"


def test_shared_frontend_activation_fixtures_match_pytorch():
    path = Path(__file__).parents[2] / "frontend/src/operations/fixtures/activations.json"
    for case in json.loads(path.read_text()):
        x = torch.tensor(case["input"], dtype=torch.float64)
        expected = (
            F.gelu(x, approximate=case["approximate"])
            if case["kind"] == "gelu"
            else getattr(torch, case["kind"])(x)
        )
        # The fixtures were written on one machine; another CPU's vector code
        # rounds the last bits of a float64 differently, so match to rounding.
        torch.testing.assert_close(
            torch.tensor(case["output"], dtype=torch.float64), expected, atol=1e-12, rtol=1e-12
        )
