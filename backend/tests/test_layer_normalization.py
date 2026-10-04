import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.operations.normalization import layer_normalization_spec
from tensorviewer.operations.registry import describe_operation
from tensorviewer.worker import execute

CASES = [
    ([2, 3, 4], [4], True, True, 1e-5),
    ([2, 2, 3], [2, 3], True, True, 0.25),
    ([2, 3, 4], [4], True, False, 1e-5),
    ([2, 3, 4], [4], False, True, 1e-5),
    ([2, 3, 4], [3, 4], False, False, 1e-5),
    ([2, 3], [2, 3], True, True, 0),
    ([2, 1], [1], True, True, 1e-5),
    ([4], [4], False, False, 0),
]


def project(shape, normalized, weight=True, bias=True, eps=1e-5, mode="values", expression=None):
    expression = (
        expression
        or f"torch.nn.functional.layer_norm(x, {normalized!r}, self.scale, self.shift, {eps})"
    )
    count = 1
    for n in normalized:
        count *= n
    return ProjectDraft(
        name="Layer normalization study",
        class_name="Example",
        constructor={},
        input=InputSpec(
            shape=shape, axis_names=[f"d{i}" for i in range(len(shape))], dtype="float64"
        ),
        capture_mode=mode,
        code=f"""import torch
from torch import nn
class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter((torch.arange({count}, dtype=torch.float64) / 2 + 0.5).reshape({normalized!r})) if {weight} else None
        self.shift = nn.Parameter((torch.arange({count}, dtype=torch.float64) / 7 - 0.2).reshape({normalized!r})) if {bias} else None
    def forward(self, x):
        return {expression}
""",
    )


@pytest.mark.parametrize("case", CASES)
def test_capture_and_every_group_match_pytorch_and_population_formula(case):
    shape, norm, has_weight, has_bias, eps = case
    trace = execute(project(*case))
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "layer_normalization"
    assert op.arguments["weight"] == ("tensor" if has_weight else None)
    assert op.arguments["bias"] == ("tensor" if has_bias else None)
    tensors = [trace.tensors[i] for i in op.inputs]
    x = torch.tensor(tensors[0].values, dtype=torch.float64).reshape(shape)
    weight = (
        torch.tensor(tensors[1].values, dtype=torch.float64).reshape(norm) if has_weight else None
    )
    bias = torch.tensor(tensors[-1].values, dtype=torch.float64).reshape(norm) if has_bias else None
    y = trace.tensors[op.outputs[0]]
    actual = torch.tensor(y.values, dtype=torch.float64).reshape(y.shape)
    torch.testing.assert_close(actual, F.layer_norm(x, norm, weight, bias, eps))
    axes = tuple(range(x.ndim - len(norm), x.ndim))
    mean = x.mean(axes, keepdim=True)
    variance = ((x - mean) ** 2).mean(axes, keepdim=True)
    manual = (x - mean) / torch.sqrt(variance + eps)
    manual = manual * (weight if weight is not None else 1) + (bias if bias is not None else 0)
    torch.testing.assert_close(actual, manual)
    assert y.axes == tensors[0].axes


def test_keyword_order_and_bias_only_are_not_confused_with_scale():
    trace = execute(
        project(
            [2, 4],
            [4],
            False,
            True,
            expression="torch.nn.functional.layer_norm(bias=self.shift, eps=0.25, input=x, normalized_shape=[4])",
        )
    )
    assert trace.error is None
    op = trace.operations[-1]
    assert op.arguments["weight"] is None and op.arguments["bias"] == "tensor"
    assert [trace.tensors[i].name for i in op.inputs] == ["x", "shift"]
    assert op.lesson.interaction == "layer_normalization"


def test_noncontiguous_input_and_module_evaluation_use_current_groups():
    draft = project(
        [2, 3, 4],
        [4],
        expression="torch.nn.functional.layer_norm(x.transpose(0, 1), [4], self.scale, self.shift)",
    )
    trace = execute(draft)
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "layer_normalization"
    x = trace.tensors[op.inputs[0]]
    assert not x.contiguous
    assert x.shape == [3, 2, 4]
    draft.code = """import torch
from torch import nn
class Example(nn.Module):
 def __init__(self):
  super().__init__()
  self.norm = nn.LayerNorm(4)
 def forward(self, x):
  return self.norm(x)
"""
    trace = execute(draft)
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "layer_normalization"
    assert op.arguments["weight"] == op.arguments["bias"] == "tensor"


def test_constant_zero_epsilon_and_nonfinite_outputs_remain_recorded():
    draft = project([2, 1], [1], False, False, 0)
    trace = execute(draft)
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "layer_normalization"
    assert trace.tensors[op.outputs[0]].values == ["nan", "nan"]
    draft = project(
        [2, 4], [4], False, False, expression="torch.nn.functional.layer_norm(x / x, [4])"
    )
    trace = execute(draft)
    assert trace.error is None
    assert trace.tensors[trace.operations[-1].outputs[0]].values[:4] == ["nan"] * 4


def test_shape_only_billion_element_trace_is_bounded():
    trace = execute(project([1024, 1024, 1024], [1024], mode="shapes"))
    assert trace.error is None
    assert len(trace.operations) == 1
    op = trace.operations[0]
    assert op.lesson.interaction == "layer_normalization"
    assert all(not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 20000


def test_unverified_geometry_and_ambiguous_legacy_operands_fall_back():
    trace = execute(project([2, 4], [4], True, False))
    op = trace.operations[-1]
    inputs = [trace.tensors[i] for i in op.inputs]
    outputs = [trace.tensors[i] for i in op.outputs]
    for args in [
        {"normalized_shape": [4]},  # cannot infer whether the second operand scales or shifts
        {**op.arguments, "normalized_shape": [2]},
        {**op.arguments, "eps": -1},
        {**op.arguments, "eps": "nan"},
        {**op.arguments, "eps": None},
        {**op.arguments, "weight": "unknown"},
    ]:
        assert describe_operation("layer_norm", args, inputs, outputs).interaction == "inspect"
    spec = layer_normalization_spec({"normalized_shape": [4], "bias": None}, inputs, outputs)
    assert spec["weight"] and not spec["bias"]
    outputs[0].shape = [4, 2]
    assert describe_operation("layer_norm", op.arguments, inputs, outputs).interaction == "inspect"


def test_frontend_golden_values_match_native_pytorch():
    path = Path(__file__).parents[2] / "frontend/src/operations/fixtures/layerNormalization.json"
    for case in json.loads(path.read_text()):
        x = torch.tensor(case["input"]["values"], dtype=torch.float64).reshape(
            case["input"]["shape"]
        )
        args = case["arguments"]

        def operand(name):
            t = case.get(name)
            return torch.tensor(t["values"], dtype=torch.float64).reshape(t["shape"]) if t else None

        y = F.layer_norm(
            x, args["normalized_shape"], operand("weight"), operand("bias"), args["eps"]
        )
        # The fixtures were written on one machine. Another CPU's vector code
        # rounds the last bits of a float64 differently, so match to rounding.
        torch.testing.assert_close(
            y.flatten(),
            torch.tensor(case["output"]["values"], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
