import json
from itertools import product
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.operations.convolution import convolution_spec
from tensorviewer.operations.registry import describe_operation
from tensorviewer.worker import execute

CASES = [
    ((2, 4, 9), (3,), 6, 2, 1, 2, 2, True),
    ((2, 2, 7), (2,), 4, 1, "same", 1, 2, True),
    ((2, 7), (3,), 3, 2, "valid", 1, 1, False),
    ((1, 4, 5, 7), (2, 3), 6, (1, 2), (1, 2), (2, 1), 2, True),
    ((2, 2, 5, 5), (2, 2), 4, 1, "same", (1, 3), 2, False),
    ((2, 5, 7), (3, 2), 3, (2, 1), (2, 0), 1, 1, True),
]


def project(
    shape,
    kernel,
    features,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    mode="values",
    forward=None,
):
    dimensions = len(kernel)
    channels = shape[-dimensions - 1]
    return ProjectDraft(
        name="Convolution study",
        class_name="Example",
        constructor={},
        input=InputSpec(shape=list(shape), dtype="float64", axis_names=[]),
        capture_mode=mode,
        code=f"""import torch
from torch import nn
class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv{dimensions}d({channels}, {features}, {kernel!r}, stride={stride!r}, padding={padding!r}, dilation={dilation!r}, groups={groups}, bias={bias})
    def forward(self, x):
        {forward or "return self.conv(x)"}
""",
    )


def vector(v, dimensions):
    return (v,) * dimensions if isinstance(v, int) else tuple(v)


@pytest.mark.parametrize("case", CASES)
def test_convolutions_and_every_output_contribution_match_pytorch(case):
    shape, kernel, features, stride, padding, dilation, groups, has_bias = case
    trace = execute(project(*case))
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "convolution"
    assert op.lesson.patch_size is None
    inputs = [trace.tensors[i] for i in op.inputs]
    x, w = [torch.tensor(t.values, dtype=torch.float64).reshape(t.shape) for t in inputs[:2]]
    bias = torch.tensor(inputs[2].values, dtype=torch.float64) if has_bias else None
    recorded = trace.tensors[op.outputs[0]]
    actual = torch.tensor(recorded.values, dtype=torch.float64).reshape(recorded.shape)
    dimensions = len(kernel)
    reference = getattr(F, f"conv{dimensions}d")(x, w, bias, stride, padding, dilation, groups)
    torch.testing.assert_close(actual, reference)
    # Independently sum every receptive field, across groups and spatial dimensions.
    stride, dilation = vector(stride, dimensions), vector(dilation, dimensions)
    pad = (
        tuple((k - 1) * d // 2 for k, d in zip(kernel, dilation))
        if padding == "same"
        else (0,) * dimensions
        if padding == "valid"
        else vector(padding, dimensions)
    )
    unbatched = len(shape) == dimensions + 1
    x_batch, y_batch = (x.unsqueeze(0), actual.unsqueeze(0)) if unbatched else (x, actual)
    channels_per_group = x_batch.shape[1] // groups
    for batch in range(y_batch.shape[0]):
        for feature in range(features):
            group = feature // (features // groups)
            for position in product(*(range(n) for n in y_batch.shape[2:])):
                total = 0.0 if bias is None else bias[feature].item()
                for channel in range(channels_per_group):
                    for offset in product(*(range(k) for k in kernel)):
                        pixel = tuple(
                            o * s - p + k * d
                            for o, s, p, k, d in zip(position, stride, pad, offset, dilation)
                        )
                        if all(0 <= v < size for v, size in zip(pixel, x_batch.shape[2:])):
                            total += (
                                x_batch[(batch, group * channels_per_group + channel, *pixel)]
                                * w[(feature, channel, *offset)]
                            ).item()
                assert total == pytest.approx(
                    y_batch[(batch, feature, *position)].item(), rel=1e-10, abs=1e-10
                )
    geometry = convolution_spec(op.kind, op.arguments, inputs, [recorded])
    assert geometry["before"] == list(pad)
    assert geometry["groups"] == groups


def test_conv1d_reordered_keywords_capture_semantic_operands_and_parameters():
    trace = execute(
        project(
            (2, 4, 7),
            (3,),
            6,
            padding=2,
            dilation=2,
            groups=2,
            forward="return torch.conv1d(groups=2, dilation=2, padding=2, bias=self.conv.bias, weight=self.conv.weight, input=x)",
        )
    )
    assert trace.error is None
    op = trace.operations[-1]
    assert [trace.tensors[i].shape for i in op.inputs] == [[2, 4, 7], [6, 2, 3], [6]]
    assert op.arguments["dilation"] == op.arguments["groups"] == 2
    assert op.lesson.interaction == "convolution"


def test_noncontiguous_and_explicit_reflect_padding_use_recorded_intermediate():
    draft = project(
        (1, 2, 5, 5),
        (3, 3),
        3,
        padding=0,
        forward="return self.conv(torch.nn.functional.pad(x.transpose(-1, -2), (1, 1, 1, 1), mode='reflect'))",
    )
    trace = execute(draft)
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "convolution"
    assert trace.tensors[op.inputs[0]].shape == [1, 2, 7, 7]
    assert op.inputs[0] != trace.input_ids[0]
    assert [o.kind for o in trace.operations] == ["transpose", "pad", "conv2d"]


def test_large_shape_convolution_has_no_receptive_field_arrays():
    trace = execute(
        project((1024, 4, 1024, 1024), (3, 3), 8, padding=2, dilation=2, groups=2, mode="shapes")
    )
    assert trace.error is None
    op = trace.operations[-1]
    assert op.lesson.interaction == "convolution"
    assert trace.tensors[op.outputs[0]].shape == [1024, 8, 1024, 1024]
    assert len(trace.model_dump_json()) < 12000
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())


def test_invalid_recorded_geometry_falls_back_without_claiming_a_mapping():
    trace = execute(project((1, 2, 5), (3,), 4))
    op = trace.operations[-1]
    inputs = [trace.tensors[i] for i in op.inputs]
    outputs = [trace.tensors[i] for i in op.outputs]
    for args in [
        {"groups": 3},
        {"stride": 0},
        {"padding": "same", "stride": 2},
        {"dilation": True},
    ]:
        assert describe_operation(op.kind, args, inputs, outputs).interaction == "inspect"
    outputs[0] = outputs[0].model_copy(update={"shape": [1, 4, 99]})
    assert describe_operation(op.kind, op.arguments, inputs, outputs).interaction == "inspect"


def test_browser_reference_outputs_remain_independent_pytorch_results():
    path = Path(__file__).parents[2] / "frontend/src/operations/fixtures/convolutions.json"
    for case in json.loads(path.read_text()):

        def value(name):
            data = case[name]
            return (
                None
                if data is None
                else torch.tensor(data["values"], dtype=torch.float64).reshape(data["shape"])
            )

        expected = getattr(F, case["kind"])(
            value("input"), value("weight"), value("bias"), **case["arguments"]
        )
        torch.testing.assert_close(expected, value("output"), rtol=0, atol=0)
