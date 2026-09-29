import pytest
import torch

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.templates import PATCH_EMBEDDING_CODE
from tensorviewer.worker import execute


def patch_project(shape=(2, 3, 8, 8), patch=2, mode="values"):
    return ProjectDraft(
        name="Patch study",
        code=PATCH_EMBEDDING_CODE,
        class_name="PatchEmbedding",
        constructor={"channels": shape[1], "features": 6, "patch": patch},
        input=InputSpec(shape=list(shape), axis_names=["batch", "channels", "height", "width"]),
        capture_mode=mode,
    )


def test_patch_projection_and_token_order_match_actual_pytorch():
    trace = execute(patch_project())
    assert trace.error is None
    projection, flatten, transpose = trace.operations
    assert projection.kind == "conv2d"
    assert projection.lesson.interaction == "patch_projection"
    assert projection.lesson.patch_size == [2, 2]
    assert projection.arguments["stride"] == [2, 2]
    assert projection.arguments["padding"] == [0, 0]
    assert flatten.lesson.mapping_rule == "identity"
    assert transpose.lesson.axis_order == [0, 2, 1]
    image, weight, bias = [trace.tensors[i] for i in projection.inputs]
    assert weight.role == bias.role == "parameter"
    x = torch.tensor(image.values).reshape(image.shape)
    w = torch.tensor(weight.values).reshape(weight.shape)
    b = torch.tensor(bias.values)
    tokens = trace.tensors[trace.output_ids[0]]
    expected = torch.nn.functional.conv2d(x, w, b, stride=2).flatten(2).transpose(1, 2)
    torch.testing.assert_close(torch.tensor(tokens.values).reshape(tokens.shape), expected)
    # Row-major patch 11 is at row 2, col 3. All C×2×2 entries feed each feature.
    batch, token, feature = 1, 11, 4
    row, col = divmod(token, 4)
    patch = x[batch, :, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2]
    torch.testing.assert_close(
        (patch * w[feature]).sum() + b[feature], expected[batch, token, feature]
    )
    assert tokens.axes == ["batch", "tokens", "features"]
    assert [trace.tensors[o.outputs[0]].name for o in trace.operations] == [
        "projected",
        "flattened",
        "tokens",
    ]


@pytest.mark.parametrize(
    "settings",
    [
        "stride=1",
        "stride=2, padding=1",
        "stride=2, dilation=2",
        "stride=2, groups=3",
    ],
)
def test_other_convolutions_do_not_claim_non_overlapping_patch_membership(settings):
    channels = 1 if "groups" in settings else 3
    draft = patch_project()
    draft.code = f"""import torch
from torch import nn
class PatchEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(6, {channels}, 2, 2))
    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, {settings})
"""
    trace = execute(draft)
    assert trace.error is None
    assert trace.operations[0].lesson.interaction == "inspect"
    assert trace.operations[0].lesson.patch_size is None


def test_rectangular_patches_and_keyword_operand_order():
    draft = patch_project(shape=(1, 3, 6, 8))
    draft.code = """import torch
from torch import nn
class PatchEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(5, 3, 2, 4))
        self.bias = nn.Parameter(torch.zeros(5))
    def forward(self, x):
        return torch.nn.functional.conv2d(bias=self.bias, stride=(2,4), weight=self.weight, input=x)
"""
    trace = execute(draft)
    assert trace.error is None
    op = trace.operations[0]
    assert [trace.tensors[i].shape for i in op.inputs] == [[1, 3, 6, 8], [5, 3, 2, 4], [5]]
    assert op.lesson.patch_size == [2, 4]


def test_incomplete_tiles_are_not_described_as_a_complete_image_partition():
    trace = execute(patch_project(shape=(1, 3, 7, 8)))
    assert trace.error is None
    assert trace.operations[0].lesson.patch_size is None


def test_large_shape_only_patch_trace_stays_bounded():
    trace = execute(patch_project(shape=(1024, 3, 1024, 1024), patch=16, mode="shapes"))
    assert trace.error is None
    assert trace.operations[0].lesson.patch_size == [16, 16]
    assert trace.tensors[trace.output_ids[0]].shape == [1024, 4096, 6]
    assert all(t.value_source == "shape" and not t.values for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 14000
