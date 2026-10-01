import pytest
import torch
from pydantic import ValidationError

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.samples import sample_image, token_ids, tokenize
from tensorviewer.worker import execute


def test_sample_image_is_deterministic_bounded_and_varies_by_channel_and_item():
    image = sample_image([2, 3, 16, 16], torch.float32)
    assert image.shape == (2, 3, 16, 16) and image.dtype == torch.float32
    assert float(image.min()) >= 0 and float(image.max()) <= 1
    assert torch.equal(image, sample_image([2, 3, 16, 16], torch.float32))
    # The disc is bright in the first channel and moves between batch items.
    assert float(image[0, 0].max()) == 1.0 and float(image[0, 0].min()) < 0.2
    assert not torch.equal(image[0], image[1])
    assert not torch.equal(image[0, 0], image[0, 1])
    # Rank two is a single grayscale plane; extra leading axes are items.
    assert sample_image([8, 8], torch.float64).shape == (8, 8)
    assert sample_image([2, 2, 1, 4, 4], torch.float32).shape == (2, 2, 1, 4, 4)
    assert sample_image([1, 5, 4, 4], torch.float32)[0, 3].equal(
        sample_image([1, 5, 4, 4], torch.float32)[0, 0]
    )


def test_sentences_become_stable_token_ids():
    tokens, vocabulary = tokenize("The cat sat on the mat.")
    assert tokens == ["the", "cat", "sat", "on", "the", "mat", "."]
    assert vocabulary == [".", "cat", "mat", "on", "sat", "the"]
    assert token_ids("The cat sat on the mat.") == [5, 1, 4, 3, 5, 2, 0]
    assert tokenize("don't  stop!")[0] == ["don't", "stop", "!"]


def draft(script, **spec):
    return ProjectDraft(name="samples", code="pass", script=script, input=InputSpec(**spec))


def test_image_and_text_inputs_run_and_keep_their_meaning():
    trace = execute(
        draft(
            "edges = F.conv2d(x, torch.ones(1, 3, 3, 3))\nsmall = F.max_pool2d(edges, 2)",
            shape=[1, 3, 8, 8],
            generator="image",
            axis_names=["batch", "channels", "height", "width"],
        )
    )
    assert trace.error is None
    source = trace.tensors[trace.input_ids[0]]
    assert source.values == sample_image([1, 3, 8, 8], torch.float32).reshape(-1).tolist()
    axes = ["batch", "channels", "height", "width"]
    assert [trace.tensors[op.outputs[0]].axes for op in trace.operations[-2:]] == [axes, axes]

    trace = execute(
        draft(
            "table = torch.arange(24.0).reshape(6, 4)\nvectors = F.embedding(x, table)",
            shape=[1, 7],
            dtype="int64",
            generator="text",
            text="The cat sat on the mat.",
            axis_names=["batch", "tokens"],
        )
    )
    assert trace.error is None
    assert trace.tensors[trace.input_ids[0]].values == [5, 1, 4, 3, 5, 2, 0]
    lookup = trace.operations[-1]
    assert lookup.lesson.relation == {"rule": "table", "operand": 1}
    assert lookup.lesson.mapping[:4] == [20, 21, 22, 23]
    shapes = execute(
        draft(
            "x + 1", shape=[1, 2], dtype="int64", generator="text", text="hi there", axis_names=[]
        ).model_copy(update={"capture_mode": "shapes"})
    )
    assert shapes.error is None and shapes.tensors[shapes.input_ids[0]].values == []


def test_sample_inputs_reject_inconsistent_settings():
    for spec in [
        dict(shape=[8], generator="image", axis_names=[]),
        dict(shape=[1, 8, 8], generator="image", dtype="int64"),
        dict(shape=[1, 3], generator="text", dtype="int64", axis_names=[]),
        dict(shape=[1, 3], generator="arange", text="a b c", axis_names=[]),
        dict(shape=[1, 2], generator="text", text="a b c", dtype="int64", axis_names=[]),
        dict(shape=[1, 3], generator="text", text="a b c", dtype="float32", axis_names=[]),
        dict(shape=[1, 1], generator="text", text="   ", dtype="int64", axis_names=[]),
    ]:
        with pytest.raises(ValidationError):
            InputSpec(**spec)
