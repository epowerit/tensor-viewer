import pytest

from tensorviewer.declarations import read_project

MODEL = '''"""Pair attention.

Two inputs and a mask.
"""
from torch import nn

# input query: batch=2, tokens=3, features=4
# input image: batch=1, channels=3, height=8, width=8 | image
# input words: text "the cat sat"
# input mask (keyword): batch=2, tokens=3 | ones | float64
# constructor: {"heads": 2}


class Helper(nn.Module):
    def forward(self, x):
        return x


class Pair(nn.Module):
    def forward(self, query, image, words, *, mask, scale=1.0):
        return query
'''


def test_everything_the_run_needs_comes_from_the_code():
    read = read_project(MODEL)
    draft = read.draft
    assert read.notes == []
    assert (read.title, read.summary) == ("Pair attention", "Two inputs and a mask.")
    assert draft.name == "Pair attention"
    # The last nn.Module runs; its forward signature names the inputs.
    assert draft.class_name == "Pair"
    assert draft.constructor == {"heads": 2}
    inputs = {item.name: item for item in draft.forward_inputs}
    assert list(inputs) == ["query", "image", "words", "mask"]  # scale has a default
    assert inputs["query"].input.shape == [2, 3, 4]
    assert inputs["query"].input.axis_names == ["batch", "tokens", "features"]
    assert inputs["query"].input.generator == "random"
    assert inputs["image"].input.generator == "image"
    assert inputs["words"].input.generator == "text"
    assert inputs["words"].input.shape == [1, 3]
    assert inputs["words"].input.dtype == "int64"
    assert inputs["mask"].binding == "keyword"
    assert (inputs["mask"].input.generator, inputs["mask"].input.dtype) == ("ones", "float64")


def test_undeclared_inputs_and_a_chosen_class():
    code = "from torch import nn\n\nclass A(nn.Module):\n    def forward(self, x, y):\n        return x + y\n"
    read = read_project(code, name="Mine")
    assert read.draft.name == "Mine"
    assert [item.input.shape for item in read.draft.forward_inputs] == [[2, 4, 8], [2, 4, 8]]
    two = code + "\n\nclass B(nn.Module):\n    def forward(self, z):\n        return z\n"
    assert read_project(two).draft.class_name == "B"
    assert read_project(two, model="A").draft.class_name == "A"
    assert read_project(two + "\n# model: A\n").draft.class_name == "A"


def test_mistakes_become_notes_not_failures():
    code = (
        "from torch import nn\n# input x: batch=two\n# input y: 3\n# constructor: [1]\n"
        "class A(nn.Module):\n    def forward(self, x):\n        return x\n"
    )
    read = read_project(code)
    assert read.draft.input.shape == [2, 4, 8]  # the default stands in
    assert any("not a size" in note for note in read.notes)
    assert any("does not name a forward parameter" in note for note in read.notes)
    assert any("JSON object" in note for note in read.notes)


def test_code_without_a_module_is_refused():
    with pytest.raises(ValueError, match="nn.Module"):
        read_project("x = 1\n")
    with pytest.raises(ValueError, match="does not parse"):
        read_project("class (:\n")
