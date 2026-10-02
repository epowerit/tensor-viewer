from tensorviewer.declarations import read_project
from tensorviewer.worker import execute

MODEL = """
import torch
import torch.nn.functional as F
from torch import nn

# input x: batch=2, features=4


class Model(nn.Module):
    def forward(self, x):
        positions = torch.arange(1, 9, 2)
        mask = torch.ones(4, 4).tril().bool()
        z = torch.zeros_like(x)
        unit = F.normalize(x, dim=-1)
        return unit + z, positions.float(), ~mask
"""


def test_creation_casts_and_normalize_have_lessons():
    trace = execute(read_project(MODEL).draft)
    lessons = {op.kind: op.lesson for op in trace.operations}
    assert lessons["arange"].summary == "Count from 1 up to, but not including, 9, in steps of 2."
    assert lessons["ones"].category == "creation"
    assert lessons["zeros_like"].summary == (
        "Make a float32 tensor shaped like x filled with zeros."
    )
    assert lessons["bool"].summary.startswith("Convert each value from float32 to bool")
    assert lessons["bool"].interaction == "relation"
    assert lessons["__invert__"].title == "Invert each element"
    assert lessons["float"].summary.startswith("Convert each value from int64 to float32")
    assert lessons["normalize"].summary.startswith(
        "Divide each vector along axis -1 (features) by its length"
    )
    assert not any(lesson.category == "generic" for lesson in lessons.values())
    inverted = next(op for op in trace.operations if op.kind == "__invert__")
    assert trace.tensors[inverted.outputs[0]].name == "~mask"
