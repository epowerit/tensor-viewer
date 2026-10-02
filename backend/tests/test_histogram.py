import torch

from tensorviewer.declarations import read_project
from tensorviewer.tracing import HISTOGRAM_BINS, value_histogram
from tensorviewer.worker import execute

MODEL = """
import torch
from torch import nn

# input x: batch=4, features=8


class Model(nn.Module):
    def forward(self, x):
        dead = torch.relu(x - 10)
        bad = x / (x - x)
        return dead, bad
"""


def test_histograms_count_bins_zeros_and_non_finite():
    histogram = value_histogram(torch.tensor([0.0, 0.0, 1.0, 2.0, float("nan"), float("inf")]))
    assert (histogram.low, histogram.high) == (0.0, 2.0)
    assert len(histogram.counts) == HISTOGRAM_BINS
    assert sum(histogram.counts) == 4
    assert (histogram.zeros, histogram.non_finite) == (2, 2)
    assert histogram.mean == 0.75
    # Equal values make one bin; nothing finite makes none.
    assert value_histogram(torch.full((3,), 5.0)).counts == [3]
    assert value_histogram(torch.tensor([float("nan")])).counts == []
    assert value_histogram(torch.tensor([True, False])) is None


def test_recorded_tensors_carry_their_distribution(tmp_path):
    trace = execute(read_project(MODEL).draft)
    by_name = {state.name: state for state in trace.tensors.values()}
    dead = by_name["dead"].histogram
    assert dead.zeros == 32 and dead.counts == [32]  # every ReLU output is zero
    bad = by_name["bad"].histogram
    assert bad.non_finite == 32  # 0/0 and x/0
    shapes = execute(read_project(MODEL).draft.model_copy(update={"capture_mode": "shapes"}))
    assert all(state.histogram is None for state in shapes.tensors.values())


def test_paged_tensors_get_their_range_from_the_histogram(tmp_path):
    code = MODEL.replace("batch=4, features=8", "batch=80, features=80")
    trace = execute(read_project(code).draft, tmp_path)
    x = trace.tensors[trace.input_ids[0]]
    assert x.value_source == "paged" and x.values == []
    assert x.histogram is not None and sum(x.histogram.counts) == 6400
    assert (x.minimum, x.maximum) == (x.histogram.low, x.histogram.high)
