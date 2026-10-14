"""Who sees whom, checked against brute-force autograd on small modules."""

import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.declarations import read_project
from tensorviewer.library import entries, project_name
from tensorviewer.models import DependencyRequest
from tensorviewer.worker import execute

HEADER = """import math

import torch
from torch import nn

# input x: batch=3, tokens=5, features=4
"""

MODULES = {
    # Each token reads itself and those before it; examples stay apart.
    "running": """
class Model(nn.Module):
    def forward(self, x):
        return x.cumsum(1)  # axes: batch, tokens, features
""",
    # A softmax slice sums to 1 whatever its input: a plain sum of the slice
    # would have no gradient at all.
    "softmax": """
class Model(nn.Module):
    def forward(self, x):
        return x.softmax(dim=-1)  # axes: batch, tokens, features
""",
    "causal": """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(4)  # axes: batch, queries, keys
        allowed = torch.ones(5, 5).tril().bool()  # axes: queries, keys
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.softmax(dim=-1)  # axes: batch, queries, keys
        return weights @ v  # axes: batch, tokens, features
""",
    # Seq-first code that reinterprets memory instead of swapping axes.
    "seq-first": """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        batch, tokens, dim = x.shape
        h = self.proj(x)  # axes: batch, tokens, features
        t = h.view(tokens, batch, dim)
        t = t.cumsum(0)
        return t.transpose(0, 1)  # axes: batch, tokens, features
""",
}
MODULES["unmasked"] = MODULES["causal"].replace(
    '        scores = scores.masked_fill(~allowed, float("-inf"))\n', ""
)
# A mask off by one: each token also reads the next.
MODULES["off-by-one"] = MODULES["causal"].replace(".tril()", ".tril(diagonal=1)")
MODULES["transposed"] = MODULES["seq-first"].replace(
    "h.view(tokens, batch, dim)", "h.transpose(0, 1)"
)


def draft_of(name):
    return read_project(HEADER + MODULES[name], name).draft


def brute_force(name, out_axis, in_axis):
    """Which input slices each output slice reads, from the full Jacobian."""
    namespace = {}
    exec(HEADER + MODULES[name], namespace)
    torch.manual_seed(0)
    model = namespace["Model"]().eval()
    x = torch.randn(3, 5, 4)
    jacobian = torch.autograd.functional.jacobian(model, x)
    rank = jacobian.ndim - x.ndim
    return [
        [
            bool(jacobian.select(out_axis, i).select(rank - 1 + in_axis, j).abs().amax() > 0)
            for j in range(x.shape[in_axis])
        ]
        for i in range(jacobian.shape[out_axis])
    ]


def answer(draft, axis, expect=None, **asked):
    trace = execute(draft)
    assert trace.error is None, trace.error
    found = execute(
        draft,
        target=DependencyRequest(tensor_id=trace.output_ids[0], axis=axis, expect=expect, **asked),
    )
    assert found.error is None, found.error
    return trace, found


@pytest.mark.parametrize("name", sorted(MODULES))
@pytest.mark.parametrize("axis", [0, 1], ids=["batch", "tokens"])
def test_each_slice_reads_what_the_full_jacobian_says(name, axis):
    _, found = answer(draft_of(name), axis)
    expected = brute_force(name, axis, axis)
    assert found.input_axis == axis
    assert found.reads == expected
    assert found.zero == [[not read for read in row] for row in expected]


def test_the_pattern_is_held_to_and_the_first_breach_is_bisected():
    trace, found = answer(draft_of("running"), 1, "causal")
    assert found.violations == [] and found.breach is None
    _, found = answer(draft_of("running"), 0, "diagonal")
    assert found.violations == [] and found.breach is None

    trace, found = answer(draft_of("unmasked"), 1, "causal")
    assert found.violations[:3] == [(0, 1), (0, 2), (0, 3)]
    step = next(op for op in trace.operations if op.id == found.breach.operation_id)
    # The scores are not a sequence of tokens; the first result that is,
    # and reads a later token, is the weighted sum of the values.
    assert step.kind == "matmul" and "weights @ v" in step.source.text
    assert (found.breach.output, found.breach.input) == (0, 1)

    trace, found = answer(draft_of("off-by-one"), 1, "causal")
    assert found.violations == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert "weights @ v" in trace.operations[found.breach.step].source.text

    trace, found = answer(draft_of("seq-first"), 0, "diagonal")
    assert found.violations[0] == (0, 1)
    step = trace.operations[found.breach.step]
    assert step.kind == "view" and found.breach.after == trace.operations[0].id
    _, found = answer(draft_of("transposed"), 0, "diagonal")
    assert found.violations == [] and found.breach is None


def test_only_the_first_slices_up_to_the_limit_are_asked():
    _, found = answer(draft_of("running"), 1, "causal", limit=2)
    assert found.size == 5 and len(found.reads) == 2 and len(found.reads[0]) == 5


def test_strength_is_relative_to_each_rows_largest():
    _, found = answer(draft_of("running"), 1)
    for row, reads in zip(found.strength, found.reads):
        assert max(row) == 1
        assert all((share > found.threshold) == read for share, read in zip(row, reads))


def test_gpt_reads_a_clean_lower_triangle_and_its_unmasked_copy_is_caught():
    entry = next(each for each in entries() if each.number == 13)
    draft = read_project(entry.code, project_name(entry)).draft
    trace = execute(draft)
    probabilities = [op for op in trace.operations if op.kind == "softmax"][-1].outputs[0]
    found = execute(
        draft, target=DependencyRequest(tensor_id=probabilities, axis=1, expect="causal")
    )
    # Token ids have no gradient; their embeddings do.
    assert found.kind == "embedding" and found.input_axis == 1
    assert found.reads == [[j <= i for j in range(13)] for i in range(13)]
    assert found.violations == [] and found.breach is None

    broken = draft.model_copy(
        update={
            "code": entry.code.replace(
                '        scores = scores.masked_fill(~allowed, float("-inf"))\n', ""
            )
        }
    )
    trace = execute(broken)
    probabilities = [op for op in trace.operations if op.kind == "softmax"][-1].outputs[0]
    found = execute(
        broken, target=DependencyRequest(tensor_id=probabilities, axis=1, expect="causal")
    )
    assert all(all(row) for row in found.reads)
    step = trace.operations[found.breach.step]
    # The first block's attention, where the later tokens first come in.
    assert step.module.endswith("blocks.0.attention")
    assert "weights @ v" in step.source.text


def test_an_integer_result_or_a_missing_axis_is_refused(tmp_path):
    entry = next(each for each in entries() if each.number == 13)
    client = TestClient(create_app(tmp_path))
    project = client.post(
        "/api/v1/projects", json=read_project(entry.code, project_name(entry)).draft.model_dump()
    ).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    trace = run["trace"]
    asked = f"/api/v1/runs/{run['id']}/dependencies"
    result = client.post(asked, json={"tensor_id": trace["output_ids"][0], "axis": 0})
    assert result.status_code == 422 and "floating-point" in result.json()["detail"]
    assert client.post(asked, json={"tensor_id": "t1", "axis": 5}).status_code == 404
    probabilities = [op for op in trace["operations"] if op["kind"] == "softmax"][-1]
    found = client.post(
        asked, json={"tensor_id": probabilities["outputs"][0], "axis": 1, "expect": "causal"}
    )
    assert found.status_code == 200, found.text
    body = found.json()
    assert body["kind"] == "embedding" and body["violations"] == [] and body["size"] == 13
