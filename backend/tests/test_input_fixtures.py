import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.worker import execute

SOURCE = """from torch import nn
class Example(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.unused = nn.Linear(width, width)
    def forward(self, x):
        return x + 1
"""


def fixture(**changes):
    return {
        "name": " Token sequence ",
        "input": {
            "shape": [2, 3, 4],
            "axis_names": ["batch", "tokens", "features"],
            "generator": "random",
            "dtype": "float64",
            "seed": 12,
            "random_stream": "input",
        },
        "capture_mode": "values",
        **changes,
    }


def test_library_persists_and_copies_settings_into_immutable_runs(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/v1/input-fixtures").json() == []
    response = client.post("/api/v1/input-fixtures", json=fixture())
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["name"] == "Token sequence"
    draft = {
        "name": "Fixture study",
        "code": SOURCE,
        "class_name": "Example",
        "constructor": {},
        "input": saved["input"],
        "capture_mode": saved["capture_mode"],
    }
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    assert run["trace"]["error"] is None
    revised = {**draft, "input": {**saved["input"], "seed": 42}}
    assert client.put(f"/api/v1/projects/{project['id']}", json=revised).status_code == 200
    second = client.post("/api/v1/input-fixtures", json=fixture(input=revised["input"])).json()
    assert saved["id"] != second["id"]
    restarted = TestClient(create_app(tmp_path))
    assert restarted.get("/api/v1/input-fixtures").json() == [second, saved]
    assert restarted.get(f"/api/v1/runs/{run['id']}").json()["project"]["input"] == saved["input"]


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "  "},
        {"name": "x" * 81},
        {"capture_mode": "unknown"},
        {"input": {"shape": [0], "axis_names": []}},
        {"input": {"shape": [2, 4], "axis_names": ["only_one"]}},
        {"input": {"seed": -1}},
        {"input": {"generator": "random", "dtype": "int64"}},
        {"input": {"random_stream": "unknown"}},
        {"input": {"shape": [1024, 1024, 1024]}},
        {"input": {"shape": [2**41], "axis_names": []}, "capture_mode": "shapes"},
    ],
)
def test_invalid_fixtures_are_not_saved(tmp_path, changes):
    client = TestClient(create_app(tmp_path))
    assert client.post("/api/v1/input-fixtures", json=fixture(**changes)).status_code == 422
    assert client.get("/api/v1/input-fixtures").json() == []


def test_huge_shape_fixture_stores_metadata_without_running_code(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Saving input settings must not execute a model or allocate a tensor")

    monkeypatch.setattr(torch, "randn", unexpected)
    monkeypatch.setattr("tensorviewer.app.run_project", unexpected)
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/v1/input-fixtures",
        json=fixture(
            input={"shape": [1024, 1024, 1024], "generator": "random", "random_stream": "input"},
            capture_mode="shapes",
        ),
    )
    assert response.status_code == 201
    assert len(response.content) < 1000


def input_values(trace):
    assert trace.error is None, trace.error
    return trace.tensors[trace.input_ids[0]].values


def test_independent_seed_reproduces_input_across_models_and_preserves_legacy():
    spec = InputSpec(shape=[2, 4], axis_names=[], generator="random", random_stream="input")
    draft = ProjectDraft(name="RNG", code=SOURCE, class_name="Example", constructor={}, input=spec)
    expected = (
        torch.randn([2, 4], generator=torch.Generator().manual_seed(spec.seed)).flatten().tolist()
    )
    assert input_values(execute(draft)) == expected
    changed = draft.model_copy(update={"constructor": {"width": 37}})
    assert input_values(execute(changed)) == expected
    assert input_values(execute(draft)) == expected
    next_seed = spec.model_copy(update={"seed": 8})
    assert input_values(execute(draft.model_copy(update={"input": next_seed}))) != expected
    legacy = InputSpec(shape=[2, 4], axis_names=[], generator="random")
    torch.manual_seed(legacy.seed)
    torch.nn.Linear(4, 4)
    legacy_expected = torch.randn([2, 4]).flatten().tolist()
    assert input_values(execute(draft.model_copy(update={"input": legacy}))) == legacy_expected
    shapes = execute(draft.model_copy(update={"capture_mode": "shapes"}))
    assert input_values(shapes) == []
    assert shapes.tensors[shapes.input_ids[0]].shape == [2, 4]


def test_independent_input_does_not_consume_model_rng():
    code = SOURCE.replace("return x + 1", "return x + __import__('torch').randn_like(x)")
    common = {"name": "Forward RNG", "code": code, "class_name": "Example", "constructor": {}}
    random = execute(
        ProjectDraft(**common, input=InputSpec(generator="random", random_stream="input"))
    )
    zeros = execute(ProjectDraft(**common, input=InputSpec(generator="zeros")))
    inputs = input_values(random)
    noise = zeros.tensors[zeros.output_ids[0]].values
    expected = (torch.tensor(inputs) + torch.tensor(noise)).tolist()
    assert random.tensors[random.output_ids[0]].values == expected
