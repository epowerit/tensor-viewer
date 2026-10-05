import subprocess
import sys

from fastapi.testclient import TestClient

from tensorviewer.app import create_app

CODE = """import torch
from torch import nn


class Twice(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.inner = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x):
        for _ in range(2):
            x = self.act(self.inner(x))
        return x * 2, x.sum(-1)
"""


def project(client, **changes):
    template = client.get("/api/v1/templates").json()[1]["project"]
    draft = {
        **template,
        "code": CODE,
        "class_name": "Twice",
        "constructor": {"width": 4},
        "input": {**template["input"], "shape": [2, 3, 4], "generator": "random"},
        **changes,
    }
    return client.post("/api/v1/projects", json=draft).json()


def run_pytest(path):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)],
        capture_output=True,
        text=True,
        cwd=path.parent,
        timeout=120,
    )


def test_a_run_becomes_a_pytest_file_that_passes_on_the_same_model(tmp_path):
    client = TestClient(create_app(tmp_path / "data"))
    made = project(client)
    run = client.post(f"/api/v1/projects/{made['id']}/runs").json()
    assert run["trace"]["error"] is None
    response = client.get(f"/api/v1/runs/{run['id']}/pytest")
    assert response.status_code == 200
    assert f'filename="test_twice_{run["id"][:8]}.py"' in response.headers["content-disposition"]
    source = response.text
    # The model, its seed, every module call, and both results are written down.
    assert "class Twice(nn.Module):" in source
    assert "'inner': [[[2, 3, 4]], [[2, 3, 4]]]" in source
    path = tmp_path / "exported" / "test_twice.py"
    path.parent.mkdir()
    path.write_text(source)
    passed = run_pytest(path)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    # The same test catches a model that computes something else.
    path.write_text(source.replace("return x * 2, x.sum(-1)", "return x * 3, x.sum(-1)"))
    failed = run_pytest(path)
    assert failed.returncode == 1
    assert "expected" in failed.stdout


def test_a_changed_shape_is_named_by_its_module(tmp_path):
    client = TestClient(create_app(tmp_path / "data"))
    made = project(client)
    run = client.post(f"/api/v1/projects/{made['id']}/runs").json()
    source = client.get(f"/api/v1/runs/{run['id']}/pytest").text
    path = tmp_path / "test_narrower.py"
    # The model's own result loses a column; the test names the module.
    path.write_text(source.replace("return x * 2, x.sum(-1)", "return x[..., :3] * 2, x.sum(-1)"))
    failed = run_pytest(path)
    assert failed.returncode == 1
    assert "Twice made [[[2, 3, 3], [2, 3]]]; the run recorded [[[2, 3, 4], [2, 3]]]" in (
        failed.stdout
    )


def test_runs_that_cannot_be_rebuilt_are_refused_with_a_reason(tmp_path):
    client = TestClient(create_app(tmp_path / "data"))
    shapes = project(client, capture_mode="shapes")
    run = client.post(f"/api/v1/projects/{shapes['id']}/runs").json()
    refused = client.get(f"/api/v1/runs/{run['id']}/pytest")
    assert refused.status_code == 422
    assert "shapes-only" in refused.json()["detail"]
    made = project(client)
    run = client.post(f"/api/v1/projects/{made['id']}/runs").json()
    what_if = client.post(
        f"/api/v1/runs/{run['id']}/what-if", json={"edits": [{"index": 0, "value": 1}]}
    ).json()
    assert client.get(f"/api/v1/runs/{what_if['id']}/pytest").status_code == 422
    assert client.get("/api/v1/runs/missing/pytest").status_code == 404


ENTRY = """import torch
from torch import nn

from .layers import Scale


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.scale = Scale(3.0)

    def forward(self, x):
        return self.scale(self.proj(x))
"""

LAYERS = """from torch import nn


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor
"""


def test_a_project_of_several_files_is_written_into_its_test(tmp_path):
    client = TestClient(create_app(tmp_path / "data"))
    made = project(
        client,
        code=ENTRY,
        class_name="Stack",
        constructor={},
        entry_path="pkg/model.py",
        files={"pkg/__init__.py": "", "pkg/layers.py": LAYERS},
    )
    run = client.post(f"/api/v1/projects/{made['id']}/runs").json()
    assert run["trace"]["error"] is None
    source = client.get(f"/api/v1/runs/{run['id']}/pytest").text
    assert "_ENTRY = 'pkg/model.py'" in source
    path = tmp_path / "exported" / "test_stack.py"
    path.parent.mkdir()
    path.write_text(source)
    passed = run_pytest(path)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    # A change in the imported file is caught by the module that changed.
    path.write_text(source.replace("return x * self.factor", "return x * self.factor + 1"))
    failed = run_pytest(path)
    assert failed.returncode == 1
    assert "expected" in failed.stdout
