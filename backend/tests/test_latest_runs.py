from fastapi.testclient import TestClient

from tensorviewer.app import create_app

CODE = """
import torch
from torch import nn


class Model(nn.Module):
    def forward(self, x):
        return x * 2
"""


def test_latest_runs_give_each_projects_newest_run(tmp_path):
    client = TestClient(create_app(tmp_path))
    created = client.post(
        "/api/v1/projects",
        json={
            "name": "double",
            "code": CODE,
            "class_name": "Model",
            "constructor": {},
            "input": {"shape": [2, 3], "axis_names": [], "generator": "arange"},
        },
    ).json()
    assert client.get("/api/v1/latest-runs").json() == []
    first = client.post(f"/api/v1/projects/{created['id']}/runs").json()
    second = client.post(f"/api/v1/projects/{created['id']}/runs").json()
    latest = client.get("/api/v1/latest-runs").json()
    assert [item["run_id"] for item in latest] == [second["id"]]
    assert latest[0]["failed"] is False and latest[0]["operation_count"] == 1
    assert first["id"] != second["id"]
