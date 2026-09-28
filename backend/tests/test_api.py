from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.runner import run_project
from tensorviewer.templates import TEMPLATES


def test_project_lifecycle_and_immutable_run_snapshot(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/v1/projects").json() == []
    template = client.get("/api/v1/templates").json()[1]["project"]
    response = client.post("/api/v1/projects", json=template)
    assert response.status_code == 201
    project = response.json()
    response = client.post(f"/api/v1/projects/{project['id']}/runs")
    assert response.status_code == 201
    run = response.json()
    assert run["trace"]["error"] is None
    assert len(run["trace"]["operations"]) == 3
    revised = {**template, "code": template["code"].replace("return packed", "return packed + 1")}
    assert client.put(f"/api/v1/projects/{project['id']}", json=revised).status_code == 200
    assert client.get(f"/api/v1/runs/{run['id']}").json()["project"]["code"] == template["code"]
    assert client.get(f"/api/v1/projects/{project['id']}/runs").json()[0]["operation_count"] == 3
    restarted = TestClient(create_app(tmp_path))
    assert restarted.get(f"/api/v1/projects/{project['id']}").json()["code"] == revised["code"]
    assert client.get("/api/v1/projects/missing").status_code == 404
    assert client.get("/api/v1/runs/missing").status_code == 404


def test_timeout_kills_worker_and_next_run_still_works():
    draft = TEMPLATES[1].project.model_copy(deep=True)
    draft.code = "while True: pass"
    trace = run_project(draft, timeout=0.5)
    assert trace.error and trace.error.type == "TimeoutError"
    assert run_project(TEMPLATES[1].project).error is None


def test_invalid_input_and_cross_origin_are_rejected(tmp_path):
    client = TestClient(create_app(tmp_path))
    draft = TEMPLATES[0].project.model_dump()
    draft["input"]["shape"] = [-1, 3, 8]
    assert client.post("/api/v1/projects", json=draft).status_code == 422
    response = client.options(
        "/api/v1/projects",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 400
