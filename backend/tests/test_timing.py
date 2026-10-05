from tensorviewer.templates import TEMPLATES
from tensorviewer.worker import execute


def test_every_recorded_step_carries_how_long_its_call_took():
    trace = execute(TEMPLATES[1].project)
    assert trace.error is None
    assert trace.operations
    assert all(op.duration_us is not None and op.duration_us >= 0 for op in trace.operations)
    # The calls alone take less than the whole run, recording included.
    assert sum(op.duration_us for op in trace.operations) / 1000 < trace.duration_ms


def test_a_shapes_only_run_has_no_times_to_report():
    draft = TEMPLATES[1].project.model_copy(update={"capture_mode": "shapes"})
    trace = execute(draft)
    assert trace.error is None
    assert all(op.duration_us is None for op in trace.operations)


def test_timings_take_each_steps_median_over_more_passes(tmp_path):
    from fastapi.testclient import TestClient

    from tensorviewer.app import create_app

    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    project = client.post("/api/v1/projects", json=template).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    response = client.post(f"/api/v1/runs/{run['id']}/timings")
    assert response.status_code == 200
    found = response.json()
    assert found["passes"] == 5 and found["error"] is None
    assert len(found["durations_us"]) == len(run["trace"]["operations"])
    assert all(value is not None and value >= 0 for value in found["durations_us"])
    # Nothing is saved.
    assert len(client.get(f"/api/v1/projects/{project['id']}/runs").json()) == 1
    assert client.post("/api/v1/runs/missing/timings").status_code == 404
