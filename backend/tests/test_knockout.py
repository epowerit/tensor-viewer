import pytest
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.models import Knockout

CODE = """import torch
from torch import nn


class Rows(nn.Module):
    def forward(self, x):
        doubled = x * 2
        shifted = doubled.add_(1)
        rows = shifted.sum(-1)
        return rows
"""


def project(client, **input):
    template = client.get("/api/v1/templates").json()[1]["project"]
    draft = {
        **template,
        "code": CODE,
        "class_name": "Rows",
        "constructor": {},
        "input": {**template["input"], "shape": [1, 3, 4], **input},
    }
    return client.post("/api/v1/projects", json=draft).json()


def output(run):
    trace = run["trace"]
    return trace["tensors"][trace["output_ids"][0]]["values"]


@pytest.fixture
def recorded(tmp_path):
    client = TestClient(create_app(tmp_path))
    made = project(client)
    run = client.post(f"/api/v1/projects/{made['id']}/runs").json()
    # x = 0..11 in rows of 4: each row sums to 2·Σx + 4.
    assert output(run) == [16, 48, 80]
    return client, made, run


def what_if(client, run, **knockout):
    return client.post(f"/api/v1/runs/{run['id']}/what-if", json={"knockout": knockout})


def test_a_knockout_replaces_a_step_result_before_anything_reads_it(recorded):
    client, _, run = recorded
    response = what_if(client, run, step=0, mode="zero")
    assert response.status_code == 201
    scratch = response.json()
    assert scratch["id"].startswith("what-if-")
    assert scratch["trace"]["error"] is None
    # doubled is zero, so each row is four ones.
    assert output(scratch) == [4, 4, 4]
    # The same steps are recorded, so they line up with the recorded run's.
    assert [op["kind"] for op in scratch["trace"]["operations"]] == [
        op["kind"] for op in run["trace"]["operations"]
    ]


def test_a_knockout_can_take_one_slice_or_put_in_the_mean(recorded):
    client, _, run = recorded
    one_row = what_if(client, run, step=0, mode="zero", axis=1, index=1).json()
    assert output(one_row) == [16, 4, 80]
    # Row 1 becomes the mean row over axis 1, which is row 1 itself here.
    average = what_if(client, run, step=0, mode="mean", axis=1, index=0).json()
    assert output(average) == [48, 48, 80]
    # The whole result's mean is 11 in every cell.
    assert output(what_if(client, run, step=0, mode="mean").json()) == [48, 48, 48]


def test_an_in_place_step_is_written_over(recorded):
    client, _, run = recorded
    # add_ writes into doubled; the sum reads that tensor.
    scratch = what_if(client, run, step=1, mode="zero").json()
    assert scratch["trace"]["error"] is None
    assert output(scratch) == [0, 0, 0]


def test_a_patch_puts_in_another_run_result_at_the_same_step(tmp_path):
    client = TestClient(create_app(tmp_path))
    ones = project(client, generator="ones")
    clean = client.post(f"/api/v1/projects/{ones['id']}/runs").json()
    assert output(clean) == [12, 12, 12]
    counting = project(client)
    corrupt = client.post(f"/api/v1/projects/{counting['id']}/runs").json()
    patched = what_if(client, corrupt, step=0, mode="patch", patch_from=clean["id"]).json()
    assert output(patched) == [12, 12, 12]
    # Only the middle row patched in.
    row = what_if(
        client, corrupt, step=0, mode="patch", axis=1, index=1, patch_from=clean["id"]
    ).json()
    assert output(row) == [16, 12, 80]
    missing = what_if(client, corrupt, step=0, mode="patch", patch_from="nope")
    assert missing.status_code == 404


def test_a_knockout_outside_the_run_is_refused(recorded):
    client, _, run = recorded
    assert what_if(client, run, step=9, mode="zero").status_code == 422
    assert what_if(client, run, step=0, mode="zero", axis=1).status_code == 422
    assert what_if(client, run, step=0, mode="patch").status_code == 422
    # A slice past the axis is found as the run makes the step.
    scratch = what_if(client, run, step=0, mode="zero", axis=1, index=5).json()
    assert "no slice 5 along axis 1" in scratch["trace"]["error"]["message"]


def test_a_request_cannot_choose_the_file_a_patch_reads(recorded):
    client, _, run = recorded
    scratch = what_if(client, run, step=0, mode="zero", patch_path="/etc/hosts").json()
    assert scratch["project"]["input"]["knockout"]["patch_path"] is None


def test_a_sweep_knocks_out_each_slice_in_turn(recorded):
    client, _, run = recorded
    response = client.post(
        f"/api/v1/runs/{run['id']}/knockout-sweep", json={"step": 0, "axis": 1, "mode": "zero"}
    )
    assert response.status_code == 200
    sweep = response.json()
    assert sweep["error"] is None
    # The output's largest value is the last row's 80; each row moves only itself.
    assert sweep["cell"] == 2 and sweep["cell_value"] == 80
    assert sweep["cell_values"] == [80, 80, 4]
    norm = (16**2 + 48**2 + 80**2) ** 0.5
    assert sweep["effects"] == pytest.approx([12 / norm, 44 / norm, 76 / norm])
    # Nothing was saved.
    assert len(client.get(f"/api/v1/projects/{run['project_id']}/runs").json()) == 1
    too_far = client.post(
        f"/api/v1/runs/{run['id']}/knockout-sweep", json={"step": 0, "axis": 3, "mode": "zero"}
    )
    assert too_far.status_code == 422


def test_a_knockout_names_its_slice_whole():
    with pytest.raises(ValueError):
        Knockout(step=0, axis=1)
    with pytest.raises(ValueError):
        Knockout(step=0, mode="zero", patch_from="run")
