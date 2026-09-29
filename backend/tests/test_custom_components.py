from fastapi.testclient import TestClient

from tensorviewer.app import create_app

SOURCE = """import torch
from torch import nn
FACTOR = 2
class RepeatFeatures(nn.Module):
    def __init__(self, repeats=FACTOR):
        super().__init__()
        self.repeats = repeats
    def forward(self, x):
        y = x.repeat_interleave(self.repeats, dim=-1)
        return y
"""


def save_component(client, code=SOURCE, **kwargs):
    response = client.post(
        "/api/v1/components",
        json={
            "name": "Repeat features",
            "description": "Repeat each feature.",
            "code": code,
            "class_name": "RepeatFeatures",
            "constructor": {"repeats": 2},
            **kwargs,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def composition(component):
    return {
        "blueprint": {
            "has_input": True,
            "components": [
                {"id": "custom-1", "kind": "custom", "custom": component},
                {"id": "linear-2", "kind": "linear", "parameters": {"features": 3}},
            ],
        },
        "input": {"shape": [2, 4], "axis_names": []},
    }


def test_custom_library_composes_runs_and_keeps_snapshots(tmp_path):
    client = TestClient(create_app(tmp_path))
    component = save_component(client)
    spec = composition(component)
    preview = client.post("/api/v1/compose", json=spec).json()
    assert preview["validation_required"] and not preview["valid"]
    assert preview["stages"][0]["shape"] is None
    plan = client.post("/api/v1/compose/check", json=spec).json()
    assert plan["valid"], plan["error"]
    assert [s["shape"] for s in plan["stages"]] == [[2, 8], [2, 3]]
    assert "nn.Linear(8, 3)" in plan["code"]
    assert client.post("/api/v1/compose", json=spec).json()["valid"]
    project = client.post(
        "/api/v1/projects",
        json={
            **spec,
            "name": "Reusable component study",
            "code": "placeholder",
        },
    ).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    assert run["trace"]["error"] is None, run["trace"]["error"]
    repeated = next(o for o in run["trace"]["operations"] if o["kind"] == "repeat_interleave")
    assert run["trace"]["tensors"][repeated["outputs"][0]]["name"] == "y"
    assert "repeat_interleave" in repeated["source"]["text"]
    assert (
        "repeat_interleave" in run["project"]["code"].splitlines()[repeated["source"]["line"] - 1]
    )
    assert run["trace"]["tensors"][repeated["outputs"][0]]["values"] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        7,
        7,
    ]
    new = save_component(client, code=SOURCE.replace("FACTOR = 2", "FACTOR = 3"))
    assert new["id"] != component["id"]
    restarted = TestClient(create_app(tmp_path))
    assert len(restarted.get("/api/v1/components").json()) == 2
    restored = restarted.get(f"/api/v1/projects/{project['id']}").json()
    assert restored["blueprint"]["components"][0]["custom"] == component
    assert restarted.post(f"/api/v1/projects/{project['id']}/runs").status_code == 409
    assert restarted.post("/api/v1/compose/check", json=spec).json()["valid"]


def test_passive_requests_do_not_execute_custom_code(tmp_path):
    marker = tmp_path / "executed"
    client = TestClient(create_app(tmp_path / "data"))
    component = save_component(
        client, code=f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + SOURCE
    )
    spec = composition(component)
    client.get("/api/v1/toolbox")
    client.post("/api/v1/compose", json=spec)
    client.post(
        "/api/v1/projects", json={**spec, "name": "No automatic execution", "code": "placeholder"}
    )
    assert not marker.exists()
    assert client.post("/api/v1/compose/check", json=spec).json()["valid"]
    assert marker.exists()


def test_shape_checks_invalidate_on_arguments_source_and_input_changes(tmp_path):
    client = TestClient(create_app(tmp_path))
    component = save_component(client)
    spec = composition(component)
    assert client.post("/api/v1/compose/check", json=spec).json()["valid"]
    spec["blueprint"]["components"][0]["arguments"] = {"repeats": 3}
    assert client.post("/api/v1/compose", json=spec).json()["validation_required"]
    plan = client.post("/api/v1/compose/check", json=spec).json()
    assert plan["stages"][0]["shape"] == [2, 12]
    spec["input"]["shape"] = [2, 5]
    assert client.post("/api/v1/compose", json=spec).json()["validation_required"]
    spec["blueprint"]["components"][0]["custom"]["code"] = SOURCE.replace("dim=-1", "dim=0")
    plan = client.post("/api/v1/compose/check", json=spec).json()
    assert plan["stages"][0]["shape"] == [6, 5]


def test_bad_custom_output_syntax_and_data_dependent_shapes(tmp_path):
    client = TestClient(create_app(tmp_path))
    for expression in ["(x, x)", "x.long()", "x[x > 0]"]:
        component = save_component(client, code=SOURCE.replace("return y", f"return {expression}"))
        plan = client.post("/api/v1/compose/check", json=composition(component)).json()
        assert not plan["valid"] and plan["error"]
        assert plan["stages"][0]["shape"] is None
    invalid = {**component, "code": "invalid python !"}
    assert client.post("/api/v1/components", json=invalid).status_code == 422
    assert (
        client.post("/api/v1/components", json={**component, "class_name": "Missing"}).status_code
        == 422
    )


def test_custom_namespaces_do_not_collide_and_retain_python_literals(tmp_path):
    client = TestClient(create_app(tmp_path))
    code = """from torch import nn
class RepeatFeatures(nn.Module):
    def __init__(self, options=None):
        super().__init__()
        self.options = options
    def forward(self, x):
        return x + (3 if self.options["enabled"] else 1)
"""
    custom = save_component(
        client, code=code, constructor={"options": {"enabled": True, "other": None}}
    )
    spec = composition(custom)
    spec["blueprint"]["components"][1] = {
        "id": "custom-2",
        "kind": "custom",
        "custom": {**custom, "code": code.replace("else 1", "else 7")},
        "arguments": {"options": {"enabled": False}},
    }
    plan = client.post("/api/v1/compose/check", json=spec).json()
    assert plan["valid"], plan["error"]
    project = client.post(
        "/api/v1/projects", json={**spec, "name": "Namespaces", "code": "placeholder"}
    ).json()
    trace = client.post(f"/api/v1/projects/{project['id']}/runs").json()["trace"]
    assert trace["error"] is None
    assert trace["tensors"][trace["output_ids"][0]]["values"] == list(range(10, 18))


def test_check_uses_actual_sequence_strides_and_blocks_stale_preview(tmp_path):
    client = TestClient(create_app(tmp_path))
    component = save_component(
        client, code=SOURCE.replace("x.repeat_interleave(self.repeats, dim=-1)", "x.view(-1)")
    )
    spec = composition(component)
    spec["blueprint"]["components"] = [
        {"id": "transpose", "kind": "transpose", "parameters": {"axis_a": 0, "axis_b": 1}},
        spec["blueprint"]["components"][0],
    ]
    plan = client.post("/api/v1/compose/check", json=spec).json()
    assert not plan["valid"] and "Sequence check" in plan["error"]
    assert plan["stages"][1]["shape"] is None
    assert not client.post("/api/v1/compose", json=spec).json()["valid"]
    spec["blueprint"]["components"].insert(1, {"id": "contiguous", "kind": "contiguous"})
    assert client.post("/api/v1/compose", json=spec).json()["validation_required"]
    assert client.post("/api/v1/compose/check", json=spec).json()["valid"]


def test_model_checks_limits_and_rejects_output_contract_changes_at_runtime(tmp_path):
    client = TestClient(create_app(tmp_path))
    component = save_component(
        client, code=SOURCE.replace("return y", "return y if x.is_meta else (y, y)")
    )
    spec = composition(component)
    assert client.post("/api/v1/compose/check", json=spec).json()["valid"]
    project = client.post(
        "/api/v1/projects", json={**spec, "name": "Contract", "code": "placeholder"}
    ).json()
    trace = client.post(f"/api/v1/projects/{project['id']}/runs").json()["trace"]
    assert "does not match" in trace["error"]["message"]
    component = save_component(
        client,
        code=SOURCE.replace(
            "self.repeats = repeats",
            "self.repeats = repeats\n        self.weight = nn.Parameter(torch.empty(9000000))",
        ),
    )
    spec = composition(component)
    plan = client.post("/api/v1/compose/check", json=spec).json()
    assert not plan["valid"] and "weights" in plan["error"]
    spec["capture_mode"] = "shapes"
    assert client.post("/api/v1/compose/check", json=spec).json()["valid"]
