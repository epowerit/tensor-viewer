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


def test_deleting_a_project_removes_its_runs_and_snapshots(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    project = client.post("/api/v1/projects", json=template).json()
    kept = client.post("/api/v1/projects", json={**template, "name": "Kept"}).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    other = client.post(f"/api/v1/projects/{kept['id']}/runs").json()
    # Small runs keep their values inline; a large one keeps a snapshot here.
    (tmp_path / "snapshots" / run["id"]).mkdir(exist_ok=True)
    (tmp_path / "snapshots" / run["id"] / "t0.npy").write_bytes(b"")
    assert client.delete(f"/api/v1/projects/{project['id']}").status_code == 204
    assert client.get(f"/api/v1/projects/{project['id']}").status_code == 404
    assert client.get(f"/api/v1/runs/{run['id']}").status_code == 404
    assert not (tmp_path / "snapshots" / run["id"]).exists()
    assert [p["id"] for p in client.get("/api/v1/projects").json()] == [kept["id"]]
    assert client.get(f"/api/v1/runs/{other['id']}").status_code == 200
    assert client.delete(f"/api/v1/projects/{project['id']}").status_code == 404


def test_runs_are_deleted_one_at_a_time_or_all_but_the_newest(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    project = client.post("/api/v1/projects", json=template).json()
    runs = [client.post(f"/api/v1/projects/{project['id']}/runs").json() for _ in range(4)]
    (tmp_path / "snapshots" / runs[0]["id"]).mkdir(exist_ok=True)
    assert client.delete(f"/api/v1/runs/{runs[0]['id']}").status_code == 204
    assert not (tmp_path / "snapshots" / runs[0]["id"]).exists()
    assert client.delete(f"/api/v1/runs/{runs[0]['id']}").status_code == 404
    # Older than the third run: the second goes; a run recorded since stays.
    newer = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    response = client.delete(
        f"/api/v1/projects/{project['id']}/runs", params={"before": runs[2]["id"]}
    )
    assert response.json() == {"deleted": 1}
    left = client.get(f"/api/v1/projects/{project['id']}/runs").json()
    assert [run["id"] for run in left] == [newer["id"], runs[3]["id"], runs[2]["id"]]
    missing = client.delete(f"/api/v1/projects/{project['id']}/runs", params={"before": "x"})
    assert missing.status_code == 404
    assert client.delete("/api/v1/projects/missing/runs", params={"before": "x"}).status_code == 404


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


def test_large_snapshots_load_exact_bounded_windows_and_survive_mutation(tmp_path):
    client = TestClient(create_app(tmp_path))
    draft = {
        "name": "Large values",
        "class_name": "Example",
        "constructor": {},
        "input": {"shape": [128, 32, 32], "axis_names": []},
        "code": "from torch import nn\nclass Example(nn.Module):\n def forward(self,x):\n  y=x.permute(0,2,1)\n  x.add_(10)\n  return y",
    }
    project = client.post("/api/v1/projects", json=draft).json()
    response = client.post(f"/api/v1/projects/{project['id']}/runs")
    run = response.json()
    assert response.status_code == 201
    assert run["trace"]["error"] is None
    assert len(response.content) < 20000
    before = run["trace"]["input_ids"][0]
    viewed = run["trace"]["operations"][0]["outputs"][0]
    after = run["trace"]["output_ids"][0]
    mutations = run["trace"]["operations"][1]["mutations"]
    assert {"before": viewed, "after": after, "kind": "alias"} in mutations
    base = f"/api/v1/runs/{run['id']}/tensors"
    assert run["trace"]["tensors"][before]["value_source"] == "paged"
    assert client.get(f"{base}/{before}/values", params={"indices": "0,65536,131071"}).json()[
        "values"
    ] == [0, 65536, 131071]
    assert client.get(f"{base}/{viewed}/values", params={"indices": "1,131071"}).json()[
        "values"
    ] == [32, 131071]
    assert client.get(f"{base}/{after}/values", params={"indices": "1,131071"}).json()[
        "values"
    ] == [42, 131081]
    restarted = TestClient(create_app(tmp_path))
    assert (
        restarted.get(f"/api/v1/runs/{run['id']}").json()["trace"]["operations"][1]["mutations"]
        == mutations
    )
    assert restarted.get(f"{base}/{after}/values", params={"indices": "131071"}).json()[
        "values"
    ] == [131081]
    for indices in ["-1", "131072", "abc", ",".join("0" for _ in range(257))]:
        assert client.get(f"{base}/{before}/values", params={"indices": indices}).status_code == 422
    assert client.get(f"{base}/missing/values", params={"indices": "0"}).status_code == 404


def test_what_if_runs_the_same_code_with_input_cells_set_outside_history(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    project = client.post("/api/v1/projects", json=template).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    response = client.post(
        f"/api/v1/runs/{run['id']}/what-if", json={"edits": [{"index": 3, "value": 42}]}
    )
    assert response.status_code == 201
    scratch = response.json()
    assert scratch["id"].startswith("what-if-")
    assert scratch["project"]["input"]["edits"] == [{"index": 3, "value": 42.0}]
    trace = scratch["trace"]
    x = trace["tensors"][trace["input_ids"][0]]["values"]
    assert x[:5] == [0, 1, 2, 42, 4]
    # The edited cell moves where the reshape and permute take it.
    packed = trace["tensors"][trace["output_ids"][0]]["values"]
    assert 42 in packed and packed.count(42) == 1
    # It reads back like a run but stays out of the project's history.
    assert client.get(f"/api/v1/runs/{scratch['id']}").status_code == 200
    assert [r["id"] for r in client.get(f"/api/v1/projects/{project['id']}/runs").json()] == [
        run["id"]
    ]
    assert (
        client.post(
            f"/api/v1/runs/{run['id']}/what-if", json={"edits": [{"index": 99, "value": 1}]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/runs/missing/what-if", json={"edits": [{"index": 0, "value": 1}]}
        ).status_code
        == 404
    )


def test_sensitivity_is_the_gradient_of_one_result_cell_with_respect_to_the_input(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = template["code"].replace("return packed", "return (packed * packed).sum(-1)")
    project = client.post("/api/v1/projects", json={**template, "code": code}).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    trace = run["trace"]
    out = trace["output_ids"][0]
    x = trace["tensors"][trace["input_ids"][0]]["values"]
    found = client.post(
        f"/api/v1/runs/{run['id']}/sensitivity", json={"tensor_id": out, "index": 0}
    )
    assert found.status_code == 200, found.text
    body = found.json()
    assert body["kind"] == "gradient" and len(body["values"]) == len(x)
    # out[0] = sum of the squares of x[0, 0, 0:4]: its gradient is 2·x there, 0 elsewhere.
    assert body["values"][:4] == [2 * v for v in x[:4]]
    assert all(v == 0 for v in body["values"][4:])
    assert body["value"] == sum(v * v for v in x[:4])
    missing = client.post(
        f"/api/v1/runs/{run['id']}/sensitivity", json={"tensor_id": "nope", "index": 0}
    )
    assert missing.status_code == 404


def test_token_sensitivity_is_the_size_of_the_gradient_at_each_words_embedding(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Running(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)

    def forward(self, x):
        return self.embed(x).cumsum(1)
"""
    draft = {
        **template,
        "code": code,
        "class_name": "Running",
        "constructor": {},
        "input": {
            **template["input"],
            "generator": "text",
            "text": "the cat sat",
            "dtype": "int64",
            "shape": [1, 3],
            "axis_names": ["batch", "tokens"],
        },
    }
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    assert run["trace"]["error"] is None, run["trace"]["error"]
    out = run["trace"]["output_ids"][0]
    # The running sum at the second word reads the first two words only.
    body = client.post(
        f"/api/v1/runs/{run['id']}/sensitivity", json={"tensor_id": out, "index": 4}
    ).json()
    assert body["kind"] == "embedding"
    first, second, third = body["values"]
    assert first == second == 1.0 and third == 0


def test_gradient_flow_gives_the_gradient_size_at_every_tensor_before_the_target(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = template["code"].replace("return packed", "return packed * 3")
    project = client.post("/api/v1/projects", json={**template, "code": code}).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    trace = run["trace"]
    out = trace["output_ids"][0]
    found = client.post(f"/api/v1/runs/{run['id']}/gradients", json={"tensor_id": out})
    assert found.status_code == 200, found.text
    norms = found.json()["norms"]
    # ∂ sum(out) / ∂ out is all ones over 16 cells; before the × 3, all threes.
    assert norms[out] == 4
    x = trace["input_ids"][0]
    assert norms[x] == 12
    packed = next(op["outputs"][0] for op in trace["operations"] if op["kind"] == "contiguous")
    assert norms[packed] == 12
    # One cell: only the cells feeding it carry a gradient.
    cell = client.post(
        f"/api/v1/runs/{run['id']}/gradients", json={"tensor_id": out, "index": 0}
    ).json()["norms"]
    assert cell[out] == 1 and cell[x] == 3
    # Steps after the target are left out.
    first = trace["operations"][0]["outputs"][0]
    early = client.post(f"/api/v1/runs/{run['id']}/gradients", json={"tensor_id": first}).json()[
        "norms"
    ]
    assert out not in early and first in early


def test_a_what_if_can_run_the_same_code_in_bfloat16(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = template["code"].replace("return packed", "return packed / 3")
    project = client.post("/api/v1/projects", json={**template, "code": code}).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    response = client.post(f"/api/v1/runs/{run['id']}/what-if", json={"precision": "bfloat16"})
    assert response.status_code == 201, response.text
    trace = response.json()["trace"]
    out = trace["tensors"][trace["output_ids"][0]]
    assert out["dtype"] == "bfloat16"
    exact = run["trace"]["tensors"][run["trace"]["output_ids"][0]]["values"]
    # bfloat16 keeps 8 significant bits: close, but not the float32 values.
    assert out["values"] != exact
    assert all(abs(a - b) <= abs(b) / 128 for a, b in zip(out["values"], exact))
    assert client.post(f"/api/v1/runs/{run['id']}/what-if", json={}).status_code == 422


def test_a_what_if_can_train_the_weights_one_step_toward_a_value(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))

    def forward(self, x):
        return x * self.weight
"""
    draft = {**template, "code": code, "class_name": "Scale", "constructor": {}}
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    trace = run["trace"]
    out = trace["output_ids"][0]
    x = trace["tensors"][trace["input_ids"][0]]["values"]
    step = {"tensor_id": out, "index": 3, "rate": 0.5, "direction": 1}
    response = client.post(f"/api/v1/runs/{run['id']}/what-if", json={"learn": step})
    assert response.status_code == 201, response.text
    learned = response.json()["trace"]
    after = learned["tensors"][learned["output_ids"][0]]["values"]
    # out[3] = x[3] · w[3]: ∂/∂w[3] = x[3], so w[3] becomes 1 + 0.5 · x[3].
    assert after[3] == x[3] * (1 + 0.5 * x[3])
    # Only that weight moved; out[11] reads w[3] too, other cells do not change.
    assert after[11] == x[11] * (1 + 0.5 * x[3])
    assert after[2] == x[2]
    lower = client.post(
        f"/api/v1/runs/{run['id']}/what-if", json={"learn": {**step, "direction": -1}}
    ).json()["trace"]
    assert lower["tensors"][lower["output_ids"][0]]["values"][3] < x[3]
    # On the log of the value: ∂ log(x·w) / ∂w = 1/w, so w[3] becomes 1.5.
    logged = client.post(
        f"/api/v1/runs/{run['id']}/what-if", json={"learn": {**step, "log": True}}
    ).json()["trace"]
    assert logged["tensors"][logged["output_ids"][0]]["values"][3] == x[3] * 1.5
    # Several steps: the value after each, from 3 (= x[3] · 1) upward.
    many = client.post(
        f"/api/v1/runs/{run['id']}/what-if", json={"learn": {**step, "steps": 3}}
    ).json()["trace"]
    curve = many["learn_curve"]
    assert len(curve) == 4 and curve[0] == x[3]
    # Each step adds rate · x[3] to w[3]: w = 1 + 0.5 · 3 · k after k steps.
    assert curve == [x[3] * (1 + 0.5 * x[3] * k) for k in range(4)]
    assert many["tensors"][many["output_ids"][0]]["values"][3] == curve[-1]


def test_training_on_the_sentence_lowers_the_next_word_loss(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 8)
        self.head = nn.Linear(8, 5)

    def forward(self, tokens):
        return self.head(self.embed(tokens)).softmax(-1)
"""
    draft = {
        **template,
        "code": code,
        "class_name": "Bigram",
        "constructor": {},
        "input_name": "tokens",
        "input": {
            **template["input"],
            "generator": "text",
            "text": "the cat sat on the mat",
            "dtype": "int64",
            "shape": [1, 6],
            "axis_names": ["batch", "tokens"],
        },
    }
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    assert run["trace"]["error"] is None, run["trace"]["error"]
    probs = run["trace"]["output_ids"][0]
    step = {
        "tensor_id": probs,
        "index": 0,
        "rate": 1,
        "log": True,
        "sentence": True,
        "steps": 20,
    }
    response = client.post(f"/api/v1/runs/{run['id']}/what-if", json={"learn": step})
    assert response.status_code == 201, response.text
    curve = response.json()["trace"]["learn_curve"]
    assert len(curve) == 21
    # Mean cross-entropy of each word predicting the next: it falls.
    assert curve[-1] < curve[0] / 2
    assert all(b <= a + 1e-6 for a, b in zip(curve, curve[1:]))


def test_watch_expressions_read_the_named_tensors_at_a_step(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    project = client.post("/api/v1/projects", json=template).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    trace = run["trace"]
    ops = trace["operations"]

    def watch(expression, at=None):
        response = client.post(
            f"/api/v1/runs/{run['id']}/evaluate", json={"expression": expression, "at": at}
        )
        assert response.status_code == 200, response.text
        return response.json()

    # x is the input: arange over [1, 2, 8].
    whole = watch("x.sum()")
    assert whole["kind"] == "tensor" and whole["shape"] == [] and whole["values"] == [120]
    shaped = watch("grouped.shape")
    assert shaped["kind"] == "value" and shaped["text"] == "torch.Size([1, 2, 2, 4])"
    rows = watch("x.float().mean(-1)")
    assert rows["shape"] == [1, 2] and rows["values"] == [3.5, 11.5]
    assert rows["stats"]["max"] == 11.5
    # Before the step that makes it, a name is not there yet.
    early = watch("packed", at=ops[0]["id"])
    assert early["kind"] == "error" and early["text"] == "packed is not computed yet at this step"
    assert "not defined" in watch("nothing_like_this")["text"]
    assert "grouped" in early["names"] and "packed" not in early["names"]
    assert watch("packed.shape")["text"] == "torch.Size([1, 2, 2, 4])"
    assert watch("1 +")["kind"] == "error"


def test_a_watch_series_reads_a_number_at_each_step_its_names_change(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Steps(nn.Module):
    def forward(self, x):
        x = x * 2
        y = x - 1
        x = x + 1
        return x + y
"""
    draft = {**template, "code": code, "class_name": "Steps", "constructor": {}}
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    series = client.post(
        f"/api/v1/runs/{run['id']}/evaluate-series", json={"expression": "x.mean()"}
    ).json()
    # The input's arange over 16 values has mean 7.5; doubled 15; plus one 16.
    assert [p["value"] for p in series["points"]] == [7.5, 15, 16]
    assert series["points"][0]["step"] == 0 and series["points"][0]["at"].startswith("input-")
    # y's step does not change x, so it adds no point.
    assert [p["step"] for p in series["points"]][1:] == [1, 3]
    shaped = client.post(
        f"/api/v1/runs/{run['id']}/evaluate-series", json={"expression": "x"}
    ).json()
    assert all(p["value"] is None and "single number" in p["error"] for p in shaped["points"])


def test_weight_spectra_give_each_weights_singular_values_and_rank(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Diagonal(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.diag(torch.arange(8.0)))
        self.b = nn.Parameter(torch.ones(8))

    def forward(self, x):
        return x @ self.w + self.b
"""
    draft = {**template, "code": code, "class_name": "Diagonal", "constructor": {}}
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    report = client.post(f"/api/v1/runs/{run['id']}/weights").json()
    assert report["error"] is None
    by_name = {w["name"]: w for w in report["weights"]}
    w = by_name["w"]
    assert w["shape"] == [8, 8]
    assert [round(v, 6) for v in w["singular"]] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert w["rank"] == 7 and w["full"] == 8 and w["condition"] is None
    assert 5 < w["effective_rank"] < 7
    # A vector has a norm but no spectrum.
    assert by_name["b"]["norm"] == 8**0.5 and by_name["b"]["singular"] == []


def test_a_run_names_the_model_buffers_apart_from_its_weights(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Masked(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(8, 8))
        self.register_buffer("mask", torch.tril(torch.ones(8, 8)))

    def forward(self, x):
        return x @ (self.w * self.mask)
"""
    draft = {**template, "code": code, "class_name": "Masked", "constructor": {}}
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    trace = run["trace"]
    assert trace["buffer_names"] == ["mask"]
    # Both are recorded as the model's own tensors; only the name tells them apart.
    roles = {t["name"]: t["role"] for t in trace["tensors"].values()}
    assert roles["w"] == roles["mask"] == "parameter"


def test_weight_updates_against_another_run_have_their_own_rank(tmp_path):
    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    code = """import torch
from torch import nn


class Mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.eye(8))

    def forward(self, x):
        return x @ self.w
"""
    draft = {**template, "code": code, "class_name": "Mix", "constructor": {}}
    project = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    out = run["trace"]["output_ids"][0]
    step = {"tensor_id": out, "index": 3, "rate": 0.1, "direction": 1}
    learned = client.post(f"/api/v1/runs/{run['id']}/what-if", json={"learn": step}).json()
    report = client.post(
        f"/api/v1/runs/{learned['id']}/weights", json={"against": run["id"]}
    ).json()
    update = report["weights"][0]["update"]
    # One step on out[0, 0, 3] = x[0, 0] · w[:, 3]: ΔW = 0.1 · x[0, 0]ᵀ e₃, rank one.
    assert update["rank"] == 1 and round(update["effective_rank"], 6) == 1
    x = run["trace"]["tensors"][run["trace"]["input_ids"][0]]["values"][:8]
    expected = 0.1 * sum(v * v for v in x) ** 0.5
    assert abs(update["norm"] - expected) < 1e-5
    assert abs(update["relative"] - expected / 8**0.5) < 1e-5
    # Without another run there is no update.
    assert client.post(f"/api/v1/runs/{run['id']}/weights").json()["weights"][0]["update"] is None
