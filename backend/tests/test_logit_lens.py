from fastapi.testclient import TestClient

from tensorviewer.app import create_app

CODE = """import torch
from torch import nn


class Tiny(nn.Module):
    def __init__(self, vocabulary=8, dim=4):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, dim)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = x + block(x)
        # The head shares the embedding's weights, as GPT's does.
        return (self.norm(x) @ self.embed.weight.T).softmax(-1)
"""


def recorded(client, code=CODE, class_name="Tiny"):
    template = client.get("/api/v1/templates").json()[1]["project"]
    draft = {
        **template,
        "code": code,
        "class_name": class_name,
        "constructor": {},
        "input_name": "tokens",
        "input": {
            **template["input"],
            "shape": [1, 5],
            "dtype": "int64",
            "generator": "arange",
            "axis_names": ["batch", "tokens"],
        },
    }
    project = client.post("/api/v1/projects", json=draft).json()
    return client.post(f"/api/v1/projects/{project['id']}/runs").json()


def test_every_layer_is_read_by_the_final_layers(tmp_path):
    client = TestClient(create_app(tmp_path))
    run = recorded(client)
    assert run["trace"]["error"] is None
    response = client.post(f"/api/v1/runs/{run['id']}/logit-lens")
    assert response.status_code == 200
    lens = response.json()
    assert [state["name"] for state in lens["states"]] == [
        "before blocks.0",
        "blocks.0",
        "blocks.1",
        "blocks.2",
    ]
    # Five positions, three likeliest ids each, probabilities in order.
    for state in lens["states"]:
        assert len(state["top"]) == 5
        for top in state["top"]:
            assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1]
    # The last layer's reading is the model's own prediction.
    trace = run["trace"]
    output = trace["tensors"][trace["output_ids"][0]]["values"]
    rows = [output[i * 8 : (i + 1) * 8] for i in range(5)]
    assert [top[0][0] for top in lens["states"][-1]["top"]] == [row.index(max(row)) for row in rows]
    # Earlier layers read differently: the patch reached the final layers.
    assert lens["states"][0]["top"] != lens["states"][-1]["top"]


def test_a_model_without_a_block_stack_is_refused(tmp_path):
    client = TestClient(create_app(tmp_path))
    flat = CODE.replace("for block in self.blocks:\n            x = x + block(x)", "x = x * 2")
    run = recorded(client, flat)
    refused = client.post(f"/api/v1/runs/{run['id']}/logit-lens")
    assert refused.status_code == 422
    assert "repeated blocks" in refused.json()["detail"]
    assert client.post("/api/v1/runs/missing/logit-lens").status_code == 404


def test_a_trained_what_if_is_read_with_its_trained_weights(tmp_path):
    client = TestClient(create_app(tmp_path))
    run = recorded(client)
    trace = run["trace"]
    learn = {
        "tensor_id": trace["output_ids"][0],
        "index": 0,
        "rate": 0.5,
        "direction": 1,
        "log": True,
        "steps": 40,
        "sentence": True,
    }
    trained = client.post(f"/api/v1/runs/{run['id']}/what-if", json={"learn": learn}).json()
    assert trained["project"]["input"]["learn"]["steps"] == 40
    curve = trained["trace"]["learn_curve"]
    assert curve[-1] < curve[0]
    lens = client.post(f"/api/v1/runs/{trained['id']}/logit-lens").json()
    output = trained["trace"]["tensors"][trained["trace"]["output_ids"][0]]["values"]
    rows = [output[i * 8 : (i + 1) * 8] for i in range(5)]
    # The last layer reads as the trained model predicts: each id the next.
    assert [top[0][0] for top in lens["states"][-1]["top"]] == [row.index(max(row)) for row in rows]
    assert [top[0][0] for top in lens["states"][-1]["top"]][:4] == [1, 2, 3, 4]


def test_gradients_of_a_trained_what_if_use_its_trained_weights(tmp_path):
    client = TestClient(create_app(tmp_path))
    run = recorded(client)
    output = run["trace"]["output_ids"][0]
    learn = {"tensor_id": output, "index": 3, "rate": 0.5, "steps": 10, "log": True}
    trained = client.post(f"/api/v1/runs/{run['id']}/what-if", json={"learn": learn}).json()
    ask = {"tensor_id": output, "index": 3}
    before = client.post(f"/api/v1/runs/{run['id']}/gradients", json=ask).json()["norms"]
    after = client.post(f"/api/v1/runs/{trained['id']}/gradients", json=ask).json()["norms"]
    assert before.keys() == after.keys()
    assert any(abs(before[k] - after[k]) > 1e-6 for k in before)
