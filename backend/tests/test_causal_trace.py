import pytest
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tests.test_logit_lens import recorded

# Blocks return the residual stream, as a transformer's do.
CODE = """import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, mix):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.mix = mix

    def forward(self, x):
        x = x + self.linear(x)
        if self.mix:
            # Each position reads the ones before it, as attention would.
            x = x + 0.5 * torch.cumsum(x, dim=1) / x.shape[1]
        return x


class Tiny(nn.Module):
    def __init__(self, vocabulary=8, dim=4, mix=False):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, dim)
        self.blocks = nn.ModuleList([Block(dim, mix) for _ in range(3)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return (self.norm(x) @ self.embed.weight.T).softmax(-1)
"""
MIXING = CODE.replace("mix=False", "mix=True")


def changed(client, run, index=2, value=7):
    response = client.post(
        f"/api/v1/runs/{run['id']}/what-if", json={"edits": [{"index": index, "value": value}]}
    )
    return response.json()


def trace(client, run, against):
    return client.post(f"/api/v1/runs/{run['id']}/causal-trace", json={"against": against["id"]})


def test_patching_the_changed_position_recovers_the_clean_result(tmp_path):
    client = TestClient(create_app(tmp_path))
    clean = recorded(client, CODE)
    corrupt = changed(client, clean)
    response = trace(client, corrupt, clean)
    assert response.status_code == 200
    found = response.json()
    assert found["states"] == ["before blocks.0", "blocks.0", "blocks.1", "blocks.2"]
    assert found["positions"] == 5
    # Positions never mix here: only the changed one matters, at every layer.
    for row in found["recovery"]:
        assert row[2] == pytest.approx(1)
        assert [row[i] for i in (0, 1, 3, 4)] == pytest.approx([0, 0, 0, 0], abs=1e-9)


def test_where_positions_mix_later_layers_spread_the_change(tmp_path):
    client = TestClient(create_app(tmp_path))
    clean = recorded(client, MIXING)
    corrupt = changed(client, clean)
    found = trace(client, corrupt, clean).json()
    first, last = found["recovery"][0], found["recovery"][-1]
    # Before the blocks, the change sits at its own position alone.
    assert first[2] == pytest.approx(1)
    # After them, it has spread to the positions that read it.
    assert last[2] < 1
    assert last[3] > 0 and last[4] > 0


def test_a_trace_needs_two_different_runs_of_one_model(tmp_path):
    client = TestClient(create_app(tmp_path))
    clean = recorded(client, CODE)
    same = trace(client, clean, clean)
    assert same.status_code == 422
    assert "same result" in same.json()["detail"]
    missing = client.post(f"/api/v1/runs/{clean['id']}/causal-trace", json={"against": "nope"})
    assert missing.status_code == 404


def test_runs_with_different_weights_are_not_traced(tmp_path):
    client = TestClient(create_app(tmp_path))
    clean = recorded(client, CODE)
    output = clean["trace"]["output_ids"][0]
    learn = {"tensor_id": output, "index": 0, "rate": 0.1, "steps": 2}
    trained = client.post(
        f"/api/v1/runs/{clean['id']}/what-if",
        json={"learn": learn, "edits": [{"index": 2, "value": 7}]},
    ).json()
    refused = trace(client, trained, clean)
    assert refused.status_code == 422
    assert "weights differ" in refused.json()["detail"]


def test_a_model_that_picks_its_next_token_is_traced_on_its_scores(tmp_path):
    # The result is one token id, which barely moves; the trace measures the
    # probabilities the model picks it from, as the logit lens reads them.
    picking = CODE.replace(
        "return (self.norm(x) @ self.embed.weight.T).softmax(-1)",
        "probabilities = (self.norm(x) @ self.embed.weight.T).softmax(-1)\n"
        "        return probabilities.argmax(dim=-1, keepdim=True)[:, -1, :]",
    )
    client = TestClient(create_app(tmp_path))
    clean = recorded(client, picking)
    corrupt = changed(client, clean)
    response = trace(client, corrupt, clean)
    assert response.status_code == 200, response.json()
    for row in response.json()["recovery"]:
        assert row[2] == pytest.approx(1)
        assert [row[i] for i in (0, 1, 3, 4)] == pytest.approx([0, 0, 0, 0], abs=1e-9)
