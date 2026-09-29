import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tensorviewer.app import create_app
from tensorviewer.models import ForwardInput, InputSpec, ProjectDraft
from tensorviewer.worker import execute

CODE = """import torch
from torch import nn
class CrossAttention(nn.Module):
    def forward(self, query, key, value, *, mask):
        kt = key.transpose(-2, -1)
        scores = query @ kt
        scores = scores / query.shape[-1] ** 0.5
        scores = scores + mask
        weights = scores.softmax(dim=-1)
        return weights @ value
"""


def spec(shape, generator="arange", **kwargs):
    return InputSpec(shape=shape, axis_names=[], generator=generator, **kwargs)


def example(**changes):
    return ProjectDraft(
        **{
            "name": "Cross attention",
            "code": CODE,
            "class_name": "CrossAttention",
            "constructor": {},
            "input_name": "query",
            "input": spec([2, 3, 4]),
            "additional_inputs": [
                ForwardInput(name="key", input=spec([2, 5, 4])),
                ForwardInput(name="value", input=spec([2, 5, 6])),
                ForwardInput(name="mask", binding="keyword", input=spec([3, 5], "zeros")),
            ],
            **changes,
        }
    )


def test_cross_attention_values_sources_and_keyword_module_boundaries():
    trace = execute(example())
    assert trace.error is None
    tensors = [trace.tensors[i] for i in trace.input_ids]
    assert [t.name for t in tensors] == ["query", "key", "value", "mask"]
    assert all(t.role == "input" for t in tensors)
    q, k, v, mask = [torch.tensor(t.values).reshape(t.shape) for t in tensors]
    expected = ((q @ k.transpose(-2, -1) / 2 + mask).softmax(dim=-1) @ v).flatten()
    torch.testing.assert_close(torch.tensor(trace.tensors[trace.output_ids[0]].values), expected)
    assert trace.module_calls[0].inputs == trace.input_ids
    assert {i for op in trace.operations for i in op.inputs}.issuperset(trace.input_ids)


def test_shapes_keep_all_inputs_and_mixed_dtypes_without_values():
    draft = example(capture_mode="shapes")
    draft.additional_inputs[-1].input = spec([3, 5], "zeros", dtype="int64")
    trace = execute(draft)
    assert trace.error is None
    assert trace.tensors[trace.input_ids[-1]].dtype == "int64"
    assert trace.tensors[trace.output_ids[0]].shape == [2, 3, 6]
    assert all(t.values == [] and t.value_source == "shape" for t in trace.tensors.values())


def test_each_input_has_independent_seed_and_private_storage():
    draft = example()
    draft.input = spec([2, 3, 4], "random", random_stream="input", seed=11)
    draft.additional_inputs[0].input = spec([2, 5, 4], "random", random_stream="input", seed=12)
    first, second = execute(draft), execute(draft)
    for i, seed in enumerate([11, 12]):
        tensor = first.tensors[first.input_ids[i]]
        expected = torch.randn(tensor.shape, generator=torch.Generator().manual_seed(seed))
        assert tensor.values == expected.flatten().tolist()
        assert tensor.values == second.tensors[second.input_ids[i]].values
    assert len({first.tensors[i].storage_id for i in first.input_ids}) == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"input_name": "class"},
        {"input_name": "a.b"},
        {"input_binding": "keyword"},
        {"additional_inputs": [{"name": "query", "input": spec([1])}]},
        {"additional_inputs": [{"name": "for", "input": spec([1])}]},
        {"additional_inputs": [{"name": f"x{i}", "input": spec([1])} for i in range(8)]},
        {"additional_inputs": [{"name": "large", "input": spec([8_388_609])}]},
        {"additional_inputs": [{"name": f"x{i}", "input": spec([8_000_000])} for i in range(4)]},
        {"blueprint": {"has_input": True, "components": []}},
    ],
)
def test_invalid_call_configurations_are_rejected_before_execution(changes):
    with pytest.raises(ValidationError):
        example(**changes)


def test_signature_failure_names_the_missing_argument():
    trace = execute(example(additional_inputs=[]))
    assert trace.error.type == "ValueError"
    assert "Forward inputs do not match CrossAttention.forward" in trace.error.message
    assert "key" in trace.error.message
    assert not trace.operations


def test_single_keyword_only_input_and_integer_auxiliary():
    code = """from torch import nn
class Example(nn.Module):
    def forward(self, *, image, indices):
        return image[:, indices]
"""
    draft = ProjectDraft(
        name="Keyword",
        code=code,
        class_name="Example",
        constructor={},
        input_name="image",
        input_binding="keyword",
        input=spec([2, 4], dtype="float64"),
        additional_inputs=[
            ForwardInput(name="indices", binding="keyword", input=spec([2], dtype="int64"))
        ],
    )
    trace = execute(draft)
    assert trace.error is None
    assert trace.tensors[trace.output_ids[0]].values == [0, 1, 4, 5]
    assert trace.tensors[trace.output_ids[0]].dtype == "float64"


def test_legacy_single_input_contract_and_partial_failure():
    code = """from torch import nn
class Example(nn.Module):
    def forward(self, x):
        return x + 1
"""
    legacy = ProjectDraft(
        name="Old", code=code, class_name="Example", constructor={}, input=spec([2])
    )
    trace = execute(
        ProjectDraft.model_validate(
            legacy.model_dump(exclude={"input_name", "input_binding", "additional_inputs"})
        )
    )
    assert trace.error is None
    assert trace.tensors[trace.input_ids[0]].name == "x"
    assert trace.tensors[trace.output_ids[0]].values == [1, 2]
    draft = example(code=CODE.replace("return weights @ value", "return weights.reshape(999)"))
    trace = execute(draft)
    assert trace.error is not None and trace.error.line
    assert len(trace.input_ids) == 4 and trace.operations[-1].status == "error"
    assert len(trace.operations) > 3


def upload(client, array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    result = client.post(
        "/api/v1/input-fixtures/upload?name=Mask&file_name=mask.npy",
        content=buffer.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert result.status_code == 201, result.text
    return InputSpec.model_validate(result.json()["input"])


def test_uploaded_keyword_input_runs_persists_and_is_immutable(tmp_path):
    client = TestClient(create_app(tmp_path))
    mask = np.zeros([3, 5], dtype=np.float32)
    mask[:, -1] = -1000
    draft = example()
    draft.additional_inputs[-1].input = upload(client, mask)
    project = client.post("/api/v1/projects", json=draft.model_dump()).json()
    result = client.post(f"/api/v1/projects/{project['id']}/runs")
    assert result.status_code == 201, result.text
    saved = result.json()
    trace = saved["trace"]
    assert trace["error"] is None
    assert trace["tensors"][trace["input_ids"][-1]]["values"] == mask.flatten().tolist()
    draft.additional_inputs[-1].input = spec([3, 5], "ones")
    assert (
        client.put(f"/api/v1/projects/{project['id']}", json=draft.model_dump()).status_code == 200
    )
    reopened = TestClient(create_app(tmp_path)).get(f"/api/v1/runs/{saved['id']}").json()
    assert reopened["project"]["additional_inputs"][-1]["input"]["generator"] == "uploaded"
    assert reopened["project"]["input_name"] == "query"


def test_extra_upload_is_validated_before_source_runs(tmp_path):
    client = TestClient(create_app(tmp_path))
    draft = example()
    draft.additional_inputs[-1].input = upload(client, np.zeros([3, 5], dtype=np.float32))
    asset = draft.additional_inputs[-1].input.uploaded
    data = draft.model_dump()
    data["additional_inputs"][-1]["input"]["uploaded"]["sha256"] = "a" * 64
    assert client.post("/api/v1/projects", json=data).status_code == 422
    project = client.post("/api/v1/projects", json=draft.model_dump()).json()
    path = tmp_path / "inputs" / f"{asset.id}.npy"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    draft.code = "raise RuntimeError('SOURCE EXECUTED')"
    trace = execute(draft, input_dir=tmp_path / "inputs")
    assert trace.error and "SOURCE EXECUTED" not in trace.error.message
    path.unlink()
    assert client.post(f"/api/v1/projects/{project['id']}/runs").status_code == 409


def test_large_shapes_are_independent_of_numeric_budgets():
    draft = example(
        capture_mode="shapes",
        code=CODE.replace(
            "kt = key.transpose(-2, -1)", "return query\n        kt = key.transpose(-2, -1)"
        ),
    )
    draft.input = spec([1024, 1024, 1024])
    trace = execute(draft)
    assert trace.error is None
    assert len(trace.input_ids) == 4
    assert trace.tensors[trace.output_ids[0]].shape == [1024, 1024, 1024]
