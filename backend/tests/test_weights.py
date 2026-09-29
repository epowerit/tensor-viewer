import io
from collections import OrderedDict

import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.storage import Store
from tensorviewer.weights import (
    MAX_WEIGHT_BYTES,
    import_checkpoint,
    normalize_checkpoint,
    read_state,
)
from tensorviewer.worker import execute

CODE = """from torch import nn
class Example(nn.Module):
    def __init__(self, features=2):
        super().__init__()
        self.linear = nn.Linear(3, features)
    def forward(self, x):
        return self.linear(x)
"""


def state():
    return OrderedDict(
        [
            ("linear.weight", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            ("linear.bias", torch.tensor([10.0, 20.0])),
        ]
    )


def draft(**changes):
    return ProjectDraft(
        **{
            "name": "Loaded weights",
            "class_name": "Example",
            "constructor": {},
            "code": CODE,
            "input": InputSpec(shape=[1, 3], axis_names=[]),
            **changes,
        }
    )


def saved(tmp_path, values=None):
    store = Store(tmp_path / "data")
    source = tmp_path / "source.pt"
    torch.save(state() if values is None else values, source)
    return store, store.import_weights(source, "Known linear weights", "source.pt")


def output(trace):
    assert trace.error is None, trace.error
    return trace.tensors[trace.output_ids[0]].values


def test_checkpoint_drives_real_parameter_and_output_values(tmp_path):
    store, weights = saved(tmp_path)
    project = draft(weights=weights)
    first = execute(project, weights_dir=store.weights_dir)
    assert output(first) == [18, 37]
    params = [t for t in first.tensors.values() if t.role == "parameter"]
    assert any(t.values == [1, 2, 3, 4, 5, 6] for t in params)
    assert first.weight_check.compatible and first.weight_check.tensor_count == 2
    project.input.seed = 103
    assert output(execute(project, weights_dir=store.weights_dir)) == [18, 37]
    assert not torch.equal(torch.tensor(output(execute(draft()))), torch.tensor([18.0, 37.0]))


def test_metadata_check_never_calls_forward_and_shape_run_retains_provenance(tmp_path):
    store, weights = saved(tmp_path)
    project = draft(
        weights=weights,
        code=CODE.replace("return self.linear(x)", "raise RuntimeError('forward was called')"),
    )
    check = execute(project, weights_dir=store.weights_dir, check_weights_only=True)
    assert check.error is None and check.weight_check.compatible
    assert not check.operations and not check.input_ids
    project = draft(weights=weights, capture_mode="shapes", input=InputSpec(shape=[1024, 1024, 3]))
    trace = execute(project, weights_dir=store.weights_dir)
    assert trace.error is None and trace.weight_check.compatible
    assert trace.tensors[trace.output_ids[0]].shape == [1024, 1024, 2]
    assert all(not t.values for t in trace.tensors.values())


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: s.pop("linear.bias"), "Missing weight: linear.bias"),
        (lambda s: s.update(extra=torch.ones(1)), "Unexpected weight: extra"),
        (lambda s: s.update({"linear.weight": torch.ones(3, 3)}), "checkpoint shape"),
        (lambda s: s.update({"linear.weight": s["linear.weight"].double()}), "checkpoint dtype"),
    ],
)
def test_mismatches_never_reach_forward(tmp_path, change, message):
    values = state()
    change(values)
    store, weights = saved(tmp_path, values)
    trace = execute(
        draft(
            weights=weights,
            code=CODE.replace("return self.linear(x)", "raise RuntimeError('forward was called')"),
        ),
        weights_dir=store.weights_dir,
    )
    assert trace.error and message in trace.error.message
    assert not trace.weight_check.compatible and not trace.operations
    assert "forward was called" not in trace.error.message


def test_matching_float64_weights_and_integral_buffers(tmp_path):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3)).double()
    model[0].running_mean.fill_(2)
    model[0].num_batches_tracked.fill_(17)
    store, weights = saved(tmp_path, model.state_dict())
    source = """from torch import nn
class Example(nn.Sequential):
    def __init__(self):
        super().__init__(nn.BatchNorm1d(3))
"""
    trace = execute(
        draft(
            code=source,
            weights=weights,
            input=InputSpec(shape=[2, 3], axis_names=[], dtype="float64"),
        ),
        weights_dir=store.weights_dir,
    )
    actual = output(trace)
    expected = model.eval()(torch.arange(6, dtype=torch.float64).reshape(2, 3)).flatten().tolist()
    assert actual == expected
    loaded = read_state(store.weights_dir / f"{weights.id}.pt", metadata=False)
    assert loaded["0.num_batches_tracked"].item() == 17
    assert loaded._metadata == model.state_dict()._metadata


def test_private_values_survive_forward_mutation_and_repeated_runs(tmp_path):
    store, weights = saved(tmp_path)
    code = CODE.replace(
        "return self.linear(x)", "self.linear.weight.add_(1)\n        return self.linear(x)"
    )
    project = draft(code=code, weights=weights)
    assert output(execute(project, weights_dir=store.weights_dir)) == [21, 40]
    assert output(execute(project, weights_dir=store.weights_dir)) == [21, 40]
    assert output(execute(draft(weights=weights), weights_dir=store.weights_dir)) == [18, 37]


@pytest.mark.parametrize("wrapper", ["state_dict", "model_state_dict"])
def test_common_wrappers_and_noncontiguous_values(tmp_path, wrapper):
    values = state()
    values["linear.weight"] = torch.arange(6.0).reshape(3, 2).T
    store, weights = saved(tmp_path, {wrapper: values, "epoch": 12})
    loaded = read_state(store.weights_dir / f"{weights.id}.pt", metadata=False)
    assert loaded["linear.weight"].is_contiguous()
    torch.testing.assert_close(loaded["linear.weight"], values["linear.weight"])


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"weight": "text"},
        {"state_dict": state(), "model_state_dict": state()},
        {"weight": torch.ones(2, dtype=torch.complex64)},
        {"weight": torch.ones(2).to_sparse()},
        {"weight": torch.ones([1] * 9)},
        {"weight": torch.ones(1).expand(8_388_609)},
        torch.nn.Linear(3, 2),
        {"": torch.ones(1)},
    ],
)
def test_rejected_files_leave_no_library_asset(tmp_path, bad):
    source = tmp_path / "bad.pt"
    torch.save(bad, source)
    store = Store(tmp_path / "data")
    with pytest.raises(ValueError):
        store.import_weights(source, "Invalid", "bad.pt")
    assert store.weights() == []
    assert list(store.weights_dir.iterdir()) == []


class ForbiddenObject:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (eval, (f"__import__('pathlib').Path({str(self.path)!r}).write_text('unsafe')",))


def test_restricted_unpickler_never_executes_objects(tmp_path):
    source, destination = tmp_path / "bad.pt", tmp_path / "output.pt"
    sentinel = tmp_path / "executed"
    torch.save({"weight": ForbiddenObject(sentinel)}, source)
    with pytest.raises(ValueError, match="tensor-only"):
        normalize_checkpoint(source, destination)
    assert not sentinel.exists() and not destination.exists()


def test_old_format_corruption_and_upload_size_checks(tmp_path):
    source = tmp_path / "old.pt"
    torch.save(state(), source, _use_new_zipfile_serialization=False)
    with pytest.raises(ValueError, match="Legacy pickle"):
        normalize_checkpoint(source, tmp_path / "output.pt")
    source.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        import_checkpoint(source, tmp_path / "output.pt")
    client = TestClient(create_app(tmp_path / "api"))
    assert (
        client.post(
            "/api/v1/weights/upload?name=test&file_name=test.pt",
            content=b"x",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(MAX_WEIGHT_BYTES + 1),
            },
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/v1/weights/upload?name=test&file_name=test.txt",
            content=b"x",
            headers={"Content-Type": "application/octet-stream"},
        ).status_code
        == 422
    )


def upload(client):
    stream = io.BytesIO()
    torch.save(state(), stream)
    result = client.post(
        "/api/v1/weights/upload?name=Linear%20checkpoint&file_name=linear.pt",
        content=stream.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert result.status_code == 201, result.text
    return result.json()


def test_api_library_check_runs_history_and_restart(tmp_path):
    client = TestClient(create_app(tmp_path))
    weights = upload(client)
    data = draft().model_dump()
    data["weights"] = weights
    check = client.post("/api/v1/weights/check", json=data)
    assert check.status_code == 200 and check.json()["compatible"], check.text
    project = client.post("/api/v1/projects", json=data).json()
    result = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    assert result["trace"]["error"] is None
    assert result["project"]["weights"]["sha256"] == weights["sha256"]
    data["weights"] = None
    assert client.put(f"/api/v1/projects/{project['id']}", json=data).status_code == 200
    restarted = TestClient(create_app(tmp_path))
    assert restarted.get("/api/v1/weights").json() == [weights]
    assert restarted.get(f"/api/v1/runs/{result['id']}").json()["project"]["weights"] == weights
    assert client.post("/api/v1/weights/check", json=data).status_code == 422


def test_blueprint_checkpoint_uses_generated_parameter_names(tmp_path):
    from tensorviewer.composer import canonical_project
    from tensorviewer.models import Blueprint, ComponentSpec

    project = canonical_project(
        draft(
            blueprint=Blueprint(
                has_input=True,
                components=[ComponentSpec(id="linear", kind="linear", parameters={"features": 2})],
            )
        )
    )
    namespace = {}
    exec(project.code, namespace)
    model = namespace["ComposedModel"]()
    for parameter in model.parameters():
        parameter.data.fill_(1)
    store, weights = saved(tmp_path, model.state_dict())
    project.weights = weights
    trace = execute(project, weights_dir=store.weights_dir)
    assert output(trace) == [4, 4]


def test_tampered_missing_or_forged_weights_fail_before_source(tmp_path):
    store, weights = saved(tmp_path)
    project = draft(weights=weights, code="raise RuntimeError('source executed')")
    path = store.weights_dir / f"{weights.id}.pt"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    for shapes in [False, True]:
        project.capture_mode = "shapes" if shapes else "values"
        trace = execute(project, weights_dir=store.weights_dir)
        assert "has changed" in trace.error.message and "source executed" not in trace.error.message
    path.unlink()
    trace = execute(project, weights_dir=store.weights_dir)
    assert "missing" in trace.error.message
    client = TestClient(create_app(tmp_path / "data"))
    data = draft(weights=weights).model_dump()
    data["weights"]["sha256"] = "f" * 64
    assert client.post("/api/v1/projects", json=data).status_code == 422
    data["weights"] = weights.model_dump()
    assert client.post("/api/v1/projects", json=data).status_code == 409
