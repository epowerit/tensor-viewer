from uuid import uuid4

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.models import InputSpec, Operation, ProjectDraft, TensorState
from tensorviewer.snapshots import json_value, save_snapshot
from tensorviewer.statistics import CHUNK_SIZE, snapshot_statistics
from tensorviewer.storage import Store
from tensorviewer.worker import execute


def snapshot(tmp_path, x, paged=True):
    tensor = TensorState(
        id="tensor-1",
        name="x",
        shape=list(x.shape),
        axes=[],
        dtype=str(x.dtype).split(".")[-1],
        strides=list(x.stride()),
        storage_id="storage-1",
        storage_offset=x.storage_offset(),
        contiguous=x.is_contiguous(),
        numel=x.numel(),
        values=[] if paged else [json_value(v) for v in x.flatten()],
        value_source="paged" if paged else "inline",
    )
    if paged:
        save_snapshot(tmp_path, tensor.id, x)
    return tensor


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("paged", [False, True])
def test_statistics_match_full_population_formula_and_pytorch(tmp_path, dtype, paged):
    x = torch.linspace(-7, 11, 2 * 768, dtype=dtype).reshape(2, 768)
    tensor = snapshot(tmp_path, x, paged)
    result = snapshot_statistics(tmp_path, tensor, 768, 768, 0.25)
    group = x[1].double()
    assert result.status == "ok"
    assert result.mean == pytest.approx(group.mean().item())
    assert result.variance == pytest.approx(group.var(unbiased=False).item())
    manual = (group - result.mean) / result.denominator
    # Float64 reference isolates population/epsilon semantics from native low-precision rounding.
    torch.testing.assert_close(manual, torch.nn.functional.layer_norm(group, (768,), eps=0.25))


def test_large_offset_and_multichunk_group_are_not_partial_statistics(tmp_path, monkeypatch):
    count = CHUNK_SIZE * 3 + 7
    x = (torch.arange(count + 12, dtype=torch.float64) % 31 + 1e12).reshape(-1)
    tensor = snapshot(tmp_path, x)
    original = np.asarray
    sizes = []

    def bounded(values, *args, **kwargs):
        sizes.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr("tensorviewer.statistics.np.asarray", bounded)
    result = snapshot_statistics(tmp_path, tensor, 12, count, 1e-5)
    group = x[12:].numpy()
    assert result.mean == pytest.approx(group.mean(), abs=0.001, rel=0)
    assert result.variance == pytest.approx(group.var(), rel=1e-6)
    assert max(sizes) <= CHUNK_SIZE
    assert sum(sizes) == 2 * count


@pytest.mark.parametrize(
    "value,status",
    [(float("nan"), "non_finite"), (float("inf"), "non_finite"), (1e308, "overflow")],
)
def test_invalid_value_outside_visible_window_does_not_produce_partial_mean(
    tmp_path, value, status
):
    x = torch.zeros(768, dtype=torch.float64)
    x[600] = value
    result = snapshot_statistics(tmp_path, snapshot(tmp_path, x), 0, 768, 1e-5)
    assert result.model_dump() == {
        "status": status,
        "mean": None,
        "variance": None,
        "denominator": None,
    }


@pytest.mark.parametrize("value", [7, 0.1, 1e300])
def test_constant_group_and_zero_epsilon(tmp_path, value):
    x = torch.ones(768, dtype=torch.float64) * value
    tensor = snapshot(tmp_path, x)
    zero = snapshot_statistics(tmp_path, tensor, 0, 768, 0)
    assert zero.status == "ok" and zero.mean == value and zero.variance == zero.denominator == 0
    assert snapshot_statistics(tmp_path, tensor, 0, 768, 0.25).denominator == 0.5


def saved_run(tmp_path, mode="values", code=None):
    store = Store(tmp_path)
    project = store.save_project(
        ProjectDraft(
            name="Large normalization",
            class_name="Example",
            constructor={},
            capture_mode=mode,
            input=InputSpec(shape=[2, 3, 768], axis_names=[]),
            code=code
            or """import torch
from torch import nn
class Example(nn.Module):
    def forward(self, x):
        view = x.transpose(0, 1)
        y = torch.nn.functional.layer_norm(view, (2, 768), eps=0.25)
        x.add_(10000)
        return y
""",
        )
    )
    run_id = str(uuid4())
    trace = execute(project, snapshot_dir=store.snapshot_dir / run_id)
    assert trace.error is None
    run = store.save_run(project, trace, run_id)
    op = next(op for op in trace.operations if op.kind == "layer_norm")
    url = f"/api/v1/runs/{run.id}/operations/{op.id}/normalization"
    return store, run, op, url


def test_endpoint_uses_immutable_logical_multi_axis_groups_after_restart(tmp_path, monkeypatch):
    store, run, op, url = saved_run(tmp_path)
    tensor = run.trace.tensors[op.inputs[0]]
    assert not tensor.contiguous and tensor.value_source == "paged"
    project = store.project(run.project_id)
    project.code = "raise RuntimeError('must not execute')"
    store.save_project(ProjectDraft(**project.model_dump()), project)

    def forbidden(*args, **kwargs):
        pytest.fail("Statistics must never execute project code")

    monkeypatch.setattr("tensorviewer.app.run_project", forbidden)
    client = TestClient(create_app(tmp_path))
    original = torch.arange(2 * 3 * 768, dtype=torch.float64).reshape(2, 3, 768).transpose(0, 1)
    for group in (0, 2, 1):
        response = client.get(url, params={"group": group})
        assert response.status_code == 200 and len(response.content) < 400
        result = response.json()
        assert result["operation_id"] == op.id and result["tensor_id"] == tensor.id
        assert (
            result["group"] == group and result["start"] == group * 1536 and result["count"] == 1536
        )
        assert result["status"] == "ok"
        assert result["mean"] == pytest.approx(original[group].mean().item())
        assert result["variance"] == pytest.approx(original[group].var(unbiased=False).item())
        assert result["denominator"] == pytest.approx((result["variance"] + 0.25) ** 0.5)
    for value in (-1, 3, "abc", "0.5", "1e999"):
        assert client.get(url, params={"group": value}).status_code == 422
    assert client.get(url).status_code == 422
    assert client.get(url.replace(op.id, "missing"), params={"group": 0}).status_code == 404
    assert client.get(url.replace(run.id, "missing"), params={"group": 0}).status_code == 404
    other = run.trace.operations[0]
    assert client.get(url.replace(op.id, other.id), params={"group": 0}).status_code == 422


@pytest.mark.parametrize("corruption", ["missing", "truncated", "shape", "dtype", "order"])
def test_invalid_snapshots_return_recoverable_error(tmp_path, corruption):
    store, run, op, url = saved_run(tmp_path)
    tensor = run.trace.tensors[op.inputs[0]]
    path = store.snapshot_dir / run.id / f"{tensor.id}.npy"
    if corruption == "missing":
        path.unlink()
    elif corruption == "truncated":
        path.write_bytes(b"invalid")
    else:
        shape = [1] if corruption == "shape" else tensor.shape
        data = np.zeros(
            shape,
            dtype="float64" if corruption == "dtype" else "float32",
            order="F" if corruption == "order" else "C",
        )
        np.save(path, data)
    response = TestClient(create_app(tmp_path)).get(url, params={"group": 0})
    assert response.status_code == 410
    assert "Run the project again" in response.json()["detail"]


def test_shape_only_and_failed_or_mutated_operations_have_no_statistics(tmp_path):
    store, run, op, url = saved_run(tmp_path, mode="shapes")
    client = TestClient(create_app(tmp_path))
    assert client.get(url, params={"group": 0}).status_code == 409
    for changes in (
        {"status": "error"},
        {"mutations": [{"before": op.inputs[0], "after": op.outputs[0], "kind": "write"}]},
    ):
        trace = run.trace.model_copy(deep=True)
        trace.operations[1] = Operation.model_validate({**op.model_dump(), **changes})
        revised = store.save_run(store.project(run.project_id), trace)
        assert client.get(url.replace(run.id, revised.id), params={"group": 0}).status_code == 422


def test_invalid_range_and_incomplete_inline_snapshot_are_rejected(tmp_path):
    tensor = snapshot(tmp_path, torch.zeros(768), paged=False)
    for start, count in [(-1, 768), (0, 0), (768, 1), (1, 768)]:
        with pytest.raises(ValueError):
            snapshot_statistics(tmp_path, tensor, start, count, 1e-5)
    tensor.values.pop()
    with pytest.raises(ValueError, match="Incomplete"):
        snapshot_statistics(tmp_path, tensor, 0, 768, 1e-5)
