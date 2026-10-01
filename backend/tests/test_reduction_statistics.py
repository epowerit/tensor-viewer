import math
from uuid import uuid4

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.models import InputSpec, Operation, ProjectDraft, TensorState
from tensorviewer.reduction_statistics import CHUNK_SIZE, group_indices, snapshot_reduction
from tensorviewer.snapshots import json_value, save_snapshot
from tensorviewer.storage import Store
from tensorviewer.worker import execute


def snapshot(tmp_path, x, id="x", paged=True):
    tensor = TensorState(
        id=id,
        name=id,
        shape=list(x.shape),
        axes=[],
        dtype=str(x.dtype).split(".")[-1],
        strides=list(x.stride()),
        storage_id=id,
        storage_offset=x.storage_offset(),
        contiguous=x.is_contiguous(),
        numel=x.numel(),
        values=[] if paged else [json_value(v) for v in x.flatten()],
        value_source="paged" if paged else "inline",
    )
    if paged:
        save_snapshot(tmp_path, tensor.id, x)
    return tensor


def calculate(tmp_path, x, kind="mean", args=None, output=0, paged=True):
    args = args or {}
    y = getattr(x, kind)(**args)
    tensors = [snapshot(tmp_path, x, paged=paged)], [snapshot(tmp_path, y, "y", paged=False)]
    return snapshot_reduction(tmp_path, kind, args, *tensors, output)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["sum", "mean"])
@pytest.mark.parametrize("paged", [False, True])
def test_full_nonadjacent_groups_use_logical_order_and_native_input_precision(
    tmp_path, dtype, kind, paged
):
    original = torch.linspace(-3, 7, 2 * 3 * 257, dtype=dtype).reshape(3, 257, 2)
    x = original.permute(2, 0, 1)
    assert not x.is_contiguous()
    for output in (0, 2):
        result = calculate(tmp_path, x, kind, {"dim": (2, 0), "keepdim": True}, output, paged)
        full = x[:, output, :].double().flatten().tolist()
        assert result["status"] == "ok"
        assert result["sum"] == pytest.approx(math.fsum(full), abs=1e-12)
        expected = math.fsum(full) / len(full) if kind == "mean" else math.fsum(full)
        assert result["result"] == pytest.approx(expected, abs=1e-12)


def test_indices_and_numeric_arrays_are_bounded_across_multiple_chunks(tmp_path, monkeypatch):
    x = torch.arange(3 * 2 * (CHUNK_SIZE + 7), dtype=torch.float64).reshape(3, 2, -1)
    tensor, output = snapshot(tmp_path, x), snapshot(tmp_path, x.mean((0, 2)), "y", False)
    original = np.asarray
    sizes = []

    def bounded(values, *args, **kwargs):
        sizes.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr("tensorviewer.reduction_statistics.np.asarray", bounded)
    result = snapshot_reduction(tmp_path, "mean", {"dim": [0, 2]}, [tensor], [output], 1)
    assert result["result"] == x[:, 1, :].mean().item()
    assert max(sizes) <= CHUNK_SIZE
    assert sum(sizes) == 2 * 3 * (CHUNK_SIZE + 7)
    # Independent logical-index oracle, including chunks that cross leading slices.
    indices = list(group_indices(x.shape, [0, 2], [2], False, 1))
    assert max(len(part) for part in indices) <= CHUNK_SIZE
    expected = torch.arange(x.numel()).reshape(x.shape)[:, 1, :].reshape(-1).numpy()
    np.testing.assert_array_equal(np.concatenate(indices), expected)


@pytest.mark.parametrize("dim", [None, [], (), 0, -1])
def test_scalar_and_all_axis_groups(tmp_path, dim):
    result = calculate(tmp_path, torch.tensor(5.0), args={"dim": dim, "keepdim": True})
    assert result == {"status": "ok", "sum": 5.0, "result": 5.0}
    if dim is None or isinstance(dim, (list, tuple)):
        x = torch.arange(768, dtype=torch.float64).reshape(2, 3, 128)
        assert calculate(tmp_path, x, args={"dim": dim})["result"] == 383.5


@pytest.mark.parametrize("paged", [False, True])
def test_integer_totals_preserve_precision_and_do_not_wrap_at_int64(tmp_path, paged):
    for value in (9007199254740993, -(2**63), 2**63 - 1):
        x = torch.full((2, 768), value, dtype=torch.int64)
        result = calculate(tmp_path, x, "sum", {"dim": 1}, 1, paged)
        assert result == {"status": "ok", "sum": str(value * 768), "result": str(value * 768)}
        if abs(value * 768) > 2**63:
            assert int(result["result"]) != x[1].sum().item()
    x = (torch.arange(768).reshape(2, 384) % 3) == 0
    assert calculate(tmp_path, x, "sum", {"dim": 1}, 1, paged)["result"] == 128
    x = torch.ones(768, dtype=torch.int32)
    assert calculate(tmp_path, x, "sum", paged=paged)["result"] == 768


def test_finite_mean_with_overflowing_total_and_cancellation(tmp_path):
    x = torch.full((768,), 1e308, dtype=torch.float64)
    result = calculate(tmp_path, x)
    assert result["status"] == "ok" and result["sum"] is None
    assert result["result"] == pytest.approx(1e308)
    assert calculate(tmp_path, x, "sum")["status"] == "overflow"
    x = torch.tensor([1e16, 1, -1e16], dtype=torch.float64).repeat(257)
    assert calculate(tmp_path, x, "sum")["result"] == 257
    assert calculate(tmp_path, x)["result"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_outside_window_never_return_partial_statistics(tmp_path, value):
    x = torch.ones((2, 768), dtype=torch.float64)
    x[1, 600] = value
    assert calculate(tmp_path, x, args={"dim": 1}, output=1) == {"status": "non_finite"}
    assert calculate(tmp_path, x, args={"dim": 1}, output=0)["result"] == 1


def saved_run(tmp_path, mode="values", code=None):
    store = Store(tmp_path)
    project = store.save_project(
        ProjectDraft(
            name="Complete reductions",
            class_name="Example",
            constructor={},
            capture_mode=mode,
            input=InputSpec(shape=[2, 3, 768], axis_names=[]),
            code=code
            or """import torch
from torch import nn
class Example(nn.Module):
    def forward(self, x):
        view = x.transpose(0, 2)
        y = view.mean((0, 2), keepdim=True)
        x.add_(10000)
        return y
""",
        )
    )
    run_id = str(uuid4())
    trace = execute(project, snapshot_dir=store.snapshot_dir / run_id)
    assert trace.error is None
    run = store.save_run(project, trace, run_id)
    op = next(op for op in trace.operations if op.kind in {"mean", "sum"})
    return store, run, op, f"/api/v1/runs/{run.id}/operations/{op.id}/reduction"


def test_endpoint_uses_captured_noncontiguous_groups_after_mutation_and_restart(
    tmp_path, monkeypatch
):
    store, run, op, url = saved_run(tmp_path)
    project = store.project(run.project_id)
    project.code = "raise RuntimeError('must not execute')"
    store.save_project(ProjectDraft(**project.model_dump()), project)

    def forbidden(*args, **kwargs):
        pytest.fail("A summary must not execute user code")

    monkeypatch.setattr("tensorviewer.app.run_project", forbidden)
    client = TestClient(create_app(tmp_path))
    original = torch.arange(2 * 3 * 768, dtype=torch.float64).reshape(2, 3, 768)
    for index in (0, 2, 1):
        response = client.get(url, params={"output_index": index})
        assert response.status_code == 200 and len(response.content) < 400
        stats = response.json()
        assert stats["run_id"] == run.id and stats["operation_id"] == op.id
        assert stats["tensor_id"] == op.inputs[0] and stats["output_index"] == index
        assert stats["status"] == "ok" and stats["count"] == 1536
        assert stats["sum"] == original[:, index, :].sum().item()
        assert stats["result"] == original[:, index, :].mean().item()
    for index in (-1, 3, "abc", "0.5", "1e999"):
        assert client.get(url, params={"output_index": index}).status_code == 422
    assert client.get(url).status_code == 422
    for old in (op.id, run.id):
        assert (
            client.get(url.replace(old, "missing"), params={"output_index": 0}).status_code == 404
        )
    assert (
        client.get(
            url.replace(op.id, run.trace.operations[0].id), params={"output_index": 0}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("corruption", ["missing", "truncated", "shape", "dtype", "order"])
def test_invalid_snapshots_are_recoverable_errors(tmp_path, corruption):
    store, run, op, url = saved_run(tmp_path)
    tensor = run.trace.tensors[op.inputs[0]]
    path = store.snapshot_dir / run.id / f"{tensor.id}.npy"
    if corruption == "missing":
        path.unlink()
    elif corruption == "truncated":
        path.write_bytes(b"invalid")
    else:
        np.save(
            path,
            np.zeros(
                [1] if corruption == "shape" else tensor.shape,
                dtype="float64" if corruption == "dtype" else "float32",
                order="F" if corruption == "order" else "C",
            ),
        )
    response = TestClient(create_app(tmp_path)).get(url, params={"output_index": 0})
    assert response.status_code == 410
    assert "Run the project again" in response.json()["detail"]


def test_shape_runs_failed_operations_and_casts_have_explicit_responses(tmp_path):
    store, run, op, url = saved_run(tmp_path, mode="shapes")
    client = TestClient(create_app(tmp_path))
    assert client.get(url, params={"output_index": 0}).status_code == 409
    for changes in [
        {"status": "error"},
        {"arguments": {"dim": [0, 0]}},
        {"inputs": ["missing"]},
        {"mutations": [{"before": op.inputs[0], "after": op.outputs[0], "kind": "write"}]},
    ]:
        trace = run.trace.model_copy(deep=True)
        trace.operations[1] = Operation.model_validate({**op.model_dump(), **changes})
        revised = store.save_run(store.project(run.project_id), trace)
        assert (
            client.get(url.replace(run.id, revised.id), params={"output_index": 0}).status_code
            == 422
        )
    _, _, _, url = saved_run(
        tmp_path,
        code="""import torch
from torch import nn
class Example(nn.Module):
    def forward(self, x):
        return x.mean(1, dtype=torch.float64)
""",
    )
    assert client.get(url, params={"output_index": 0}).status_code == 422


def test_invalid_inline_values_and_allocation_bounds(tmp_path):
    x = snapshot(tmp_path, torch.ones(768), paged=False)
    y = snapshot(tmp_path, torch.tensor(1.0), "y", False)
    for index in (-1, 1, True):
        with pytest.raises(ValueError):
            snapshot_reduction(tmp_path, "mean", {}, [x], [y], index)
    x.values[-1] = "corrupt"
    with pytest.raises(ValueError):
        snapshot_reduction(tmp_path, "mean", {}, [x], [y], 0)
    x.values.pop()
    with pytest.raises(ValueError, match="Incomplete"):
        snapshot_reduction(tmp_path, "mean", {}, [x], [y], 0)
    x.shape, x.numel = [8_388_609], 8_388_609
    with pytest.raises(ValueError, match="Invalid numeric snapshot"):
        snapshot_reduction(tmp_path, "mean", {}, [x], [y], 0)
    x = snapshot(tmp_path, torch.ones(768, dtype=torch.int64), paged=False)
    y.dtype = "int64"
    for value in (1.5, 2**64, "1.5", "nan", float(2**53)):
        x.values[-1] = value
        with pytest.raises(ValueError):
            snapshot_reduction(tmp_path, "sum", {}, [x], [y], 0)
