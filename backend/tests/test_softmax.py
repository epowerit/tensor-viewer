import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.models import InputSpec, ProjectDraft, TensorState
from tensorviewer.operations.registry import describe_operation
from tensorviewer.snapshots import json_value, save_snapshot
from tensorviewer.softmax_statistics import CHUNK_SIZE, snapshot_softmax
from tensorviewer.storage import Store
from tensorviewer.worker import execute


def trace_run(expression, shape=(2, 3, 4), dtype="float64", mode="values", directory=None):
    draft = ProjectDraft(
        name="Softmax lesson",
        class_name="Example",
        constructor={},
        capture_mode=mode,
        input=InputSpec(shape=list(shape), dtype=dtype, axis_names=[]),
        code=f"""import torch
from torch import nn
from torch.nn import functional as F
class Example(nn.Module):
    def forward(self, x):
        return {expression}
""",
    )
    trace = execute(draft, snapshot_dir=directory)
    assert trace.error is None
    return draft, trace, trace.operations[-1]


@pytest.mark.parametrize(
    "expression",
    [
        "nn.Softmax(dim=1)(x)",
        "F.softmax(x, dim=-1)",
        "x.softmax(0)",
        "torch.softmax(dim=1, input=x)",
        "F.softmax(x, 1, dtype=torch.float64)",
        "x.transpose(0, 2).softmax(1)",
        "x[0,0,0].softmax(-1)",
    ],
)
def test_recorded_calls_preserve_native_weights_and_explicit_axis(expression):
    _, trace, op = trace_run(expression)
    assert op.lesson.interaction == "normalization"
    x, y = trace.tensors[op.inputs[0]], trace.tensors[op.outputs[0]]
    expected = (
        torch.tensor(x.values, dtype=torch.float64).reshape(x.shape).softmax(op.arguments["dim"])
    )
    torch.testing.assert_close(
        torch.tensor(y.values, dtype=torch.float64).reshape(y.shape), expected
    )
    assert x.shape == y.shape and x.axes == y.axes


def tensors(directory, x, dim, paged=True):
    result = []
    for name, value in [("x", x), ("y", x.softmax(dim))]:
        t = TensorState(
            id=name,
            name=name,
            shape=list(value.shape),
            axes=[],
            dtype=str(value.dtype).split(".")[-1],
            numel=value.numel(),
            strides=list(value.stride()),
            storage_id=name,
            storage_offset=value.storage_offset(),
            contiguous=value.is_contiguous(),
            value_source="paged" if paged else "inline",
            values=[] if paged else [json_value(n) for n in value.flatten()],
        )
        if paged:
            save_snapshot(directory, name, value)
        result.append(t)
    return [result[0]], [result[1]]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("paged", [True, False])
def test_full_denominators_follow_logical_groups_and_native_precision(tmp_path, dtype, axis, paged):
    x = torch.linspace(-4, 5, 2 * 3 * 257, dtype=dtype).reshape(3, 257, 2).permute(2, 0, 1)
    x[0, 0, 100] = -math.inf
    inputs, outputs = tensors(tmp_path, x, axis, paged)
    reference = x.movedim(axis, -1).reshape(-1, x.shape[axis]).double()
    for group in (0, len(reference) - 1):
        stats = snapshot_softmax(tmp_path, {"dim": axis}, inputs, outputs, group)
        values = reference[group]
        maximum = values.max().item()
        total = math.fsum(math.exp(v - maximum) for v in values.tolist())
        assert stats["status"] == "ok"
        assert stats["maximum"] == maximum
        assert stats["denominator"] == pytest.approx(total, rel=1e-14)
        assert stats["masked_count"] == int(torch.isneginf(values).sum())


def test_large_groups_scan_in_bounded_chunks_and_never_use_a_window_total(tmp_path, monkeypatch):
    x = torch.zeros((2, CHUNK_SIZE * 2 + 17, 3), dtype=torch.float64)
    x[1, -1, 2] = 7
    inputs, outputs = tensors(tmp_path, x, 1)
    original = np.asarray
    lengths = []

    def bounded(values, *args, **kwargs):
        lengths.append(len(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr("tensorviewer.softmax_statistics.np.asarray", bounded)
    stats = snapshot_softmax(tmp_path, {"dim": 1}, inputs, outputs, 5)
    assert stats["maximum"] == 7
    assert stats["denominator"] == pytest.approx(1 + (x.shape[1] - 1) * math.exp(-7))
    assert max(lengths) <= CHUNK_SIZE and sum(lengths) == 2 * x.shape[1]


@pytest.mark.parametrize("paged", [True, False])
def test_masks_invalid_scores_extremes_and_scalar_have_explicit_semantics(tmp_path, paged):
    cases = [
        ([0.0, -math.inf, 1.0], "ok", 1),
        ([-math.inf] * 768, "all_masked", 768),
        ([0.0] * 767 + [math.inf], "non_finite", None),
        ([0.0] * 767 + [math.nan], "non_finite", None),
        ([-1.7e308, 1.7e308], "ok", 0),
    ]
    for values, status, masked in cases:
        x = torch.tensor(values, dtype=torch.float64)
        result = snapshot_softmax(tmp_path, {"dim": 0}, *tensors(tmp_path, x, 0, paged), 0)
        assert result["status"] == status and result.get("masked_count") == masked
        if status != "ok":
            assert result.get("maximum") is None and result.get("denominator") is None
        elif len(values) == 2:
            assert result["denominator"] == 1
    x = torch.tensor(42.0, dtype=torch.float64)
    assert snapshot_softmax(tmp_path, {"dim": -1}, *tensors(tmp_path, x, -1, paged), 0) == {
        "status": "ok",
        "maximum": 42.0,
        "denominator": 1.0,
        "masked_count": 0,
    }


def test_metadata_paged_runs_and_unsupported_calls(tmp_path):
    _, trace, op = trace_run("x.softmax(-1)", shape=(128, 16, 768), directory=tmp_path)
    assert (
        op.lesson.interaction == "normalization"
        and trace.tensors[op.inputs[0]].value_source == "paged"
    )
    _, trace, op = trace_run("x.softmax(1)", shape=(1024, 1024, 1024), mode="shapes")
    assert op.lesson.interaction == "normalization" and len(trace.model_dump_json()) < 10000
    assert all(t.value_source == "shape" for t in trace.tensors.values())
    for expression, dtype in [
        ("F.softmax(x)", "float64"),
        ("x.softmax(1,dtype=torch.float64)", "float32"),
        ("x.softmax(1,dtype=torch.float64)", "int64"),
        ("torch.softmax(x,1,out=x)", "float64"),
        ("x[:0].softmax(1)", "float64"),
    ]:
        _, _, op = trace_run(expression, dtype=dtype)
        assert op.lesson.interaction == "inspect"
    _, trace, op = trace_run("x.softmax(1)")
    inputs, outputs = [trace.tensors[op.inputs[0]]], [trace.tensors[op.outputs[0]]]
    for dim in (None, True, 3, -4, 1.5, [1]):
        assert describe_operation("softmax", {"dim": dim}, inputs, outputs).interaction == "inspect"


def saved_run(tmp_path, mode="values"):
    store = Store(tmp_path)
    draft, _, _ = trace_run("x.softmax(1)", shape=(2, 3, 768), mode=mode)
    draft.code = draft.code.replace(
        "return x.softmax(1)",
        "view = x.transpose(0, 2)\n        y = view.softmax(0)\n        x.add_(10000)\n        return y",
    )
    project = store.save_project(draft)
    id = str(uuid4())
    trace = execute(project, snapshot_dir=store.snapshot_dir / id)
    assert trace.error is None
    run = store.save_run(project, trace, id)
    op = next(op for op in trace.operations if op.kind == "softmax")
    return store, run, op, f"/api/v1/runs/{run.id}/operations/{op.id}/softmax"


def test_endpoint_uses_immutable_groups_after_mutation_and_restart(tmp_path, monkeypatch):
    store, run, op, url = saved_run(tmp_path)
    project = store.project(run.project_id)
    project.code = "raise RuntimeError('must not execute')"
    store.save_project(ProjectDraft(**project.model_dump()), project)
    monkeypatch.setattr(
        "tensorviewer.app.run_project", lambda *a, **k: pytest.fail("must not run code")
    )
    client = TestClient(create_app(tmp_path))
    for group in (0, 5, 2):
        response = client.get(url, params={"group": group})
        assert response.status_code == 200 and len(response.content) < 450
        data = response.json()
        assert (
            data["run_id"] == run.id
            and data["operation_id"] == op.id
            and data["tensor_id"] == op.inputs[0]
        )
        assert data["group"] == group and data["count"] == 768
        assert data["maximum"] == (group % 2) * 2304 + (group // 2) * 768 + 767
        assert data["denominator"] == pytest.approx(1 / (1 - math.exp(-1)))
    for value in (-1, 6, "bad", "0.5"):
        assert client.get(url, params={"group": value}).status_code == 422
    assert client.get(url).status_code == 422
    assert client.get(url.replace(op.id, "missing"), params={"group": 0}).status_code == 404
    assert client.get(url.replace(run.id, "missing"), params={"group": 0}).status_code == 404
    assert (
        client.get(url.replace(op.id, run.trace.operations[0].id), params={"group": 0}).status_code
        == 422
    )
    path = store.snapshot_dir / run.id / f"{op.inputs[0]}.npy"
    for invalid in (
        np.zeros((1,)),
        np.zeros((768, 3, 2), dtype="float32"),
        np.asfortranarray(np.zeros((768, 3, 2))),
    ):
        np.save(path, invalid)
        assert client.get(url, params={"group": 0}).status_code == 410
    path.unlink()
    assert client.get(url, params={"group": 0}).status_code == 410
    _, _, _, shape_url = saved_run(tmp_path, mode="shapes")
    assert client.get(shape_url, params={"group": 0}).status_code == 409


def test_fixture_values_are_native_pytorch_outputs():
    cases = json.loads(
        (Path(__file__).parents[2] / "frontend/src/operations/fixtures/softmax.json").read_text()
    )
    for case in cases:
        x = torch.tensor([float(v) for v in case["input"]], dtype=torch.float64).reshape(
            case["shape"]
        )
        # exp rounds its last bit differently on another CPU: match to rounding.
        assert [json_value(v) for v in x.softmax(case["dim"]).flatten()] == pytest.approx(
            case["output"], rel=1e-12, abs=1e-12
        )
