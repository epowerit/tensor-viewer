import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.input_files import MAX_UPLOAD_BYTES


def npy(data, **kwargs):
    stream = io.BytesIO()
    np.save(stream, data, **kwargs)
    return stream.getvalue()


def upload(client, data, **params):
    return client.post(
        "/api/v1/input-fixtures/upload",
        params={"name": "Sample input", "file_name": "sample.npy", **params},
        content=data,
        headers={"Content-Type": "application/octet-stream"},
    )


def project(client, fixture, code=None):
    response = client.post(
        "/api/v1/projects",
        json={
            "name": "Uploaded study",
            "class_name": "Example",
            "constructor": {},
            "input": fixture["input"],
            "capture_mode": fixture["capture_mode"],
            "code": code
            or "from torch import nn\nclass Example(nn.Module):\n def forward(self,x):\n  x.add_(2)\n  return x.transpose(-1,-2)",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def values(client, run, tensor_id, indices):
    response = client.get(
        f"/api/v1/runs/{run['id']}/tensors/{tensor_id}/values",
        params={"indices": ",".join(map(str, indices))},
    )
    assert response.status_code == 200, response.text
    return response.json()["values"]


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_uploaded_values_survive_mutation_restart_and_future_uploads(tmp_path, dtype):
    client = TestClient(create_app(tmp_path))
    original = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    response = upload(client, npy(original))
    assert response.status_code == 201, response.text
    fixture = response.json()
    assert fixture["input"]["shape"] == [2, 3, 4]
    assert fixture["input"]["dtype"] == dtype
    assert fixture["input"]["axis_names"] == []  # Never guess axis meanings from sizes.
    assert len(response.content) < 1500
    saved = project(client, fixture)
    for _ in range(2):
        run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
        assert run["trace"]["error"] is None, run["trace"]["error"]
        assert values(client, run, run["trace"]["input_ids"][0], [0, 7, 23]) == [0, 7, 23]
        expected = (original + 2).swapaxes(-1, -2).reshape(-1).tolist()
        assert values(client, run, run["trace"]["output_ids"][0], list(range(24))) == expected
    newer = upload(client, npy(original + 100)).json()
    assert newer["input"]["uploaded"]["id"] != fixture["input"]["uploaded"]["id"]
    restarted = TestClient(create_app(tmp_path))
    assert restarted.get("/api/v1/input-fixtures").json() == [newer, fixture]
    old = restarted.get(f"/api/v1/runs/{run['id']}").json()
    assert old["project"]["input"] == fixture["input"]
    assert values(restarted, old, old["trace"]["input_ids"][0], [23]) == [23]


def test_fortran_endian_and_exact_int64_values(tmp_path):
    client = TestClient(create_app(tmp_path))
    for data in [
        np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)),
        np.arange(12, dtype=">f4").reshape(3, 4),
        np.array([[np.iinfo(np.int64).min, np.iinfo(np.int64).max]], dtype=np.int64),
        np.array([[np.nan, np.inf, -np.inf]], dtype=np.float64),
    ]:
        fixture = upload(client, npy(data)).json()
        saved = project(
            client,
            fixture,
            "from torch import nn\nclass Example(nn.Module):\n def forward(self,x):\n  return x.clone()",
        )
        run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
        assert run["trace"]["error"] is None
        captured = values(client, run, run["trace"]["input_ids"][0], list(range(data.size)))
        if data.dtype == np.int64:
            assert captured == [str(np.iinfo(np.int64).min), str(np.iinfo(np.int64).max)]
        elif np.isnan(data).any():
            assert captured == ["nan", "inf", "-inf"]
        else:
            assert captured == data.reshape(-1).tolist()
        assert run["trace"]["tensors"][run["trace"]["input_ids"][0]]["contiguous"]


def test_paged_input_and_shape_only_execution(tmp_path):
    client = TestClient(create_app(tmp_path))
    data = np.arange(128 * 32 * 32, dtype=np.float32).reshape(128, 32, 32)
    fixture = upload(client, npy(data)).json()
    saved = project(client, fixture)
    run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
    before = run["trace"]["input_ids"][0]
    assert run["trace"]["tensors"][before]["value_source"] == "paged"
    assert values(client, run, before, [0, 65536, 131071]) == [0, 65536, 131071]
    assert len(str(run)) < 15000
    saved = project(client, {**fixture, "capture_mode": "shapes"})
    run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
    assert run["trace"]["error"] is None
    assert all(
        t["values"] == [] and t["value_source"] == "shape" for t in run["trace"]["tensors"].values()
    )


@pytest.mark.parametrize(
    "data",
    [
        b"not numpy",
        b"",
        npy(np.ones((2, 3), dtype=np.float16)),
        npy(np.ones((2, 3), dtype=np.complex64)),
        npy(np.array(["text"])),
        npy(np.ones((1,) * 7)),
        npy(np.array(3.0)),
        npy(np.ones((0, 2))),
        npy(np.ones(2))[:-1],
        npy(np.ones(2)) + b"extra",
    ],
)
def test_rejects_invalid_files_without_saving_partial_entries(tmp_path, data):
    client = TestClient(create_app(tmp_path))
    assert upload(client, data).status_code == 422
    assert client.get("/api/v1/input-fixtures").json() == []
    assert list((tmp_path / "inputs").iterdir()) == []


def test_object_arrays_are_rejected_without_unpickling(tmp_path):
    marker = tmp_path / "pickle-executed"

    class Payload:
        def __reduce__(self):
            return eval, (f"__import__('pathlib').Path({str(marker)!r}).touch()",)

    response = upload(TestClient(create_app(tmp_path)), npy(np.array([Payload()], dtype=object)))
    assert response.status_code == 422
    assert not marker.exists()


def test_rejects_header_bombs_npz_and_request_limits(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f8", "fortran_order": False, "shape": (2**40,)}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed headers must be rejected before loading or mapping an array")

    monkeypatch.setattr(np, "load", forbidden)
    assert upload(client, stream.getvalue()).status_code == 422
    assert upload(client, b"\x93NUMPY\x02\x00" + (2**31).to_bytes(4, "little")).status_code == 422
    zip_file = io.BytesIO()
    np.savez_compressed(zip_file, values=np.ones(2))
    assert upload(client, zip_file.getvalue()).status_code == 422
    assert upload(client, b"anything", file_name="sample.npz").status_code == 422
    assert upload(client, b"anything", name="   ").status_code == 422
    assert (
        client.post(
            "/api/v1/input-fixtures/upload?name=Big&file_name=big.npy",
            content=b"a",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(MAX_UPLOAD_BYTES + 1),
            },
        ).status_code
        == 413
    )
    monkeypatch.setattr("tensorviewer.app.MAX_UPLOAD_BYTES", 4)
    assert upload(client, iter([b"1234", b"5678"])).status_code == 413
    assert list((tmp_path / "inputs").iterdir()) == []


def test_forged_metadata_missing_files_and_corruption_fail_without_running_code(tmp_path):
    client = TestClient(create_app(tmp_path))
    fixture = upload(client, npy(np.ones((2, 4), dtype=np.float32))).json()
    wrong = {**fixture["input"], "shape": [4, 2]}
    assert (
        client.post("/api/v1/input-fixtures", json={**fixture, "input": wrong}).status_code == 422
    )
    wrong = {**fixture["input"], "uploaded": {**fixture["input"]["uploaded"], "id": "f" * 32}}
    assert (
        client.post("/api/v1/input-fixtures", json={**fixture, "input": wrong}).status_code == 422
    )
    marker = tmp_path / "executed"
    saved = project(client, fixture, f"from pathlib import Path\nPath({str(marker)!r}).touch()")
    path = tmp_path / "inputs" / (fixture["input"]["uploaded"]["id"] + ".npy")
    path.write_bytes(npy(np.full((2, 4), 99, dtype=np.float32)))
    run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
    assert "changed" in run["trace"]["error"]["message"]
    assert not marker.exists()
    path.unlink()
    assert client.post(f"/api/v1/projects/{saved['id']}/runs").status_code == 409


def test_custom_check_after_shape_transform_accepts_uploaded_input(tmp_path):
    client = TestClient(create_app(tmp_path))
    fixture = upload(client, npy(np.ones((2, 3, 4), dtype=np.float32))).json()
    custom = client.post(
        "/api/v1/components",
        json={
            "name": "Double",
            "code": "from torch import nn\nclass Double(nn.Module):\n def forward(self,x):\n  return x*2",
            "class_name": "Double",
        },
    ).json()
    request = {
        "input": fixture["input"],
        "blueprint": {
            "has_input": True,
            "components": [
                {"id": "flatten", "kind": "flatten", "parameters": {}},
                {"id": "custom", "kind": "custom", "custom": custom},
            ],
        },
    }
    result = client.post("/api/v1/compose/check", json=request)
    assert result.status_code == 200, result.text
    assert result.json()["valid"], result.json()
    assert result.json()["stages"][-1]["shape"] == [2, 12]
    saved = client.post(
        "/api/v1/projects",
        json={**request, "name": "Uploaded custom sequence", "code": "placeholder"},
    ).json()
    run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
    assert run["trace"]["error"] is None
    assert values(client, run, run["trace"]["output_ids"][0], [0, 23]) == [2, 2]
