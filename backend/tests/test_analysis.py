import numpy as np
from fastapi.testclient import TestClient

from tensorviewer import analysis
from tensorviewer.app import create_app

VALUES = np.array([0.0, -3.0, 0.5, np.nan, 2.0, -np.inf, -0.25])


def test_search_matches_the_browser_queries():
    def found(test, **options):
        return analysis.search(VALUES, test, **options)["indices"]

    assert found("gt", value=0.4) == [2, 4]
    assert found("eq", value=0.0) == [0]
    assert found("ne", value=0.0) == [1, 2, 4, 6]  # NaN and -inf are not "not zero"
    assert found("gt", value=1, magnitude=True) == [1, 4]
    assert found("nan") == [3] and found("inf") == [5]
    assert found("finite") == [0, 1, 2, 4, 6]
    # A histogram bar is half-open; the last one keeps its end.
    assert found("between", value=0.0, high=2.0) == [0, 2]
    assert found("between", value=0.0, high=2.0, closed=True) == [0, 2, 4]
    capped = analysis.search(np.zeros(10), "eq", limit=3)
    assert capped == {"count": 10, "indices": [0, 1, 2], "truncated": True}


def test_landmarks_find_the_first_extremes_and_every_fault():
    marks = analysis.landmarks(VALUES)
    assert {key: marks[key] for key in ("min", "max", "broken", "broken_count")} == {
        "min": 1,
        "max": 4,
        "broken": [3, 5],
        "broken_count": 2,
    }
    finite = [0.0, -3.0, 0.5, 2.0, -0.25]
    assert marks["quantiles"] == list(np.percentile(finite, [1, 50, 99]))
    assert analysis.landmarks(np.array([np.nan]))["min"] is None


def test_planes_follow_the_chosen_axes_and_reduce_like_torch():
    cube = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    # Rows are axis 2 and columns axis 1: the plane is transposed.
    assert analysis.plane(cube, 2, 1, [1, 0, 0]).tolist() == cube[1].T.tolist()
    result = analysis.margins(cube, 1, 2, [0, 0, 0], "sum")
    assert result == {"rows": [6.0, 22.0, 38.0], "columns": [12.0, 15.0, 18.0, 21.0], "all": 66.0}
    cube[0, 1, 2] = np.nan
    spread = analysis.margins(cube, 1, 2, [0, 0, 0], "max")
    assert spread["rows"][1] == "nan" and spread["columns"][2] == "nan"
    region = analysis.region(cube, 1, 2, [0, 0, 0], (0, 1), (1, 2))
    assert region["count"] == 3 and region["broken"] == 1
    assert region["sum"] == 1.0 + 2.0 + 5.0
    assert region["std"] == float(np.std([1.0, 2.0, 5.0]))
    assert region["values"] == [1.0, 2.0, 5.0, "nan"]


def test_large_tensors_answer_through_the_api(tmp_path):
    client = TestClient(create_app(tmp_path))
    code = (
        "import torch\nfrom torch import nn\n\n\nclass Faults(nn.Module):\n"
        "    def forward(self, x):\n        return torch.log(x - 100)\n"
    )
    project = client.post(
        "/api/v1/projects",
        json={
            "name": "Large",
            "code": code,
            "class_name": "Faults",
            "constructor": {},
            "input": {"shape": [2, 50, 50], "generator": "arange", "dtype": "float32"},
            "capture_mode": "values",
        },
    ).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    result = next(tensor for tensor in run["trace"]["tensors"].values() if tensor["name"] == "log")
    assert result["value_source"] == "paged"
    base = f"/api/v1/runs/{run['id']}/tensors/{result['id']}"
    found = client.get(f"{base}/search", params={"test": "nan"}).json()
    assert found["count"] == 100 and found["indices"][:2] == [0, 1]
    marks = client.get(f"{base}/landmarks").json()
    assert marks["min"] == 101 and marks["max"] == 4999 and marks["broken_count"] == 101
    margins = client.get(
        f"{base}/margins", params={"row": 1, "column": 2, "fixed": "1,0,0", "reduce": "max"}
    ).json()
    assert len(margins["rows"]) == 50 and margins["all"] == float(np.float32(np.log(4899)))
    region = client.get(
        f"{base}/region",
        params={"row": 1, "column": 2, "fixed": "0,0,0", "rows": "2,3", "columns": "0,1"},
    ).json()
    assert region["count"] == 3 and region["broken"] == 1  # log(0) is -inf
    assert client.get(f"{base}/search", params={"test": "maybe"}).status_code == 422
    assert (
        client.get(f"{base}/margins", params={"row": 1, "column": 1, "fixed": "0,0,0"}).status_code
        == 422
    )


def test_comparisons_count_changes_like_the_browser():
    now = np.array([1.5, 2.0, np.nan, 4.0, np.nan, np.inf])
    before = np.array([1.0, 2.0, 3.0, -np.inf, np.nan, np.inf])
    summary = analysis.compare(now, before)
    assert {key: summary[key] for key in ("changed", "compared", "low", "high")} == {
        "changed": 3,  # 1.5, NaN appearing, and -inf becoming 4
        "compared": 6,
        "low": 0.0,
        "high": 0.5,
    }
    assert summary["max_abs"] == 0.5 and summary["allclose"] is False
    close = analysis.compare(np.array([1.0, np.inf]), np.array([1.0 + 1e-9, np.inf]))
    assert close["allclose"] is True and close["changed"] == 1
    assert analysis.compare(np.zeros(3), np.zeros(4)) is None


def test_runs_compare_their_large_tensors(tmp_path):
    client = TestClient(create_app(tmp_path))
    draft = {
        "name": "Large",
        "code": (
            "from torch import nn\n\n\nclass Shift(nn.Module):\n"
            "    def forward(self, x):\n        return x - 1\n"
        ),
        "class_name": "Shift",
        "constructor": {},
        "input": {"shape": [2, 50, 50], "generator": "arange", "dtype": "float32"},
        "capture_mode": "values",
    }
    project = client.post("/api/v1/projects", json=draft).json()
    first = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    draft["code"] = draft["code"].replace("x - 1", "x - 3")
    client.put(f"/api/v1/projects/{project['id']}", json=draft)
    second = client.post(f"/api/v1/projects/{project['id']}/runs").json()

    def result_id(run):
        return run["trace"]["operations"][0]["outputs"][0]

    response = client.post(
        f"/api/v1/runs/{second['id']}/compare",
        json={
            "other_run": first["id"],
            "pairs": [
                [result_id(second), result_id(first)],
                [second["trace"]["input_ids"][0], first["trace"]["input_ids"][0]],
                [result_id(second), "missing"],
            ],
        },
    )
    assert response.status_code == 200, response.text
    changed, same, missing = response.json()["results"]
    assert {key: changed[key] for key in ("changed", "compared", "low", "high")} == {
        "changed": 5000,
        "compared": 5000,
        "low": -2.0,
        "high": 0.0,
    }
    assert changed["mean_abs"] == 2.0
    assert same["changed"] == 0 and missing is None


def test_tensors_download_as_npy(tmp_path):
    import io

    client = TestClient(create_app(tmp_path))
    template = client.get("/api/v1/templates").json()[1]["project"]
    project = client.post("/api/v1/projects", json=template).json()
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    tensor_id = run["trace"]["input_ids"][0]
    tensor = run["trace"]["tensors"][tensor_id]
    response = client.get(f"/api/v1/runs/{run['id']}/tensors/{tensor_id}/npy")
    assert response.status_code == 200
    assert f'filename="{tensor["name"]}.npy"' in response.headers["content-disposition"]
    loaded = np.load(io.BytesIO(response.content))
    assert list(loaded.shape) == tensor["shape"]
    assert loaded.reshape(-1).tolist() == [float(v) for v in tensor["values"]]


def test_axis_profiles_summarise_each_index():
    cube = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    profile = analysis.axis_profile(cube, 1)
    assert profile["mean"] == [cube[:, i].mean() for i in range(3)]
    assert profile["max"] == [cube[:, i].max() for i in range(3)]
    assert profile["std"] == [cube[:, i].std() for i in range(3)]
    cube[0, 2, 0] = np.nan
    assert analysis.axis_profile(cube, 1)["min"][2] == "nan"


def test_thumbnails_average_each_slice_into_blocks():
    images = np.arange(3 * 4 * 4, dtype=np.float64).reshape(3, 4, 4)
    result = analysis.thumbnails(images, 1, 2, [0, 0, 0], 0, size=2)
    assert (result["rows"], result["columns"], result["total"]) == (2, 2, 3)
    # Slice 1 holds 16..31; its top-left 2×2 block is 16, 17, 20, 21.
    assert result["thumbs"][1][0] == (16 + 17 + 20 + 21) / 4
    assert len(result["thumbs"]) == 3
    small = analysis.downsample(np.ones((3, 5)), 8)
    assert small.shape == (3, 5)


def test_sums_over_an_axis_are_checked_like_a_contract():
    rows = np.array([[0.25, 0.75], [0.5, 0.6], [np.nan, 1.0]])
    assert analysis.sums(rows, -1, 1.0) == {"count": 3, "off": 2, "worst": "nan"}
    assert analysis.sums(rows[:2], -1, 1.0) == {"count": 2, "off": 1, "worst": 1.1}
    assert analysis.sums(np.full((2, 2), 0.5), 1, 1.0)["off"] == 0
