import pytest
from fastapi.testclient import TestClient

from tensorviewer.app import create_app
from tensorviewer.declarations import read_project
from tensorviewer.library import entries, project_name
from tensorviewer.tracing import MAX_OPERATIONS
from tensorviewer.worker import execute

LIBRARY = entries()


def test_the_library_runs_from_basics_to_multimodal():
    assert [entry.number for entry in LIBRARY] == list(range(1, len(LIBRARY) + 1))
    assert len(LIBRARY) >= 20
    tracks = [entry.track for entry in LIBRARY]
    assert tracks[0] == "Foundations" and tracks[-1] == "Multimodal"
    assert {"Attention", "Text transformers", "Vision transformers"} <= set(tracks)


@pytest.mark.parametrize("entry", LIBRARY, ids=[entry.id for entry in LIBRARY])
@pytest.mark.parametrize("capture", ["values", "shapes"])
def test_every_library_project_reads_and_runs(entry, capture):
    read = read_project(entry.code, project_name(entry))
    assert read.notes == []
    draft = read.draft.model_copy(update={"capture_mode": capture})
    trace = execute(draft)
    assert trace.error is None, trace.error
    assert 0 < len(trace.operations) < MAX_OPERATIONS


def test_pasted_library_code_is_the_same_project(tmp_path):
    client = TestClient(create_app(tmp_path))
    installed = client.post("/api/v1/library/install").json()
    assert [project["name"] for project in installed] == [project_name(e) for e in LIBRARY]
    # Installing again adds nothing; the newest project is the first lesson.
    assert client.post("/api/v1/library/install").json() == []
    assert client.get("/api/v1/projects").json()[0]["name"] == project_name(LIBRARY[0])
    # Reading the same code, as a paste or upload does, gives the same draft.
    entry = LIBRARY[16]
    read = client.post(
        "/api/v1/sources/read", json={"code": entry.code, "name": project_name(entry)}
    ).json()
    library_project = next(p for p in installed if p["name"] == project_name(entry))
    for field in ("code", "class_name", "constructor", "input", "input_name", "additional_inputs"):
        assert read["draft"][field] == library_project[field]
    assert client.post("/api/v1/sources/read", json={"code": "x = 1"}).status_code == 422
