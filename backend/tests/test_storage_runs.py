import sqlite3

from tensorviewer.declarations import read_project
from tensorviewer.models import Run, Trace
from tensorviewer.storage import Store

MODEL = """
import torch
from torch import nn

# input x: batch=2, features=4


class Double(nn.Module):
    def forward(self, x):
        return x * 2
"""


def project_column(path):
    with sqlite3.connect(path / "tensorviewer.sqlite3") as c:
        return dict(c.execute("SELECT id, project FROM runs").fetchall())


def test_a_new_run_keeps_its_draft_beside_its_body(tmp_path):
    store = Store(tmp_path)
    project = store.save_project(read_project(MODEL).draft)
    run = store.save_run(project, Trace())
    # Listing reads the draft from its own column, not from inside the body.
    assert project_column(tmp_path)[run.id] == run.project.model_dump_json()
    [listed] = store.runs(project.id)
    assert listed.id == run.id and listed.project == run.project


def test_runs_saved_before_the_column_take_their_draft_from_the_body_once(tmp_path):
    # A database from before the column: six columns, the draft only in the body.
    store = Store(tmp_path)
    project = store.save_project(read_project(MODEL).draft)
    run = store.save_run(project, Trace())
    with sqlite3.connect(tmp_path / "tensorviewer.sqlite3") as c:
        c.execute("DROP TABLE runs")
        c.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, project_id TEXT, created_at TEXT,"
            " operation_count INTEGER, failed INTEGER, body TEXT)"
        )
        bare = Run.model_validate_json(run.model_dump_json())
        bare.project = None
        c.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            [
                (run.id, project.id, run.created_at, 0, 0, run.model_dump_json()),
                ("no-draft", project.id, "2000-01-01", 0, 0, bare.model_dump_json()),
            ],
        )
    reopened = Store(tmp_path)
    assert project_column(tmp_path) == {run.id: None, "no-draft": None}
    listed = {item.id: item.project for item in reopened.runs(project.id)}
    assert listed == {run.id: run.project, "no-draft": None}
    # Filled in once: a run without a draft is marked, so it is not read again.
    assert project_column(tmp_path) == {run.id: run.project.model_dump_json(), "no-draft": ""}
    assert {item.id: item.project for item in reopened.runs(project.id)} == listed
