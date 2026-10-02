import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .input_files import checksum, import_array
from .models import (
    CustomComponent,
    CustomComponentDraft,
    InputFixture,
    InputFixtureDraft,
    InputSpec,
    Project,
    ProjectDraft,
    Run,
    RunSummary,
    SavedWeights,
    Trace,
    UploadedTensor,
)
from .weights import import_checkpoint


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = directory / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        self.input_dir = directory / "inputs"
        self.input_dir.mkdir(exist_ok=True)
        self.weights_dir = directory / "weights"
        self.weights_dir.mkdir(exist_ok=True)
        self.path = directory / "tensorviewer.sqlite3"
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, updated_at TEXT, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, project_id TEXT, created_at TEXT, operation_count INTEGER, failed INTEGER, body TEXT)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_project ON runs(project_id, created_at)"
            )

            connection.execute(
                "CREATE TABLE IF NOT EXISTS components (id TEXT PRIMARY KEY, created_at TEXT, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS input_fixtures (id TEXT PRIMARY KEY, created_at TEXT, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS input_files (id TEXT PRIMARY KEY, body TEXT)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS weights (id TEXT PRIMARY KEY, created_at TEXT, body TEXT)"
            )

    def weights(self) -> list[SavedWeights]:
        with self.connect() as c:
            return [
                SavedWeights.model_validate_json(row[0])
                for row in c.execute("SELECT body FROM weights ORDER BY created_at DESC")
            ]

    def saved_weights(self, asset_id: str) -> SavedWeights | None:
        with self.connect() as c:
            row = c.execute("SELECT body FROM weights WHERE id=?", (asset_id,)).fetchone()
            return SavedWeights.model_validate_json(row[0]) if row else None

    def import_weights(self, path: Path, name: str, file_name: str) -> SavedWeights:
        destination = self.weights_dir / f"{uuid4().hex}.pt"
        try:
            tensors = import_checkpoint(path, destination)
            saved = SavedWeights(
                id=destination.stem,
                name=name.strip(),
                file_name=file_name,
                created_at=now(),
                sha256=checksum(destination),
                byte_count=destination.stat().st_size,
                tensors=tensors,
            )
            with self.connect() as c:
                c.execute(
                    "INSERT INTO weights VALUES (?, ?, ?)",
                    (saved.id, saved.created_at, saved.model_dump_json()),
                )
            return saved
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def uploaded_tensor(self, asset_id: str) -> UploadedTensor | None:
        with self.connect() as c:
            row = c.execute("SELECT body FROM input_files WHERE id=?", (asset_id,)).fetchone()
            return UploadedTensor.model_validate_json(row[0]) if row else None

    def import_input(self, path: Path, name: str, file_name: str) -> InputFixture:
        destination = self.input_dir / f"{uuid4().hex}.npy"
        try:
            uploaded = import_array(path, destination, file_name)
            draft = InputFixtureDraft(
                name=name,
                input=InputSpec(
                    shape=uploaded.shape,
                    dtype=uploaded.dtype,
                    axis_names=[],
                    generator="uploaded",
                    uploaded=uploaded,
                ),
            )
            fixture = InputFixture(**draft.model_dump(), id=str(uuid4()), created_at=now())
            with self.connect() as c:
                c.execute(
                    "INSERT INTO input_files VALUES (?, ?)",
                    (uploaded.id, uploaded.model_dump_json()),
                )
                c.execute(
                    "INSERT INTO input_fixtures VALUES (?, ?, ?)",
                    (fixture.id, fixture.created_at, fixture.model_dump_json()),
                )
            return fixture
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def input_fixtures(self) -> list[InputFixture]:
        with self.connect() as c:
            return [
                InputFixture.model_validate_json(row[0])
                for row in c.execute("SELECT body FROM input_fixtures ORDER BY created_at DESC")
            ]

    def save_input_fixture(self, draft: InputFixtureDraft) -> InputFixture:
        fixture = InputFixture(**draft.model_dump(), id=str(uuid4()), created_at=now())
        with self.connect() as c:
            c.execute(
                "INSERT INTO input_fixtures VALUES (?, ?, ?)",
                (fixture.id, fixture.created_at, fixture.model_dump_json()),
            )
        return fixture

    def components(self) -> list[CustomComponent]:
        with self.connect() as c:
            return [
                CustomComponent.model_validate_json(row[0])
                for row in c.execute("SELECT body FROM components ORDER BY created_at DESC")
            ]

    def save_component(self, draft: CustomComponentDraft) -> CustomComponent:
        component = CustomComponent(**draft.model_dump(), id=str(uuid4()), created_at=now())
        with self.connect() as c:
            c.execute(
                "INSERT INTO components VALUES (?, ?, ?)",
                (component.id, component.created_at, component.model_dump_json()),
            )
        return component

    def connect(self):
        return sqlite3.connect(self.path)

    def projects(self) -> list[Project]:
        with self.connect() as c:
            return [
                Project.model_validate_json(row[0])
                for row in c.execute("SELECT body FROM projects ORDER BY updated_at DESC")
            ]
        return None

    def project(self, project_id: str) -> Project | None:
        with self.connect() as c:
            row = c.execute("SELECT body FROM projects WHERE id=?", (project_id,)).fetchone()
            return Project.model_validate_json(row[0]) if row else None
        return None

    def save_project(self, draft: ProjectDraft, existing: Project | None = None) -> Project:
        project = Project(
            **draft.model_dump(),
            id=existing.id if existing else str(uuid4()),
            created_at=existing.created_at if existing else now(),
            updated_at=now(),
        )
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO projects VALUES (?, ?, ?)",
                (project.id, project.updated_at, project.model_dump_json()),
            )
        return project

    def save_run(self, project: Project, trace: Trace, run_id: str | None = None) -> Run:
        run = Run(
            id=run_id or str(uuid4()),
            project_id=project.id,
            created_at=now(),
            project=ProjectDraft(**project.model_dump()),
            trace=trace,
        )
        with self.connect() as c:
            c.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    project.id,
                    run.created_at,
                    len(trace.operations),
                    bool(trace.error),
                    run.model_dump_json(),
                ),
            )
        return run

    def runs(self, project_id: str) -> list[RunSummary]:
        with self.connect() as c:
            return [
                RunSummary(
                    id=r[0],
                    project_id=project_id,
                    created_at=r[1],
                    operation_count=r[2],
                    failed=bool(r[3]),
                    project=ProjectDraft.model_validate_json(r[4]) if r[4] else None,
                )
                for r in c.execute(
                    # json_extract reads the draft without decoding the trace in Python.
                    "SELECT id, created_at, operation_count, failed, json_extract(body, '$.project') FROM runs WHERE project_id=? ORDER BY created_at DESC LIMIT 20",
                    (project_id,),
                )
            ]
        return None

    def run(self, run_id: str) -> Run | None:
        with self.connect() as c:
            row = c.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
            return Run.model_validate_json(row[0]) if row else None
        return None
