import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import Project, ProjectDraft, Run, RunSummary, Trace


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = directory / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
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
                )
                for r in c.execute(
                    "SELECT id, created_at, operation_count, failed FROM runs WHERE project_id=? ORDER BY created_at DESC LIMIT 20",
                    (project_id,),
                )
            ]
        return None

    def run(self, run_id: str) -> Run | None:
        with self.connect() as c:
            row = c.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
            return Run.model_validate_json(row[0]) if row else None
        return None
