import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .models import Project, ProjectDraft, Run, RunSummary, Template
from .runner import run_project
from .storage import Store
from .templates import TEMPLATES


def create_app(data_dir: Path | None = None):
    app = FastAPI(
        title="TensorViewer",
        version="0.1.0",
        description="Local, trusted-code tensor lessons. API contract version 1.",
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )
    store = Store(
        data_dir
        or Path(os.getenv("TENSORVIEWER_DATA_DIR", Path(__file__).resolve().parents[1] / ".data"))
    )
    run_lock = Lock()

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "schema_version": "1", "execution": "local-trusted-code"}

    @app.get("/api/v1/templates", response_model=list[Template])
    def templates():
        return TEMPLATES

    @app.get("/api/v1/projects", response_model=list[Project])
    def projects():
        return store.projects()

    @app.post("/api/v1/projects", response_model=Project, status_code=201)
    def create_project(draft: ProjectDraft):
        return store.save_project(draft)

    def find_project(project_id):
        project = store.project(project_id)
        if project is None:
            raise HTTPException(404, "Project not found")
        return project

    @app.get("/api/v1/projects/{project_id}", response_model=Project)
    def get_project(project_id: str):
        return find_project(project_id)

    @app.put("/api/v1/projects/{project_id}", response_model=Project)
    def update_project(project_id: str, draft: ProjectDraft):
        return store.save_project(draft, find_project(project_id))

    @app.post("/api/v1/projects/{project_id}/runs", response_model=Run, status_code=201)
    def execute(project_id: str):
        project = find_project(project_id)
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        try:
            return store.save_run(project, run_project(project))
        finally:
            run_lock.release()

    @app.get("/api/v1/projects/{project_id}/runs", response_model=list[RunSummary])
    def list_runs(project_id: str):
        find_project(project_id)
        return store.runs(project_id)

    @app.get("/api/v1/runs/{run_id}", response_model=Run)
    def get_run(run_id: str):
        run = store.run(run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    return app


app = create_app()
