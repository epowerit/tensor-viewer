import os
import shutil
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .composer import CATALOG, canonical_project, compose
from .models import (
    CompositionPlan,
    CompositionRequest,
    Project,
    ProjectDraft,
    Run,
    RunSummary,
    Template,
)
from .runner import run_project
from .snapshots import read_snapshot
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

    @app.get("/api/v1/toolbox")
    def toolbox():
        return CATALOG

    @app.post("/api/v1/compose", response_model=CompositionPlan)
    def preview_composition(request: CompositionRequest):
        return compose(request)

    @app.get("/api/v1/projects", response_model=list[Project])
    def projects():
        return store.projects()

    @app.post("/api/v1/projects", response_model=Project, status_code=201)
    def create_project(draft: ProjectDraft):
        return store.save_project(canonical_project(draft))

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
        return store.save_project(canonical_project(draft), find_project(project_id))

    @app.post("/api/v1/projects/{project_id}/runs", response_model=Run, status_code=201)
    def execute(project_id: str):
        project = find_project(project_id)
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        try:
            run_id = str(uuid4())
            snapshot_dir = store.snapshot_dir / run_id
            try:
                trace = run_project(project, snapshot_dir=snapshot_dir)
                return store.save_run(project, trace, run_id)
            except Exception:
                shutil.rmtree(snapshot_dir, ignore_errors=True)
                raise
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

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/values")
    def tensor_values(run_id: str, tensor_id: str, indices: str):
        run = store.run(run_id)
        tensor = run.trace.tensors.get(tensor_id) if run else None
        if tensor is None:
            raise HTTPException(404, "Tensor not found")
        try:
            if len(indices) > 8192:
                raise ValueError()
            positions = [int(index) for index in indices.split(",")]
            if not 1 <= len(positions) <= 256 or any(i < 0 or i >= tensor.numel for i in positions):
                raise ValueError()
        except ValueError:
            raise HTTPException(422, "Request 1–256 valid logical element indices.") from None
        if tensor.value_source == "shape":
            raise HTTPException(409, "Shape runs do not compute numeric values.")
        try:
            values = (
                read_snapshot(store.snapshot_dir / run.id, tensor.id, positions)
                if tensor.value_source == "paged"
                else [tensor.values[i] for i in positions]
            )
        except (OSError, ValueError):
            raise HTTPException(
                410, "The tensor snapshot is no longer available. Run the project again."
            ) from None
        return {"indices": positions, "values": values}

    return app


app = create_app()
