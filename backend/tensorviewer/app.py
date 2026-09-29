import os
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from .composer import CATALOG, canonical_project, compose
from .custom_components import ComponentChecks
from .input_files import MAX_UPLOAD_BYTES
from .models import (
    CompositionPlan,
    CompositionRequest,
    CustomComponent,
    CustomComponentDraft,
    InputFixture,
    InputFixtureDraft,
    InputSpec,
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
    checks = ComponentChecks()

    def validate_input(spec: InputSpec):
        if not spec.uploaded:
            return
        saved = store.uploaded_tensor(spec.uploaded.id)
        if saved is None or saved != spec.uploaded:
            raise HTTPException(
                422, "This uploaded input is not in the local library or its metadata has changed."
            )
        if not (store.input_dir / f"{saved.id}.npy").is_file():
            raise HTTPException(
                409,
                "The uploaded tensor file is missing. Upload it again or restore the inputs backup.",
            )

    def validate_project_inputs(project: ProjectDraft):
        for item in project.forward_inputs:
            validate_input(item.input)

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "schema_version": "1", "execution": "local-trusted-code"}

    @app.get("/api/v1/templates", response_model=list[Template])
    def templates():
        return TEMPLATES

    @app.get("/api/v1/toolbox")
    def toolbox():
        return CATALOG + [
            dict(
                kind="custom",
                title=c.name,
                description=c.description,
                group="Custom",
                parameters=[],
                custom=c,
            )
            for c in store.components()
        ]

    @app.post("/api/v1/compose", response_model=CompositionPlan)
    def preview_composition(request: CompositionRequest):
        validate_input(request.input)
        plan = compose(request, checks.resolve)
        if plan.valid and any(c.custom for c in request.blueprint.components):
            plan = checks.verify_sequence(plan, request)
        return plan

    @app.post("/api/v1/compose/check", response_model=CompositionPlan)
    def check_composition(request: CompositionRequest):
        validate_input(request.input)
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run or shape check is in progress. Wait for it to finish.")
        try:
            plan = compose(request, lambda *args: checks.resolve(*args, execute=True))
            if plan.valid and any(c.custom for c in request.blueprint.components):
                plan = checks.verify_sequence(plan, request, execute=True)
            return plan
        finally:
            run_lock.release()

    @app.post("/api/v1/components", response_model=CustomComponent, status_code=201)
    def save_component(draft: CustomComponentDraft):
        return store.save_component(draft)

    @app.get("/api/v1/components", response_model=list[CustomComponent])
    def components():
        return store.components()

    @app.get("/api/v1/input-fixtures", response_model=list[InputFixture])
    def input_fixtures():
        return store.input_fixtures()

    @app.post("/api/v1/input-fixtures", response_model=InputFixture, status_code=201)
    def save_input_fixture(draft: InputFixtureDraft):
        validate_input(draft.input)
        return store.save_input_fixture(draft)

    @app.post(
        "/api/v1/input-fixtures/upload",
        response_model=InputFixture,
        status_code=201,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def upload_input(
        request: Request,
        name: str = Query(min_length=1, max_length=80),
        file_name: str = Query(min_length=1, max_length=200),
    ):
        if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
            raise HTTPException(415, "Upload the .npy file as binary data.")
        if not name.strip():
            raise HTTPException(422, "Give this saved input a name.")
        file_name = file_name.replace("\\", "/").split("/")[-1]
        if not file_name.lower().endswith(".npy"):
            raise HTTPException(422, "Choose a .npy tensor file.")
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, "Use at most 8,388,608 values in a file up to 64 MiB plus its header."
            )
        with tempfile.NamedTemporaryFile(dir=store.input_dir, suffix=".upload") as stream:
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "This tensor file exceeds the upload limit.")
                stream.write(chunk)
            stream.flush()
            try:
                return await run_in_threadpool(
                    store.import_input, Path(stream.name), name, file_name
                )
            except (ValueError, EOFError) as exc:
                raise HTTPException(422, str(exc)) from None

    @app.get("/api/v1/projects", response_model=list[Project])
    def projects():
        return store.projects()

    @app.post("/api/v1/projects", response_model=Project, status_code=201)
    def create_project(draft: ProjectDraft):
        validate_project_inputs(draft)
        return store.save_project(canonical_project(draft, checks.resolve))

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
        validate_project_inputs(draft)
        return store.save_project(
            canonical_project(draft, checks.resolve), find_project(project_id)
        )

    @app.post("/api/v1/projects/{project_id}/runs", response_model=Run, status_code=201)
    def execute(project_id: str):
        project = find_project(project_id)
        validate_project_inputs(project)
        if project.blueprint and any(c.custom for c in project.blueprint.components):
            plan = preview_composition(
                CompositionRequest(
                    blueprint=project.blueprint,
                    input=project.input,
                    capture_mode=project.capture_mode,
                ),
            )
            if not plan.valid:
                raise HTTPException(409, plan.error)
            project = project.model_copy(update={"code": plan.code})
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        try:
            run_id = str(uuid4())
            snapshot_dir = store.snapshot_dir / run_id
            try:
                trace = run_project(project, snapshot_dir=snapshot_dir, input_dir=store.input_dir)
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
