import keyword
import os
import re
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from . import analysis
from .composer import CATALOG, canonical_project, compose
from .custom_components import ComponentChecks
from .declarations import ReadProject, read_project
from .environments import (
    EnvironmentRequest,
    RuntimeEnvironment,
    create_environment,
    environment_python,
    list_environments,
)
from .input_files import MAX_UPLOAD_BYTES
from .library import LibraryEntry
from .library import entries as library_entries
from .library import project_name as library_project_name
from .models import (
    CompositionPlan,
    CompositionRequest,
    CustomComponent,
    CustomComponentDraft,
    Evaluation,
    GradientFlow,
    GradientRequest,
    InputFixture,
    InputFixtureDraft,
    InputSpec,
    Knockout,
    KnockoutSweep,
    LatestRun,
    LayerNormalizationStatistics,
    Project,
    ProjectDraft,
    ReductionStatistics,
    Run,
    RunSummary,
    SavedWeights,
    Sensitivity,
    SensitivityRequest,
    SeriesPoint,
    SeriesRequest,
    SoftmaxStatistics,
    SweepResult,
    Template,
    Trace,
    WatchRequest,
    WatchSeries,
    WeightCheck,
    WeightReport,
    WeightSpectrum,
    WeightsRequest,
    WhatIfRequest,
)
from .operations.normalization import layer_normalization_spec
from .operations.reduction import reduction_spec
from .operations.softmax import softmax_spec
from .reduction_statistics import snapshot_reduction, supports_reference
from .runner import (
    run_evaluation,
    run_gradients,
    run_project,
    run_sensitivity,
    run_sweep,
)
from .snapshots import read_snapshot, snapshot_array
from .softmax_statistics import snapshot_softmax
from .source_projects import import_git
from .statistics import snapshot_statistics
from .storage import SCRATCH, Store
from .templates import TEMPLATES
from .weights import MAX_WEIGHT_BYTES


class GitImportRequest(BaseModel):
    repository: str = Field(min_length=1, max_length=1000)
    revision: str = Field(default="HEAD", min_length=1, max_length=200)
    subdirectory: str = Field(default=".", min_length=1, max_length=240)


class SourceRead(BaseModel):
    code: str = Field(min_length=1, max_length=500000)
    name: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, pattern=r"^[A-Za-z_]\w*$", max_length=100)


class ShapeCheck(BaseModel):
    project: ProjectDraft
    trace: Trace


class SourceImport(BaseModel):
    files: dict[str, str]
    repository: dict[str, str]
    skipped: int


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
    environments = store.path.parent / "environments"

    @app.post("/api/v1/sources/git", response_model=SourceImport)
    async def import_repository(request: GitImportRequest):
        try:
            return await run_in_threadpool(
                import_git, request.repository, request.revision, request.subdirectory
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/v1/environments", response_model=list[RuntimeEnvironment])
    def python_environments():
        return list_environments(environments)

    @app.post("/api/v1/environments", response_model=RuntimeEnvironment)
    async def setup_environment(request: EnvironmentRequest):
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run or environment setup is in progress.")
        try:
            return await run_in_threadpool(create_environment, environments, request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        finally:
            run_lock.release()

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
        try:
            environment_python(environments, project.environment)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
        for item in project.forward_inputs:
            validate_input(item.input)
        if project.weights:
            saved = store.saved_weights(project.weights.id)
            if saved is None or saved != project.weights:
                raise HTTPException(
                    422, "These weights are not in the local library or their metadata has changed."
                )
            if not (store.weights_dir / f"{saved.id}.pt").is_file():
                raise HTTPException(
                    409,
                    "The saved weights file is missing. Import it again or restore the weights backup.",
                )

    @app.get("/api/v1/weights", response_model=list[SavedWeights])
    def list_weights():
        return store.weights()

    @app.post(
        "/api/v1/weights/upload",
        response_model=SavedWeights,
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
    async def upload_weights(
        request: Request,
        name: str = Query(min_length=1, max_length=80),
        file_name: str = Query(min_length=1, max_length=200),
    ):
        if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
            raise HTTPException(415, "Upload the checkpoint as binary data.")
        file_name = file_name.replace("\\", "/").split("/")[-1]
        if not name.strip() or not file_name.lower().endswith((".pt", ".pth")):
            raise HTTPException(422, "Choose a .pt or .pth state dictionary and give it a name.")
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_WEIGHT_BYTES:
            raise HTTPException(413, "Use a checkpoint file up to 64 MiB.")
        with tempfile.NamedTemporaryFile(dir=store.weights_dir, suffix=".upload") as stream:
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_WEIGHT_BYTES:
                    raise HTTPException(413, "Use a checkpoint file up to 64 MiB.")
                stream.write(chunk)
            stream.flush()
            try:
                return await run_in_threadpool(
                    store.import_weights, Path(stream.name), name, file_name
                )
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None

    @app.post("/api/v1/weights/check", response_model=WeightCheck)
    def check_weights(draft: ProjectDraft):
        if not draft.weights:
            raise HTTPException(422, "Choose saved weights before checking compatibility.")
        validate_project_inputs(draft)
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run or shape check is in progress. Wait for it to finish.")
        try:
            project = canonical_project(draft, checks.resolve)
            result = run_project(
                project,
                input_dir=store.input_dir,
                weights_dir=store.weights_dir,
                check_weights_only=True,
                python_executable=environment_python(environments, project.environment),
            )
            if result.error:
                return WeightCheck(
                    compatible=False,
                    issues=result.weight_check.issues
                    if result.weight_check and result.weight_check.issues
                    else [result.error.message],
                )
            return result.weight_check or WeightCheck(
                compatible=False, issues=["The worker did not validate these weights."]
            )
        finally:
            run_lock.release()

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "schema_version": "1", "execution": "local-trusted-code"}

    @app.get("/api/v1/templates", response_model=list[Template])
    def templates():
        return TEMPLATES

    @app.post("/api/v1/sources/read", response_model=ReadProject)
    def read_source(request: SourceRead):
        """A ready-to-run draft from code alone, exactly as the library uses."""
        try:
            return read_project(request.code, request.name, request.model)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/v1/library", response_model=list[LibraryEntry])
    def library():
        return list(library_entries())

    @app.post("/api/v1/library/install", response_model=list[Project])
    def install_library():
        """Create library projects that are not in the workspace yet, by name."""
        present = {project.name for project in store.projects()}
        created = []
        # Newest first in the project list: create the last one first.
        for entry in reversed(library_entries()):
            name = library_project_name(entry)
            if name in present:
                continue
            draft = read_project(entry.code, name).draft
            validate_project_inputs(draft)
            created.append(store.save_project(canonical_project(draft, checks.resolve)))
        return list(reversed(created))

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

    @app.get("/api/v1/latest-runs", response_model=list[LatestRun])
    def latest_runs():
        return store.latest_runs()

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

    @app.delete("/api/v1/projects/{project_id}", status_code=204)
    def delete_project(project_id: str):
        # A run finishing after the delete would save into a project that is
        # gone, so a project is not deleted while a run records.
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is in progress. Delete the project once it finishes.")
        try:
            if not store.delete_project(project_id):
                raise HTTPException(404, "Project not found")
        finally:
            run_lock.release()

    @app.post("/api/v1/shape-check", response_model=ShapeCheck)
    def shape_check(draft: ProjectDraft):
        """Dry-run a draft on metadata tensors. Nothing is saved and no values exist."""
        if draft.blueprint:
            raise HTTPException(422, "The builder checks shapes as components change.")
        validate_project_inputs(draft)
        project = draft.model_copy(update={"capture_mode": "shapes"})
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        try:
            trace = run_project(
                project,
                timeout=10,
                input_dir=store.input_dir,
                weights_dir=store.weights_dir,
                python_executable=environment_python(environments, project.environment),
            )
        finally:
            run_lock.release()
        return ShapeCheck(project=project, trace=trace)

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
                trace = run_project(
                    project,
                    snapshot_dir=snapshot_dir,
                    input_dir=store.input_dir,
                    weights_dir=store.weights_dir,
                    python_executable=environment_python(environments, project.environment),
                )
                return store.save_run(project, trace, run_id)
            except Exception:
                shutil.rmtree(snapshot_dir, ignore_errors=True)
                raise
        finally:
            run_lock.release()

    def knocked_result(recorded: Run, rule: Knockout | KnockoutSweep):
        """The recorded tensor a knockout replaces; 422 when there is none."""
        operations = recorded.trace.operations
        if rule.step >= len(operations):
            raise HTTPException(422, f"The run has {len(operations)} steps, not {rule.step + 1}.")
        outputs = operations[rule.step].outputs
        if rule.output >= len(outputs):
            raise HTTPException(422, f"Step {rule.step + 1} has no result {rule.output + 1}.")
        return recorded.trace.tensors[outputs[rule.output]]

    def patch_file(recorded: Run, rule: Knockout | KnockoutSweep, scratch: Path) -> str:
        """Where the worker reads the other run's result a patch puts in."""
        other = store.run(rule.patch_from or "")
        if other is None:
            raise HTTPException(404, "The run to patch from was not found")
        target = knocked_result(recorded, rule)
        mine = recorded.trace.operations[rule.step]
        theirs = (
            other.trace.operations[rule.step] if rule.step < len(other.trace.operations) else None
        )
        if theirs is None or theirs.kind != mine.kind or rule.output >= len(theirs.outputs):
            raise HTTPException(
                422,
                f"The other run has no {mine.kind} at step {rule.step + 1} to patch in.",
            )
        source_id = theirs.outputs[rule.output]
        source = other.trace.tensors[source_id]
        if source.shape != target.shape:
            raise HTTPException(
                422,
                f"The other run's step {rule.step + 1} made {source.shape}, not "
                f"{target.shape}: only a result of the same shape can be patched in.",
            )
        found = tensor_spec(other, source_id)
        if found is None:
            raise HTTPException(
                422, "The other run recorded that step's shape only, not its values."
            )
        if "path" in found:
            return found["path"]
        path = scratch / "patch.npy"
        values = np.array([float(value) for value in found["values"]], dtype=np.float64)
        np.save(path, values.reshape(source.shape), allow_pickle=False)
        return str(path)

    @app.post("/api/v1/runs/{run_id}/what-if", response_model=Run, status_code=201)
    def what_if(run_id: str, request: WhatIfRequest):
        """Runs a recorded run's code again with some input cells set, in
        another precision, after training steps on its weights, or with one
        step's result knocked out or patched in from another run.

        The result is kept in memory for a while, outside the project's
        history, so its values can be read like any run's.
        """
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        if recorded.project.blueprint:
            raise HTTPException(422, "What-if runs need a code project.")
        with tempfile.TemporaryDirectory(prefix="tensorviewer-patch-") as scratch:
            knockout = request.knockout
            if knockout:
                knocked_result(recorded, knockout)
                knockout = knockout.model_copy(
                    update={
                        "patch_path": patch_file(recorded, knockout, Path(scratch))
                        if knockout.mode == "patch"
                        else None
                    }
                )
            try:
                spec = InputSpec.model_validate(
                    {
                        **recorded.project.input.model_dump(),
                        "edits": [edit.model_dump() for edit in request.edits],
                        "precision": request.precision,
                        "knockout": knockout.model_dump() if knockout else None,
                    }
                )
            except ValidationError as error:
                raise HTTPException(422, error.errors()[0]["msg"]) from None
            project = recorded.project.model_copy(update={"input": spec, "capture_mode": "values"})
            validate_project_inputs(project)
            if not run_lock.acquire(blocking=False):
                raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
            try:
                scratch_id = f"{SCRATCH}{uuid4()}"
                snapshot_dir = store.snapshot_dir / scratch_id
                try:
                    trace = run_project(
                        project,
                        snapshot_dir=snapshot_dir,
                        input_dir=store.input_dir,
                        weights_dir=store.weights_dir,
                        python_executable=environment_python(environments, project.environment),
                        learn=request.learn,
                    )
                except Exception:
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
                    raise
                if request.learn and trace.error and not trace.operations:
                    raise HTTPException(422, trace.error.message)
                return store.save_scratch_run(recorded.project_id, project, trace, scratch_id)
            finally:
                run_lock.release()

    @app.post("/api/v1/runs/{run_id}/knockout-sweep", response_model=SweepResult)
    def knockout_sweep(run_id: str, request: KnockoutSweep):
        """Knocks out each slice of one step's result in turn, along an axis,
        and reports how far the model's output moved each time. Nothing is
        saved."""
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        if recorded.project.blueprint:
            raise HTTPException(422, "Knockouts need a code project.")
        if request.mode == "patch" and not request.patch_from:
            raise HTTPException(422, "A patch names the run it comes from.")
        target = knocked_result(recorded, request)
        if request.axis >= len(target.shape):
            raise HTTPException(
                422, f"Step {request.step + 1}'s result {target.shape} has no axis {request.axis}."
            )
        with tempfile.TemporaryDirectory(prefix="tensorviewer-patch-") as scratch:
            request = request.model_copy(
                update={
                    "patch_path": patch_file(recorded, request, Path(scratch))
                    if request.mode == "patch"
                    else None
                }
            )
            # A what-if's own cells and precision stay; only one knockout at a time.
            input_spec = recorded.project.input.model_copy(update={"knockout": None})
            project = recorded.project.model_copy(
                update={"input": input_spec, "capture_mode": "values"}
            )
            if not run_lock.acquire(blocking=False):
                raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
            try:
                return run_sweep(
                    project,
                    request,
                    input_dir=store.input_dir,
                    weights_dir=store.weights_dir,
                    python_executable=environment_python(environments, project.environment),
                )
            finally:
                run_lock.release()

    def tensor_spec(recorded: Run, tensor_id: str) -> dict | None:
        """How the evaluator loads a recorded state: inline values or its snapshot."""
        tensor = recorded.trace.tensors[tensor_id]
        if tensor.value_source == "inline" and len(tensor.values) == tensor.numel:
            return {"dtype": tensor.dtype, "shape": tensor.shape, "values": tensor.values}
        path = store.snapshot_dir / recorded.id / f"{tensor_id}.npy"
        if path.exists():
            return {"dtype": tensor.dtype, "shape": tensor.shape, "path": str(path)}
        return None

    @app.post("/api/v1/runs/{run_id}/weights", response_model=WeightReport)
    def weight_spectra(run_id: str, request: WeightsRequest | None = None):
        """Every weight the run read, with its norm and singular values, and
        with `against`, what changed in it since another run."""
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        earlier = store.run(request.against) if request and request.against else None
        if request and request.against and earlier is None:
            raise HTTPException(404, "The run to compare with was not found")
        trace = recorded.trace
        weights = {
            tensor.name: (tensor_id, tensor, found)
            for tensor_id, tensor in trace.tensors.items()
            if tensor.role == "parameter" and (found := tensor_spec(recorded, tensor_id))
        }
        if not weights:
            return WeightReport()
        against = (
            {
                tensor.name: found
                for tensor_id, tensor in earlier.trace.tensors.items()
                if tensor.role == "parameter"
                and tensor.name in weights
                and (found := tensor_spec(earlier, tensor_id))
            }
            if earlier
            else {}
        )
        answer = run_evaluation(
            {
                "weights": {name: spec for name, (_, _, spec) in weights.items()},
                "against": against,
            },
            timeout=30,
            python_executable=environment_python(environments, recorded.project.environment),
        )
        if answer.get("kind") == "error":
            return WeightReport(error=answer.get("text"))
        return WeightReport(
            weights=[
                WeightSpectrum(
                    tensor_id=weights[entry["name"]][0],
                    shape=weights[entry["name"]][1].shape,
                    numel=weights[entry["name"]][1].numel,
                    **entry,
                )
                for entry in answer.get("weights", [])
                if entry["name"] in weights
            ]
        )

    @app.post("/api/v1/runs/{run_id}/evaluate-series", response_model=WatchSeries)
    def evaluate_series(run_id: str, request: SeriesRequest):
        """A watch expression at every step where one of the names it reads
        changed, as one number per step. Nothing is saved."""
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        trace = recorded.trace
        used = set(re.findall(r"[A-Za-z_]\w*", request.expression))
        visible: dict[str, str] = {}
        for tensor_id in trace.input_ids:
            visible[trace.tensors[tensor_id].name] = tensor_id
        points: list[dict] = []
        # The inputs' own values first: step 0, on the input's node.
        start = {name: visible[name] for name in sorted(used) if name in visible}
        if start:
            points.append(
                {
                    "at": f"input-{next(iter(start.values()))}",
                    "step": 0,
                    "line": None,
                    "names": start,
                }
            )
        truncated = False
        for op in trace.operations:
            for tensor_id in [*op.outputs, *(m.after for m in op.mutations)]:
                visible[trace.tensors[tensor_id].name] = tensor_id
            names = {name: visible[name] for name in sorted(used) if name in visible}
            if names and (not points or names != points[-1]["names"]):
                if len(points) == 200:
                    truncated = True
                    break
                points.append(
                    {
                        "at": op.id,
                        "step": op.index + 1,
                        "line": op.source.line if op.source else None,
                        "names": names,
                    }
                )
        states = {
            tensor_id: found
            for point in points
            for tensor_id in point["names"].values()
            if (found := tensor_spec(recorded, tensor_id)) is not None
        }
        params = (
            {
                tensor.name: found
                for tensor_id, tensor in trace.tensors.items()
                if tensor.role == "parameter"
                and (found := tensor_spec(recorded, tensor_id)) is not None
            }
            if "params" in used
            else {}
        )
        answer = run_evaluation(
            {
                "expression": request.expression,
                "states": states,
                "params": params,
                "points": [
                    {
                        "at": point["at"],
                        "names": {n: t for n, t in point["names"].items() if t in states},
                    }
                    for point in points
                ],
            },
            timeout=20,
            python_executable=environment_python(environments, recorded.project.environment),
        )
        if answer.get("kind") == "error":
            return WatchSeries(error=answer.get("text"))
        found = {point["at"]: point for point in answer.get("points", [])}
        return WatchSeries(
            points=[
                SeriesPoint(
                    at=point["at"],
                    step=point["step"],
                    line=point["line"],
                    value=found.get(point["at"], {}).get("value"),
                    error=found.get(point["at"], {}).get("error"),
                )
                for point in points
            ],
            truncated=truncated,
            error=answer.get("error"),
        )

    @app.post("/api/v1/runs/{run_id}/evaluate", response_model=Evaluation)
    def evaluate_watch(run_id: str, request: WatchRequest):
        """Evaluates an expression over the run's named tensors at a step.

        Each name reads its latest state up to that step (and the step's own
        result); `params["…"]` reads the model's weights. Nothing is saved.
        """
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        trace = recorded.trace
        visible: dict[str, str] = {}
        for tensor_id in trace.input_ids:
            visible[trace.tensors[tensor_id].name] = tensor_id
        for op in trace.operations:
            for tensor_id in [*op.outputs, *(m.after for m in op.mutations)]:
                visible[trace.tensors[tensor_id].name] = tensor_id
            if op.id == request.at:
                break
        names = sorted(
            name for name in visible if name.isidentifier() and not keyword.iskeyword(name)
        )
        used = set(re.findall(r"[A-Za-z_]\w*", request.expression))

        def spec(tensor_id: str) -> dict | None:
            return tensor_spec(recorded, tensor_id)

        tensors = {
            name: found
            for name in names
            if name in used and (found := spec(visible[name])) is not None
        }
        params = (
            {
                tensor.name: found
                for tensor_id, tensor in trace.tensors.items()
                if tensor.role == "parameter" and (found := spec(tensor_id)) is not None
            }
            if "params" in used
            else {}
        )
        answer = run_evaluation(
            {"expression": request.expression, "tensors": tensors, "params": params},
            python_executable=environment_python(environments, recorded.project.environment),
        )
        # A name the run assigns later reads as not computed yet, not unknown.
        missing = re.match(r"NameError: name '(\w+)' is not defined", answer.get("text") or "")
        later = {tensor.name for tensor in trace.tensors.values()}
        if answer.get("kind") == "error" and missing and missing.group(1) in later:
            answer["text"] = f"{missing.group(1)} is not computed yet at this step"
        return Evaluation(**answer, names=names)

    @app.post("/api/v1/runs/{run_id}/gradients", response_model=GradientFlow)
    def gradients(run_id: str, request: GradientRequest):
        """The gradient's size at every recorded tensor, for one value of a run.

        Runs the recorded code again with gradients on; nothing is saved.
        """
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        tensor = recorded.trace.tensors.get(request.tensor_id)
        if tensor is None or (request.index is not None and request.index >= tensor.numel):
            raise HTTPException(404, "No such result in this run")
        if recorded.project.blueprint:
            raise HTTPException(422, "Gradients need a code project.")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        try:
            found = run_gradients(
                recorded.project.model_copy(update={"capture_mode": "values"}),
                request,
                input_dir=store.input_dir,
                weights_dir=store.weights_dir,
                python_executable=environment_python(environments, recorded.project.environment),
            )
        finally:
            run_lock.release()
        if found.error:
            raise HTTPException(422, found.error.message)
        return found

    @app.post("/api/v1/runs/{run_id}/sensitivity", response_model=Sensitivity)
    def sensitivity(run_id: str, request: SensitivityRequest):
        """How much one result cell moves per unit change of each input cell.

        Runs the recorded code again with gradients on; nothing is saved.
        """
        recorded = store.run(run_id)
        if recorded is None:
            raise HTTPException(404, "Run not found")
        tensor = recorded.trace.tensors.get(request.tensor_id)
        if tensor is None or request.index >= max(1, tensor.numel):
            raise HTTPException(404, "No such cell in this run")
        if recorded.project.blueprint:
            raise HTTPException(422, "Sensitivity needs a code project.")
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "A run is already in progress. Wait for it to finish.")
        try:
            found = run_sensitivity(
                recorded.project.model_copy(update={"capture_mode": "values"}),
                request,
                input_dir=store.input_dir,
                weights_dir=store.weights_dir,
                python_executable=environment_python(environments, recorded.project.environment),
            )
        finally:
            run_lock.release()
        if found.error:
            raise HTTPException(422, found.error.message)
        return found

    @app.get("/api/v1/projects/{project_id}/runs", response_model=list[RunSummary])
    def list_runs(project_id: str):
        find_project(project_id)
        return store.runs(project_id)

    @app.delete("/api/v1/projects/{project_id}/runs")
    def delete_runs_before(project_id: str, before: str):
        """Clear a project's history of every run recorded before the run `before`."""
        find_project(project_id)
        deleted = store.delete_runs_before(project_id, before)
        if deleted is None:
            raise HTTPException(404, "Run not found in this project")
        return {"deleted": deleted}

    @app.delete("/api/v1/runs/{run_id}", status_code=204)
    def delete_run(run_id: str):
        if not store.delete_run(run_id):
            raise HTTPException(404, "Run not found")

    @app.get("/api/v1/runs/{run_id}", response_model=Run)
    def get_run(run_id: str):
        run = store.run(run_id)
        if run is None:
            raise HTTPException(404, "Run not found")
        return run

    @app.get(
        "/api/v1/runs/{run_id}/operations/{operation_id}/normalization",
        response_model=LayerNormalizationStatistics,
    )
    def normalization_statistics(run_id: str, operation_id: str, group: int = Query(ge=0)):
        run = get_run(run_id)
        op = next((op for op in run.trace.operations if op.id == operation_id), None)
        if op is None:
            raise HTTPException(404, "Operation not found")
        try:
            if op.kind != "layer_norm" or op.status != "ok" or op.mutations:
                raise ValueError()
            inputs = [run.trace.tensors[i] for i in op.inputs]
            outputs = [run.trace.tensors[i] for i in op.outputs]
            spec = layer_normalization_spec(op.arguments, inputs, outputs)
        except (KeyError, ValueError):
            raise HTTPException(422, "This operation has no supported LayerNorm group.") from None
        tensor, size = inputs[0], spec["size"]
        if tensor.numel % size or group >= tensor.numel // size:
            raise HTTPException(422, "Choose a valid normalization group.")
        if tensor.value_source == "shape":
            raise HTTPException(409, "Shape runs do not compute numeric statistics.")
        try:
            stats = snapshot_statistics(
                store.snapshot_dir / run.id, tensor, group * size, size, spec["eps"]
            )
        except (OSError, ValueError, EOFError):
            raise HTTPException(
                410, "The tensor snapshot is unavailable or invalid. Run the project again."
            ) from None
        return LayerNormalizationStatistics(
            **stats.model_dump(),
            operation_id=op.id,
            tensor_id=tensor.id,
            group=group,
            start=group * size,
            count=size,
        )

    @app.get(
        "/api/v1/runs/{run_id}/operations/{operation_id}/reduction",
        response_model=ReductionStatistics,
    )
    def reduction_statistics(run_id: str, operation_id: str, output_index: int = Query(ge=0)):
        run = get_run(run_id)
        op = next((op for op in run.trace.operations if op.id == operation_id), None)
        if op is None:
            raise HTTPException(404, "Operation not found")
        try:
            if op.status != "ok" or op.mutations:
                raise ValueError()
            inputs = [run.trace.tensors[i] for i in op.inputs]
            outputs = [run.trace.tensors[i] for i in op.outputs]
            spec = reduction_spec(op.kind, op.arguments, inputs, outputs)
        except (KeyError, ValueError, TypeError):
            raise HTTPException(422, "This operation has no supported reduction group.") from None
        tensor = inputs[0]
        if output_index >= outputs[0].numel:
            raise HTTPException(422, "Choose a valid reduction output cell.")
        if tensor.value_source == "shape":
            raise HTTPException(409, "Shape runs do not compute numeric statistics.")
        if not supports_reference(tensor, outputs[0]):
            raise HTTPException(422, "Reference arithmetic does not emulate dtype conversions.")
        try:
            stats = snapshot_reduction(
                store.snapshot_dir / run.id, op.kind, op.arguments, inputs, outputs, output_index
            )
        except (OSError, ValueError, EOFError, TypeError, OverflowError):
            raise HTTPException(
                410, "The tensor snapshot is unavailable or invalid. Run the project again."
            ) from None
        return ReductionStatistics(
            **stats,
            run_id=run.id,
            operation_id=op.id,
            tensor_id=tensor.id,
            output_index=output_index,
            count=spec["size"],
        )

    @app.get(
        "/api/v1/runs/{run_id}/operations/{operation_id}/softmax",
        response_model=SoftmaxStatistics,
    )
    def softmax_statistics(run_id: str, operation_id: str, group: int = Query(ge=0)):
        run = get_run(run_id)
        op = next((op for op in run.trace.operations if op.id == operation_id), None)
        if op is None:
            raise HTTPException(404, "Operation not found")
        try:
            if op.kind != "softmax" or op.status != "ok" or op.mutations:
                raise ValueError()
            inputs = [run.trace.tensors[i] for i in op.inputs]
            outputs = [run.trace.tensors[i] for i in op.outputs]
            spec = softmax_spec(op.arguments, inputs, outputs)
        except (KeyError, ValueError, TypeError):
            raise HTTPException(422, "This operation has no supported softmax group.") from None
        if group >= spec["groups"]:
            raise HTTPException(422, "Choose a valid softmax group.")
        tensor = inputs[0]
        if tensor.value_source == "shape":
            raise HTTPException(409, "Shape runs do not compute numeric statistics.")
        try:
            stats = snapshot_softmax(
                store.snapshot_dir / run.id, op.arguments, inputs, outputs, group
            )
        except (OSError, ValueError, EOFError, TypeError, OverflowError):
            raise HTTPException(
                410, "The tensor snapshot is unavailable or invalid. Run the project again."
            ) from None
        return SoftmaxStatistics(
            **stats,
            run_id=run.id,
            operation_id=op.id,
            tensor_id=tensor.id,
            group=group,
            count=spec["size"],
        )

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

    def tensor_array(run_id: str, tensor_id: str):
        """A recorded tensor's values as an array in its shape, wherever they live."""
        run = store.run(run_id)
        tensor = run.trace.tensors.get(tensor_id) if run else None
        if tensor is None:
            raise HTTPException(404, "Tensor not found")
        if tensor.value_source == "shape":
            raise HTTPException(409, "Shape runs do not compute numeric values.")
        try:
            if tensor.value_source == "paged":
                return snapshot_array(store.snapshot_dir / run.id, tensor.id)
            return analysis.as_array(tensor.values).reshape(tensor.shape)
        except (OSError, ValueError):
            raise HTTPException(
                410, "The tensor snapshot is no longer available. Run the project again."
            ) from None

    def plane_axes(array, row: int, column: int, fixed: str):
        """Validated plane axes (-1 for none) and the fixed coordinate of every axis."""
        try:
            coordinates = [int(part) for part in fixed.split(",")] if fixed else []
            if len(coordinates) != array.ndim or any(
                not 0 <= c < size for c, size in zip(coordinates, array.shape)
            ):
                raise ValueError()
            axes = [None if axis == -1 else axis for axis in (row, column)]
            if any(a is not None and not 0 <= a < array.ndim for a in axes) or (
                axes[0] is not None and axes[0] == axes[1]
            ):
                raise ValueError()
        except ValueError:
            raise HTTPException(
                422, "Give two different plane axes and one coordinate per axis."
            ) from None
        return axes[0], axes[1], coordinates

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/search")
    def tensor_search(
        run_id: str,
        tensor_id: str,
        test: str,
        value: float = 0.0,
        high: float = 0.0,
        magnitude: bool = False,
        closed: bool = False,
        limit: int = Query(5000, ge=1, le=20000),
    ):
        """Where the values pass a comparison, or fall in a histogram bar."""
        if test not in analysis.TESTS:
            raise HTTPException(422, f"Use one of: {', '.join(sorted(analysis.TESTS))}.")
        array = tensor_array(run_id, tensor_id)
        return analysis.search(array.reshape(-1), test, value, high, magnitude, closed, limit)

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/landmarks")
    def tensor_landmarks(run_id: str, tensor_id: str):
        """The first smallest and largest finite values, and the non-finite ones."""
        return analysis.landmarks(tensor_array(run_id, tensor_id).reshape(-1))

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/margins")
    def tensor_margins(
        run_id: str, tensor_id: str, row: int, column: int, fixed: str = "", reduce: str = "sum"
    ):
        """Row and column reductions of the plane through `fixed`."""
        if reduce not in analysis.REDUCTIONS:
            raise HTTPException(422, "Reduce by sum, mean, max or min.")
        array = tensor_array(run_id, tensor_id)
        row_axis, column_axis, coordinates = plane_axes(array, row, column, fixed)
        try:
            return analysis.margins(array, row_axis, column_axis, coordinates, reduce)
        except ValueError as error:
            raise HTTPException(413, str(error)) from None

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/region")
    def tensor_region(
        run_id: str,
        tensor_id: str,
        row: int,
        column: int,
        rows: str,
        columns: str,
        fixed: str = "",
    ):
        """Totals over a selected rectangle of the plane through `fixed`."""
        array = tensor_array(run_id, tensor_id)
        row_axis, column_axis, coordinates = plane_axes(array, row, column, fixed)
        try:
            r0, r1 = (int(part) for part in rows.split(","))
            c0, c1 = (int(part) for part in columns.split(","))
            height = array.shape[row_axis] if row_axis is not None else 1
            width = array.shape[column_axis] if column_axis is not None else 1
            if not (0 <= r0 <= r1 < height and 0 <= c0 <= c1 < width):
                raise ValueError()
        except ValueError:
            raise HTTPException(
                422, "Give the first and last row and column of the rectangle."
            ) from None
        return analysis.region(array, row_axis, column_axis, coordinates, (r0, r1), (c0, c1))

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/axis")
    def tensor_axis(run_id: str, tensor_id: str, axis: int):
        """Statistics of each index along one axis, e.g. per channel."""
        array = tensor_array(run_id, tensor_id)
        if not 0 <= axis < array.ndim:
            raise HTTPException(422, f"Choose an axis from 0 to {array.ndim - 1}.")
        try:
            return analysis.axis_profile(array, axis)
        except ValueError as error:
            raise HTTPException(413, str(error)) from None

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/thumbnails")
    def tensor_thumbnails(
        run_id: str, tensor_id: str, row: int, column: int, axis: int, fixed: str = ""
    ):
        """A small picture of the plane at each index of a hidden axis."""
        array = tensor_array(run_id, tensor_id)
        row_axis, column_axis, coordinates = plane_axes(array, row, column, fixed)
        if not 0 <= axis < array.ndim or axis in (row_axis, column_axis):
            raise HTTPException(422, "Choose a hidden axis, not a plane axis.")
        try:
            return analysis.thumbnails(array, row_axis, column_axis, coordinates, axis)
        except ValueError as error:
            raise HTTPException(413, str(error)) from None

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/sums")
    def tensor_sums(run_id: str, tensor_id: str, axis: int, value: float):
        """Whether every sum over an axis is `value`, for a sums contract."""
        array = tensor_array(run_id, tensor_id)
        if not -array.ndim <= axis < array.ndim:
            raise HTTPException(422, f"{tensor_id} has no axis {axis}.")
        return analysis.sums(array, axis, value)

    @app.get("/api/v1/runs/{run_id}/tensors/{tensor_id}/npy")
    def tensor_npy(run_id: str, tensor_id: str):
        """The recorded values as a .npy file, for numpy.load or torch.from_numpy."""
        array = tensor_array(run_id, tensor_id)
        tensor = store.run(run_id).trace.tensors[tensor_id]
        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in tensor.name) or "tensor"
        return Response(
            analysis.npy_bytes(array, tensor.dtype),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{name}.npy"'},
        )

    class ComparePairs(BaseModel):
        other_run: str
        #: (this run's tensor, the earlier run's matching tensor) pairs.
        pairs: list[tuple[str, str]] = Field(max_length=400)

    @app.post("/api/v1/runs/{run_id}/compare")
    def compare_runs(run_id: str, body: ComparePairs):
        """Per-tensor change summaries against an earlier run, null where unknown."""
        results = []
        for tensor_id, other_id in body.pairs:
            try:
                now = tensor_array(run_id, tensor_id)
                before = tensor_array(body.other_run, other_id)
            except HTTPException:
                results.append(None)
                continue
            results.append(analysis.compare(now, before) if now.shape == before.shape else None)
        return {"results": results}

    return app


app = create_app()
