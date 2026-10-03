import os
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
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
    InputFixture,
    InputFixtureDraft,
    InputSpec,
    LatestRun,
    LayerNormalizationStatistics,
    Project,
    ProjectDraft,
    ReductionStatistics,
    Run,
    RunSummary,
    SavedWeights,
    SoftmaxStatistics,
    Template,
    Trace,
    WeightCheck,
)
from .operations.normalization import layer_normalization_spec
from .operations.reduction import reduction_spec
from .operations.softmax import softmax_spec
from .reduction_statistics import snapshot_reduction, supports_reference
from .runner import run_project
from .snapshots import read_snapshot, snapshot_array
from .softmax_statistics import snapshot_softmax
from .source_projects import import_git
from .statistics import snapshot_statistics
from .storage import Store
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
