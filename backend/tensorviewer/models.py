from math import prod
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class InputSpec(BaseModel):
    shape: list[int] = Field(default_factory=lambda: [1, 3, 8], min_length=1, max_length=6)
    generator: Literal["arange", "random", "ones", "zeros"] = "arange"
    dtype: Literal["float32", "float64", "int64"] = "float32"
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    axis_names: list[str] = Field(default_factory=lambda: ["batch", "tokens", "features"])

    @model_validator(mode="after")
    def small_positive_tensor(self):
        if any(n < 1 for n in self.shape) or prod(self.shape) > 4096:
            raise ValueError("Use positive dimensions with at most 4,096 input elements.")
        if self.axis_names and len(self.axis_names) != len(self.shape):
            raise ValueError("Supply one axis name per dimension, or an empty list.")
        if self.generator == "random" and self.dtype == "int64":
            raise ValueError("Random normal inputs require a floating-point dtype.")
        return self


class ProjectDraft(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    code: str = Field(min_length=1, max_length=50000)
    class_name: str = Field(default="Attention", pattern=r"^[A-Za-z_]\w*$", max_length=100)
    constructor: dict[str, Any] = Field(default_factory=lambda: {"embed_dim": 8, "num_heads": 2})
    input: InputSpec = Field(default_factory=InputSpec)


class Project(ProjectDraft):
    id: str
    created_at: str
    updated_at: str


class TensorState(BaseModel):
    id: str
    name: str
    shape: list[int]
    axes: list[str]
    dtype: str
    strides: list[int]
    storage_id: str
    storage_offset: int
    contiguous: bool
    numel: int
    values: list[float | int | str]
    minimum: float | None = None
    maximum: float | None = None
    role: Literal["input", "parameter", "intermediate"] = "intermediate"


class SourceLocation(BaseModel):
    line: int
    text: str


class Lesson(BaseModel):
    title: str
    summary: str
    detail: str
    category: Literal["layout", "compute", "normalize", "memory", "generic"]
    interaction: Literal["mapping", "dot_product", "normalization", "inspect"] = "inspect"
    # output flat index -> first input flat index; only exact, supported mappings.
    mapping: list[int] | None = None
    axis_order: list[int] | None = None


class Operation(BaseModel):
    id: str
    index: int
    kind: str
    function: str
    inputs: list[str]
    outputs: list[str]
    arguments: dict[str, Any]
    source: SourceLocation | None
    module: str
    lesson: Lesson
    status: Literal["ok", "error"] = "ok"
    error: str | None = None


class RunError(BaseModel):
    type: str
    message: str
    line: int | None = None


class Trace(BaseModel):
    schema_version: Literal["1"] = "1"
    operations: list[Operation] = Field(default_factory=list)
    tensors: dict[str, TensorState] = Field(default_factory=dict)
    input_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    error: RunError | None = None
    stdout: str = ""
    duration_ms: float = 0


class Run(BaseModel):
    id: str
    project_id: str
    created_at: str
    project: ProjectDraft
    trace: Trace


class RunSummary(BaseModel):
    id: str
    project_id: str
    created_at: str
    operation_count: int
    failed: bool


class Template(BaseModel):
    id: str
    description: str
    project: ProjectDraft
