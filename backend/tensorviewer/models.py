import ast
import json
import keyword
from math import prod
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class UploadedTensor(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    file_name: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    shape: list[int] = Field(min_length=1, max_length=6)
    dtype: Literal["float32", "float64", "int64"]
    byte_count: int = Field(gt=0)


class InputSpec(BaseModel):
    shape: list[int] = Field(default_factory=lambda: [1, 3, 8], min_length=1, max_length=6)
    generator: Literal["arange", "random", "ones", "zeros", "uploaded"] = "arange"
    uploaded: UploadedTensor | None = None
    dtype: Literal["float32", "float64", "int64"] = "float32"
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    # Keep old projects' RNG behavior; new UI inputs opt into an independent stream.
    random_stream: Literal["model", "input"] = "model"
    axis_names: list[str] = Field(default_factory=lambda: ["batch", "tokens", "features"])

    @model_validator(mode="after")
    def small_positive_tensor(self):
        if (self.generator == "uploaded") != (self.uploaded is not None):
            raise ValueError(
                "Uploaded inputs require a saved tensor file; generated inputs cannot reference one."
            )
        if self.uploaded and (
            self.shape != self.uploaded.shape or self.dtype != self.uploaded.dtype
        ):
            raise ValueError(
                "An uploaded input's shape and dtype must match its file. Transform it in the model instead."
            )
        if any(n < 1 for n in self.shape) or prod(self.shape) > 2**40:
            raise ValueError("Use positive dimensions with at most 2^40 logical elements.")
        if self.axis_names and len(self.axis_names) != len(self.shape):
            raise ValueError("Supply one axis name per dimension, or an empty list.")
        if self.generator == "random" and self.dtype == "int64":
            raise ValueError("Random normal inputs require a floating-point dtype.")
        return self


class InputFixtureDraft(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    input: InputSpec
    capture_mode: Literal["values", "shapes"] = "values"

    @model_validator(mode="after")
    def valid_fixture(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Give this saved input a name.")
        if self.capture_mode == "values" and prod(self.input.shape) > 8_388_608:
            raise ValueError("Use Shapes mode for more than 8,388,608 input elements.")
        return self


class InputFixture(InputFixtureDraft):
    id: str
    created_at: str


class CustomComponentDraft(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=240)
    code: str = Field(min_length=1, max_length=12000)
    class_name: str = Field(pattern=r"^[A-Za-z_]\w*$", max_length=100)
    constructor: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_source(self):
        try:
            tree = ast.parse(self.code)
            compile(tree, "<custom-component>", "exec")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid Python: {exc}") from None
        if not any(isinstance(n, ast.ClassDef) and n.name == self.class_name for n in tree.body):
            raise ValueError(f"Define {self.class_name} as a top-level class.")
        if not self.name.strip():
            raise ValueError("Give this component a name.")
        if len(json.dumps(self.constructor, allow_nan=False)) > 8000:
            raise ValueError("Constructor arguments must fit within 8,000 characters.")
        return self


class CustomComponent(CustomComponentDraft):
    id: str
    created_at: str


class ComponentSpec(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=64)
    kind: str = Field(max_length=40)
    parameters: dict[str, int] = Field(default_factory=dict)
    custom: CustomComponent | None = None
    arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def custom_contract(self):
        if (self.kind == "custom") != (self.custom is not None):
            raise ValueError("Custom components require a saved source snapshot.")
        if self.arguments is not None and len(json.dumps(self.arguments, allow_nan=False)) > 8000:
            raise ValueError("Constructor arguments must fit within 8,000 characters.")
        return self


class Blueprint(BaseModel):
    has_input: bool = False
    components: list[ComponentSpec] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def unique_components(self):
        if len({item.id for item in self.components}) != len(self.components):
            raise ValueError("Component IDs must be unique.")
        return self


class CompositionRequest(BaseModel):
    blueprint: Blueprint
    input: InputSpec = Field(default_factory=InputSpec)
    capture_mode: Literal["values", "shapes"] = "values"


class ComponentStage(BaseModel):
    id: str
    title: str
    input_shape: list[int]
    shape: list[int] | None = None
    axes: list[str] = Field(default_factory=list)
    error: str | None = None


class CompositionPlan(BaseModel):
    validation_required: bool = False
    code: str
    stages: list[ComponentStage]
    valid: bool
    error: str | None = None


class ForwardInput(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=100)
    binding: Literal["positional", "keyword"] = "positional"
    input: InputSpec

    @model_validator(mode="after")
    def valid_name(self):
        if keyword.iskeyword(self.name):
            raise ValueError("Input names cannot be Python keywords.")
        return self


class ProjectDraft(BaseModel):
    blueprint: Blueprint | None = None
    capture_mode: Literal["values", "shapes"] = "values"
    name: str = Field(min_length=1, max_length=100)
    code: str = Field(min_length=1, max_length=500000)
    class_name: str = Field(default="Attention", pattern=r"^[A-Za-z_]\w*$", max_length=100)
    constructor: dict[str, Any] = Field(default_factory=lambda: {"embed_dim": 8, "num_heads": 2})
    input: InputSpec = Field(default_factory=InputSpec)
    input_name: str = Field(default="x", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=100)
    input_binding: Literal["positional", "keyword"] = "positional"
    additional_inputs: list[ForwardInput] = Field(default_factory=list, max_length=7)

    @property
    def forward_inputs(self) -> list[ForwardInput]:
        return [
            ForwardInput(name=self.input_name, binding=self.input_binding, input=self.input),
            *self.additional_inputs,
        ]

    @model_validator(mode="after")
    def bounded_values(self):
        inputs = self.forward_inputs
        if len({item.name for item in inputs}) != len(inputs):
            raise ValueError("Give every forward input a unique name.")
        keyword_seen = False
        for item in inputs:
            if keyword_seen and item.binding == "positional":
                raise ValueError("Place positional inputs before keyword inputs.")
            keyword_seen |= item.binding == "keyword"
        if self.blueprint and (
            self.additional_inputs or self.input_name != "x" or self.input_binding != "positional"
        ):
            raise ValueError(
                "The sequence builder uses forward(x). Use custom code for multiple inputs."
            )
        if self.capture_mode == "values":
            counts = [prod(item.input.shape) for item in inputs]
            if max(counts) > 8_388_608:
                raise ValueError(
                    "Value runs support up to 8,388,608 elements per input. Use Shapes mode for larger tensors."
                )
            if sum(counts) > 32_000_000:
                raise ValueError("Use Shapes mode for more than 32 million total input elements.")
        return self


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
    value_source: Literal["inline", "paged", "shape"] = "inline"
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
    interaction: Literal[
        "mapping",
        "dot_product",
        "normalization",
        "patch_projection",
        "linear_projection",
        "inspect",
    ] = "inspect"
    patch_size: list[int] | None = None
    # output flat index -> first input flat index; only exact, supported mappings.
    mapping: list[int] | None = None
    axis_order: list[int] | None = None
    mapping_rule: Literal["identity", "permutation", "unfold"] | None = None


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


class ModuleCall(BaseModel):
    id: str
    parent_id: str | None = None
    path: str
    module_type: str
    start_index: int
    end_index: int
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class Trace(BaseModel):
    schema_version: Literal["1"] = "1"
    operations: list[Operation] = Field(default_factory=list)
    module_calls: list[ModuleCall] = Field(default_factory=list)
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
