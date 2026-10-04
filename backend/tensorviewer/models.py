import ast
import json
import keyword
from math import isfinite, prod
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class UploadedTensor(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    file_name: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    shape: list[int] = Field(min_length=1, max_length=6)
    dtype: Literal["float32", "float64", "int64"]
    byte_count: int = Field(gt=0)


class InputEdit(BaseModel):
    """One cell of a generated input set to another value: a what-if."""

    index: int = Field(ge=0)
    value: float


Precision = Literal["bfloat16", "float16", "float64"]


class LearnStep(BaseModel):
    """One step of gradient descent on every weight, on one recorded value.

    `direction` 1 raises the value, -1 lowers it: each weight moves by
    rate · direction · ∂ value / ∂ weight before the run is recorded again.
    """

    tensor_id: str = Field(min_length=1, max_length=40)
    index: int = Field(ge=0)
    rate: float = Field(gt=0, le=100)
    direction: Literal[1, -1] = 1
    # Step on log(value) instead: for a probability, one step of cross-entropy.
    log: bool = False
    # How many steps to take, each on the value as the last one left it.
    steps: int = Field(default=1, ge=1, le=100)
    # Train on the whole sentence instead: the tensor holds scores (or, with
    # `log`, probabilities) over the vocabulary at each word, and each word
    # learns to predict the next; the curve is the mean cross-entropy.
    sentence: bool = False


class WhatIfRequest(BaseModel):
    edits: list[InputEdit] = Field(default_factory=list, max_length=64)
    # Run the model and its floating-point input in this dtype instead.
    precision: Precision | None = None
    # Train the weights one step first.
    learn: LearnStep | None = None

    @model_validator(mode="after")
    def something_changes(self):
        if not self.edits and self.precision is None and self.learn is None:
            raise ValueError("A what-if sets an input cell, a precision, or a learning step.")
        return self


class InputSpec(BaseModel):
    shape: list[int] = Field(default_factory=lambda: [1, 3, 8], min_length=1, max_length=6)
    generator: Literal["arange", "random", "ones", "zeros", "uploaded", "image", "text"] = "arange"
    uploaded: UploadedTensor | None = None
    # The sentence behind a "text" input; its token ids are the values.
    text: str | None = Field(default=None, max_length=400)
    dtype: Literal["float32", "float64", "int64"] = "float32"
    seed: int = Field(default=7, ge=0, le=2**32 - 1)
    # Keep old projects' RNG behavior; new UI inputs opt into an independent stream.
    random_stream: Literal["model", "input"] = "model"
    axis_names: list[str] = Field(default_factory=lambda: ["batch", "tokens", "features"])
    # Cells set after the input is made, by flat index; what-if runs only.
    edits: list[InputEdit] = Field(default_factory=list, max_length=64)
    # The dtype the model computes in, when not the input's; what-if runs only.
    precision: Precision | None = None

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
        if self.generator == "image" and (len(self.shape) < 2 or self.dtype == "int64"):
            raise ValueError(
                "A sample image needs height and width as its last two axes and a floating-point dtype."
            )
        for edit in self.edits:
            if edit.index >= prod(self.shape):
                raise ValueError("An input edit names a cell outside the input.")
            if not isfinite(edit.value):
                raise ValueError("An input edit needs a finite value.")
            if self.dtype == "int64" and not float(edit.value).is_integer():
                raise ValueError("An integer input takes whole-number edits.")
        if (self.generator == "text") != (self.text is not None):
            raise ValueError("Sentence inputs need a sentence; other inputs cannot have one.")
        if self.text is not None:
            from .samples import MAX_TOKENS, tokenize

            count = len(tokenize(self.text)[0])
            if not 1 <= count <= MAX_TOKENS:
                raise ValueError(f"Use a sentence with 1 to {MAX_TOKENS} words and symbols.")
            if self.shape != [1, count] or self.dtype != "int64":
                raise ValueError(
                    f"This sentence has {count} tokens: its shape is [1, {count}] and its dtype is int64."
                )
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
    sources: list[str] | None = Field(default=None, min_length=1, max_length=2)

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
        if any(item.id == "input" for item in self.components):
            raise ValueError("The component ID 'input' is reserved for the input tensor.")
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
    source_shapes: list[list[int]] = Field(default_factory=list)
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


class WeightTensor(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    shape: list[int] = Field(max_length=8)
    dtype: str = Field(max_length=32)


class SavedWeights(BaseModel):
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    name: str = Field(min_length=1, max_length=80)
    file_name: str = Field(min_length=1, max_length=200)
    created_at: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    byte_count: int = Field(gt=0)
    tensors: list[WeightTensor] = Field(min_length=1, max_length=2048)


class WeightCheck(BaseModel):
    compatible: bool
    issues: list[str] = Field(default_factory=list)
    tensor_count: int = 0


class ProjectDraft(BaseModel):
    blueprint: Blueprint | None = None
    # Console projects keep the user's statements; code is derived from them.
    script: str | None = Field(default=None, max_length=20000)
    capture_mode: Literal["values", "shapes"] = "values"
    name: str = Field(min_length=1, max_length=100)
    code: str = Field(min_length=1, max_length=500000)
    class_name: str = Field(default="Attention", pattern=r"^[A-Za-z_]\w*$", max_length=100)
    constructor: dict[str, Any] = Field(default_factory=lambda: {"embed_dim": 8, "num_heads": 2})
    input: InputSpec = Field(default_factory=InputSpec)
    input_name: str = Field(default="x", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=100)
    input_binding: Literal["positional", "keyword"] = "positional"
    additional_inputs: list[ForwardInput] = Field(default_factory=list, max_length=7)
    weights: SavedWeights | None = None
    files: dict[str, str] = Field(default_factory=dict)
    entry_path: str = "model.py"
    import_root: str = "."
    repository: dict[str, str] | None = None
    environment: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @property
    def forward_inputs(self) -> list[ForwardInput]:
        return [
            ForwardInput(name=self.input_name, binding=self.input_binding, input=self.input),
            *self.additional_inputs,
        ]

    @model_validator(mode="after")
    def bounded_values(self):
        from .console import console_code
        from .source_projects import valid_path, validate_files

        if self.script is not None:
            if (
                self.blueprint
                or self.files
                or self.entry_path != "model.py"
                or self.import_root != "."
                or self.repository
            ):
                raise ValueError("Console projects hold one script, not a model or source tree.")
            inputs = self.forward_inputs
            self.code = console_code(
                self.script,
                [item.name for item in inputs if item.binding == "positional"],
                [item.name for item in inputs if item.binding == "keyword"],
            )
            self.class_name = "Console"
            self.constructor = {}
        valid_path(self.entry_path)
        valid_path(self.import_root, directory=True)
        if not self.entry_path.endswith(".py"):
            raise ValueError("The entry point must be a Python file.")
        if self.entry_path in self.files:
            raise ValueError("The entry file belongs in code, not additional files.")
        validate_files({**self.files, self.entry_path: self.code})
        if self.blueprint and (
            self.files
            or self.entry_path != "model.py"
            or self.import_root != "."
            or self.environment
        ):
            raise ValueError("Multi-file projects use custom code, not generated builder code.")
        if self.repository and (
            set(self.repository) != {"url", "revision", "subdirectory", "sha256"}
            or sum(len(s) for s in self.repository.values()) > 2048
        ):
            raise ValueError("Invalid repository provenance.")
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


class Histogram(BaseModel):
    """How a tensor's finite values are spread, in equal-width bins."""

    low: float
    high: float
    # One count per bin from low to high; a single bin when all values agree.
    counts: list[int]
    zeros: int
    non_finite: int
    mean: float | None = None
    std: float | None = None


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
    histogram: Histogram | None = None
    role: Literal["input", "parameter", "intermediate"] = "intermediate"


class SourceLocation(BaseModel):
    line: int
    text: str
    file: str | None = None


class PopulationStatistics(BaseModel):
    status: Literal["ok", "non_finite", "overflow"]
    mean: float | None = None
    variance: float | None = None
    denominator: float | None = None


class LayerNormalizationStatistics(PopulationStatistics):
    operation_id: str
    tensor_id: str
    group: int
    start: int
    count: int


class ReductionStatistics(BaseModel):
    run_id: str
    operation_id: str
    tensor_id: str
    output_index: int
    count: int
    status: Literal["ok", "non_finite", "overflow"]
    # Large integer sums remain decimal strings across JSON/JavaScript.
    # A mean can be finite even when its unscaled sum overflows float64.
    sum: int | float | str | None = None
    result: int | float | str | None = None


class SoftmaxStatistics(BaseModel):
    run_id: str
    operation_id: str
    tensor_id: str
    group: int
    count: int
    status: Literal["ok", "non_finite", "all_masked"]
    maximum: float | None = None
    denominator: float | None = None
    masked_count: int | None = None


class Lesson(BaseModel):
    title: str
    summary: str
    detail: str
    category: Literal["layout", "compute", "normalize", "memory", "creation", "generic"]
    interaction: Literal[
        "mapping",
        "dot_product",
        "normalization",
        "patch_projection",
        "linear_projection",
        "broadcast_add",
        "tensor_assembly",
        "convolution",
        "pooling",
        "layer_normalization",
        "relation",
        "activation",
        "reduction",
        "inspect",
    ] = "inspect"
    # A validated cell-to-cell rule evaluated per coordinate by the viewer.
    relation: dict[str, Any] | None = None
    patch_size: list[int] | None = None
    # output flat index -> first input flat index; only exact, supported mappings.
    mapping: list[int] | None = None
    axis_order: list[int] | None = None
    mapping_rule: Literal["identity", "permutation", "unfold", "roll"] | None = None


class TensorMutation(BaseModel):
    before: str
    after: str
    kind: Literal["write", "alias", "metadata"]


class LoopStep(BaseModel):
    """One enclosing loop of an operation and the iteration it ran in."""

    # Identifies one run of a loop: the frame and header, plus the iterations
    # of the loops around it, so each pass of an outer loop is its own run.
    id: str
    line: int
    file: str | None = None
    text: str
    iteration: int


class Operation(BaseModel):
    id: str
    index: int
    kind: str
    function: str
    inputs: list[str]
    outputs: list[str]
    mutations: list[TensorMutation] = Field(default_factory=list)
    arguments: dict[str, Any]
    source: SourceLocation | None
    module: str
    lesson: Lesson
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    loops: list[LoopStep] = Field(default_factory=list)


class RunError(BaseModel):
    type: str
    message: str
    line: int | None = None
    file: str | None = None


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
    runtime: dict[str, str] = Field(default_factory=dict)
    operations: list[Operation] = Field(default_factory=list)
    module_calls: list[ModuleCall] = Field(default_factory=list)
    tensors: dict[str, TensorState] = Field(default_factory=dict)
    input_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    error: RunError | None = None
    stdout: str = ""
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float = 0
    weight_check: WeightCheck | None = None
    # After training steps (a what-if): the value they aimed at, before each
    # step and after the last.
    learn_curve: list[float | None] | None = None


class SensitivityRequest(BaseModel):
    """A result cell whose dependence on the input is wanted."""

    tensor_id: str = Field(min_length=1, max_length=40)
    index: int = Field(ge=0)


class Sensitivity(BaseModel):
    """How much a result cell moves per unit change of each input cell.

    `gradient`: ∂ result / ∂ input, one value per input cell. `embedding`: for
    integer inputs (token ids), the size (L2 norm) of the gradient at the
    embedding each id was looked up as, one value per token.
    """

    tensor_id: str
    index: int
    input_id: str
    kind: Literal["gradient", "embedding"]
    # None where the gradient is not finite.
    values: list[float | None]
    # The result cell's value, as this run computed it.
    value: float | None = None
    error: RunError | None = None


class GradientRequest(BaseModel):
    """A value to differentiate: one cell of a result, or (no index) its sum."""

    tensor_id: str = Field(min_length=1, max_length=40)
    index: int | None = Field(default=None, ge=0)


class GradientFlow(BaseModel):
    """The size (L2 norm) of ∂ target / ∂ each recorded tensor it depends on.

    Tensors the target does not depend on, such as those computed after it,
    are left out; a tensor that is used but whose gradient vanishes reads 0.
    """

    tensor_id: str
    index: int | None = None
    norms: dict[str, float | None] = Field(default_factory=dict)
    error: RunError | None = None


class WatchRequest(BaseModel):
    """A Python expression over a run's named tensors, as they were at a step."""

    expression: str = Field(min_length=1, max_length=500)
    # The step whose state the names read; the end of the run when absent.
    at: str | None = Field(default=None, max_length=40)


class WatchStats(BaseModel):
    min: float
    max: float
    mean: float
    std: float


class Evaluation(BaseModel):
    """A watch expression's value: a tensor summarized, any other value as
    text, or the error it raised."""

    kind: Literal["tensor", "value", "error"]
    text: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    numel: int | None = None
    # The first values, in memory order.
    values: list[float | int | bool | str] | None = None
    stats: WatchStats | None = None
    non_finite: int | None = None
    # The names the expression could use at that step.
    names: list[str] = Field(default_factory=list)


class SeriesRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=500)


class SeriesPoint(BaseModel):
    at: str
    step: int
    line: int | None = None
    value: float | None = None
    error: str | None = None


class WatchSeries(BaseModel):
    """A watch expression at every step where one of its names changed: the
    expression must give one number."""

    points: list[SeriesPoint] = Field(default_factory=list)
    truncated: bool = False
    error: str | None = None


class WeightUpdate(BaseModel):
    """How a weight changed since another run: W − W₀, with its spectrum."""

    norm: float
    # ‖W − W₀‖ / ‖W₀‖
    relative: float | None = None
    singular: list[float] = Field(default_factory=list)
    rank: int | None = None
    full: int | None = None
    effective_rank: float | None = None
    condition: float | None = None


class WeightsRequest(BaseModel):
    # Another run to compare each weight with, by name.
    against: str | None = Field(default=None, max_length=80)


class WeightSpectrum(BaseModel):
    """A weight of the model: its size, its norm, and as a matrix
    ([first axis, everything else]) its singular values."""

    tensor_id: str
    name: str
    shape: list[int]
    numel: int
    norm: float
    # Largest first; the first 64.
    singular: list[float] = Field(default_factory=list)
    # Singular values above float32 rounding of the largest, of the most there can be.
    rank: int | None = None
    full: int | None = None
    # exp(entropy) of the normalized singular values: how many directions it uses.
    effective_rank: float | None = None
    condition: float | None = None
    update: WeightUpdate | None = None


class WeightReport(BaseModel):
    weights: list[WeightSpectrum] = Field(default_factory=list)
    error: str | None = None


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
    # The code and inputs the run used, without its trace: enough to say
    # what changed between runs.
    project: ProjectDraft | None = None


class LatestRun(BaseModel):
    """A project's newest run, for marking projects in lists."""

    project_id: str
    run_id: str
    created_at: str
    operation_count: int
    failed: bool


class Template(BaseModel):
    id: str
    description: str
    project: ProjectDraft
