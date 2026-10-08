"""Record high-level eager PyTorch calls without rewriting user code."""

import ast
import inspect
import math
import re
import time
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch.overrides import TorchFunctionMode

from .loops import LoopTracker, file_loops
from .models import (
    Histogram,
    Lesson,
    ModuleCall,
    Operation,
    SourceLocation,
    TensorMutation,
    TensorState,
    Trace,
)
from .mutations import TensorObservation, affected_tensors
from .operations import describe_operation
from .operations.assembly import assembly_axes
from .operations.creation import CREATION
from .operations.pooling import POOL_ARGUMENTS
from .operations.reduction import reduction_axes
from .operations.relations import (
    BINARY,
    REDUCTIONS,
    index_argument,
    index_text,
    relation_axes,
    replayed_selection,
)
from .snapshots import save_snapshot

MAX_OPERATIONS = 256
MAX_TENSOR_ELEMENTS = 8_388_608
MAX_CAPTURED_ELEMENTS = 32_000_000
INLINE_ELEMENTS = 4096


class TraceLimitError(RuntimeError):
    pass


HISTOGRAM_BINS = 24


def value_histogram(tensor: torch.Tensor) -> Histogram | None:
    """Equal-width bins over a tensor's finite values, with zeros and NaN/Inf counted."""
    if tensor.dtype == torch.bool or tensor.is_complex() or tensor.numel() == 0:
        return None
    data = tensor.detach().reshape(-1).to(torch.float64)
    finite = data[torch.isfinite(data)]
    non_finite = data.numel() - finite.numel()
    if not finite.numel():
        return Histogram(low=0, high=0, counts=[], zeros=0, non_finite=non_finite)
    low, high = finite.min().item(), finite.max().item()
    counts = (
        [finite.numel()]
        if low == high
        else torch.histc(finite, bins=HISTOGRAM_BINS, min=low, max=high).long().tolist()
    )
    return Histogram(
        low=low,
        high=high,
        counts=counts,
        zeros=int((finite == 0).sum().item()),
        non_finite=non_finite,
        mean=finite.mean().item(),
        std=finite.std(unbiased=False).item(),
    )


def tensors_in(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item)


def replace_in(value, old: torch.Tensor, new: torch.Tensor):
    """`value` with the tensor `old` swapped for `new`, inside tuples and lists."""
    if value is old:
        return new
    if isinstance(value, list):
        return [replace_in(item, old, new) for item in value]
    if isinstance(value, tuple):
        parts = [replace_in(item, old, new) for item in value]
        # A named tuple takes its fields; torch's return types take a sequence.
        return type(value)(*parts) if hasattr(value, "_fields") else type(value)(parts)
    return value


def knocked_value(target: torch.Tensor, rule, patch: torch.Tensor | None) -> torch.Tensor:
    """What a knockout puts in place of `target`: zero, its mean, or a patch,
    in the whole of it or in one slice along an axis."""
    if rule.axis is not None and (
        rule.axis >= target.ndim or rule.index >= target.shape[rule.axis]
    ):
        raise ValueError(
            f"Step {rule.step + 1}'s result is {list(target.shape)}: it has no slice "
            f"{rule.index} along axis {rule.axis}."
        )
    if rule.mode == "zero":
        fill = torch.zeros_like(target)
    elif rule.mode == "mean":
        if not target.dtype.is_floating_point:
            raise ValueError(
                "Only a floating-point result has a mean to put in; knock it out to zero instead."
            )
        mean = target.mean() if rule.axis is None else target.mean(dim=rule.axis, keepdim=True)
        fill = mean.expand_as(target)
    else:
        if patch is None or list(patch.shape) != list(target.shape):
            raise ValueError(
                f"The other run's step {rule.step + 1} made "
                f"{list(patch.shape) if patch is not None else 'nothing'}, "
                f"not {list(target.shape)}: only a result of the same shape can be patched in."
            )
        fill = patch.to(dtype=target.dtype)
    if rule.axis is None:
        return fill.clone()
    chosen = torch.zeros(target.shape[rule.axis], dtype=torch.bool)
    chosen[rule.index] = True
    chosen = chosen.reshape([-1 if axis == rule.axis else 1 for axis in range(target.ndim)])
    return torch.where(chosen, fill, target)


def plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# Unassigned results of these operators are named as written: `~mask`.
PREFIX_OPERATORS = {"__invert__": "~", "__neg__": "-"}

# Operations whose same-shape result keeps its input's axes.
SAME_AXES = {
    "contiguous",
    "clone",
    "softmax",
    "log_softmax",
    "layer_norm",
    "gelu",
    "relu",
    "sigmoid",
    "tanh",
    "div",
    "mul",
    "add",
    "sub",
    "silu",
    "exp",
    "log",
    "neg",
    "abs",
    "sqrt",
    "rsqrt",
    "pow",
    "clamp",
    "dropout",
    "masked_fill",
    "tril",
    "triu",
    "bool",
    "float",
    "to",
    "type_as",
    "detach",
    "__invert__",
    "logical_not",
}


def feature_axes(kind, source, output):
    """Names for an operation that keeps its input's axes.

    `linear` maps the last axis to new features and keeps the others; an
    `embedding` lookup appends a feature axis to the indices' axes; `gather`
    and `index_select` pick positions along an axis without changing rank.
    """
    if kind == "linear" and source.shape[:-1] == output.shape[:-1] and source.shape:
        return source.axes.copy()
    if kind == "embedding" and output.shape[:-1] == source.shape:
        return [*source.axes, "features"]
    if kind in {"gather", "index_select"} and len(output.shape) == len(source.shape):
        # Picking along one axis keeps every axis's meaning: chosen experts
        # are still experts.
        return source.axes.copy()
    return None


def arguments_for(kind, args, kwargs):
    result = {k: plain(v) for k, v in kwargs.items()}
    rest = list(args[1:])
    if kind == "einsum" and args and isinstance(args[0], str):
        # The equation comes first; the operands follow it.
        result["equation"] = args[0]
    elif kind == "__getitem__" and rest:
        result["index"] = index_argument(rest[0])
    elif kind in CREATION:
        # torch.arange(9) has no tensor operand: every positional argument counts.
        result["args"] = plain(list(args))
    elif kind in {"reshape", "view", "permute", "expand", "repeat", "flip"}:
        key = {"permute": "dims", "flip": "dims", "expand": "size", "repeat": "repeats"}.get(
            kind, "shape"
        )
        if rest:
            result[key] = plain(
                rest[0] if len(rest) == 1 and isinstance(rest[0], (tuple, list)) else rest
            )
    else:
        names = {
            **{name: ["other"] for name in BINARY},
            **{name: ["dim", "keepdim"] for name in REDUCTIONS},
            "std": ["dim", "unbiased"],
            "var": ["dim", "unbiased"],
            "cumsum": ["dim"],
            "cumprod": ["dim"],
            "sort": ["dim", "descending"],
            "argsort": ["dim", "descending"],
            "topk": ["k", "dim", "largest", "sorted"],
            "einsum": ["equation"],
            "pad": ["pad", "mode", "value"],
            "swapaxes": ["dim0", "dim1"],
            "swapdims": ["dim0", "dim1"],
            "movedim": ["source", "destination"],
            "moveaxis": ["source", "destination"],
            "unflatten": ["dim", "sizes"],
            "where": ["input", "other"],
            "gather": ["dim", "index"],
            "index_select": ["dim", "index"],
            "tril": ["diagonal"],
            "triu": ["diagonal"],
            "clamp": ["min", "max"],
            "clip": ["min", "max"],
            **POOL_ARGUMENTS,
            "transpose": ["dim0", "dim1"],
            "softmax": ["dim", "dtype"],
            "normalize": ["p", "dim", "eps"],
            "log_softmax": ["dim", "dtype"],
            "one_hot": ["num_classes"],
            "batch_norm": [
                "running_mean",
                "running_var",
                "weight",
                "bias",
                "training",
                "momentum",
                "eps",
            ],
            "cross_entropy": ["target", "weight"],
            "nll_loss": ["target", "weight"],
            "repeat_interleave": ["repeats", "dim"],
            "unfold": ["dimension", "size", "step"],
            "flatten": ["start_dim", "end_dim"],
            "squeeze": ["dim"],
            "unsqueeze": ["dim"],
            "mean": ["dim", "keepdim"],
            "sum": ["dim", "keepdim"],
            "div": ["other"],
            "mul": ["other"],
            "add": ["other"],
            "sub": ["other"],
            "conv1d": ["weight", "bias", "stride", "padding", "dilation", "groups"],
            "conv2d": ["weight", "bias", "stride", "padding", "dilation", "groups"],
            "layer_norm": ["normalized_shape", "weight", "bias", "eps"],
            "gelu": ["approximate"],
            "roll": ["shifts", "dims"],
            "masked_fill": ["mask", "value"],
            "cat": ["dim"],
            "concat": ["dim"],
            "concatenate": ["dim"],
            "stack": ["dim"],
            "split": ["split_size_or_sections", "dim"],
            "chunk": ["chunks", "dim"],
            "unbind": ["dim"],
        }.get(kind, [])
        for i, value in enumerate(rest):
            if not isinstance(value, torch.Tensor):
                result[names[i] if i < len(names) else f"arg{i + 1}"] = plain(value)
    if kind == "layer_norm":
        # Shape alone cannot distinguish scale-only from bias-only calls.
        for position, name in ((2, "weight"), (3, "bias")):
            result[name] = plain(args[position] if len(args) > position else kwargs.get(name))
    if kind == "batch_norm":
        # Every statistic and parameter is optional; record which were supplied.
        for position, name in enumerate(["running_mean", "running_var", "weight", "bias"], 1):
            result[name] = plain(args[position] if len(args) > position else kwargs.get(name))
    return result


def operands_for(kind, args, kwargs):
    """Preserve semantic operand order even when keyword arguments are reordered."""
    names = {
        "matmul": ["input", "other"],
        "mm": ["input", "mat2"],
        "bmm": ["input", "mat2"],
        "linear": ["input", "weight", "bias"],
        "layer_norm": ["input", "normalized_shape", "weight", "bias", "eps"],
        "conv1d": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "conv2d": ["input", "weight", "bias", "stride", "padding", "dilation", "groups"],
        "add": ["input", "other"],
        "sub": ["input", "other"],
        "mul": ["input", "other"],
        "div": ["input", "other"],
        "masked_fill": ["input", "mask", "value"],
        "cat": ["tensors", "dim"],
        "concat": ["tensors", "dim"],
        "concatenate": ["tensors", "dim"],
        "stack": ["tensors", "dim"],
        "where": ["condition", "input", "other"],
        "gather": ["input", "dim", "index"],
        "index_select": ["input", "dim", "index"],
        "embedding": ["input", "weight"],
        "batch_norm": ["input", "running_mean", "running_var", "weight", "bias"],
    }.get(kind, ["input"])
    ordered = list(args)
    ordered.extend(kwargs[name] for name in names[len(args) :] if name in kwargs)
    ordered.extend(value for name, value in kwargs.items() if name not in names)
    return tensors_in(ordered)


def assigned_names(target) -> list[str] | None:
    """Names bound by `a = …` or `a, b = …`; other targets are not variables."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)) and all(
        isinstance(item, ast.Name) for item in target.elts
    ):
        return [item.id for item in target.elts]
    return None


def register_assignments(tree: ast.AST, table: dict) -> None:
    """Index `name = value` statements by every line their value covers."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if isinstance(node.value, ast.Constant):
            # A literal evaluates no tensor expression; embedded component
            # source is such a literal and spans the lines of its own code.
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        names = assigned_names(target)
        if not names:
            continue
        value = node.value
        span = (value.lineno, value.col_offset, value.end_lineno, value.end_col_offset)
        for line in range(value.lineno, (value.end_lineno or value.lineno) + 1):
            table.setdefault(line, []).append((span, names))


def within(span, outer) -> bool:
    return (span[0], span[1]) >= (outer[0], outer[1]) and (span[2], span[3]) <= (
        outer[2],
        outer[3],
    )


class Recorder(TorchFunctionMode):
    def __init__(
        self,
        code: str,
        filename: str,
        model: torch.nn.Module,
        snapshot_dir=None,
        shapes=False,
        source_files=None,
        knockout=None,
        patch=None,
        values=True,
    ):
        super().__init__()
        self.trace = Trace()
        # An analysis pass (the logit lens's, a causal trace's) reads only the
        # steps and the live tensors, never the recorded values: without them
        # a pass skips copying, summarizing and saving every tensor's values.
        self.values = values
        # A what-if inside the model: one step's result replaced as it is made.
        self.knockout = knockout
        self.patch = patch
        self.knocked = False
        # The tensors the model returned, as the worker measures them.
        self.results: list[torch.Tensor] = []
        self.snapshot_dir = snapshot_dir
        self.shapes = shapes
        self.filename = filename
        self.lines = code.splitlines()
        self.live: dict[int, tuple[torch.Tensor, TensorObservation, str]] = {}
        self.storage_ids: dict[int, str] = {}
        # Keep storages alive so allocator address reuse cannot imply false aliasing.
        self.storages: list = []
        self.captured_elements = 0
        self.module_stack: list[str] = []
        self.call_stack: list[ModuleCall] = []
        self.call_modules: list[torch.nn.Module] = []
        self.call_count = 0
        self.suspended = False
        self.hooks = []
        self.parameters = {id(t): name for name, t in model.named_parameters()}
        self.parameters.update({id(t): name for name, t in model.named_buffers()})
        self.trace.buffer_names = [name for name, _ in model.named_buffers()]
        self.names = {}
        # What an unassigned output is called: `x[:, 0]` reads better than
        # `__getitem__`. Keyed by operation id; other kinds use their kind.
        self.default_names: dict[str, str] = {}
        self.default_templates: dict[str, tuple[str, str, str]] = {}
        # Where the user expression behind each operation sits: frame identity,
        # bytecode offset, and source span (line, column, end line, end column).
        self.position = None
        self.positions: list = []
        self.code_positions: dict = {}
        self.module_assignments: dict = {}
        self.source_files = source_files or {}
        self.file_names = {}
        for _, (path, content) in self.source_files.items():
            names = {}
            try:
                file_tree = ast.parse(content)
            except SyntaxError:
                # An unused file may target a different Python version. Normal
                # imports still report syntax errors in any executed source.
                continue
            register_assignments(file_tree, names)
            self.file_names[path] = names
        tree = ast.parse(code)
        self.component_lines = {}
        embedded = []
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and re.fullmatch(r"_tv_source_\d+", node.targets[0].id)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                source = node.value.value
                try:
                    parsed = ast.parse(source)
                except SyntaxError:
                    continue
                offset = node.lineno - 1
                self.component_lines.update(
                    {offset + i: line for i, line in enumerate(source.splitlines(), 1)}
                )
                embedded.append(ast.increment_lineno(parsed, offset))
        for root in [tree, *embedded]:
            register_assignments(root, self.names)
        loop_files = {
            name: found
            for name, found in [
                (filename, file_loops(code)),
                *(
                    (name, file_loops(content, path))
                    for name, (path, content) in self.source_files.items()
                ),
            ]
            if found
        }
        self.loops = LoopTracker(loop_files)
        for name, module in model.named_modules():
            self.hooks.append(
                module.register_forward_pre_hook(
                    self._enter(name or model.__class__.__name__), with_kwargs=True
                )
            )
            self.hooks.append(
                module.register_forward_hook(self._leave, always_call=True, with_kwargs=True)
            )

    def __enter__(self):
        result = super().__enter__()
        self.loops.start()
        return result

    def __exit__(self, *exc):
        self.loops.stop()
        return super().__exit__(*exc)

    @contextmanager
    def pause_capture(self):
        previous = self.suspended
        self.suspended = True
        try:
            yield
        finally:
            self.suspended = previous

    def _enter(self, name):
        def hook(module, args, kwargs):
            call = ModuleCall(
                id=f"call{self.call_count}",
                parent_id=self.call_stack[-1].id if self.call_stack else None,
                path=name,
                module_type=module.__class__.__name__,
                start_index=len(self.trace.operations),
                end_index=len(self.trace.operations),
            )
            self.call_count += 1
            source = self.source()
            binding = self.assignment(source) if source else None
            if binding:
                self.module_assignments[call.id] = (self.position, binding[1])
            self.call_stack.append(call)
            self.call_modules.append(module)
            self.trace.module_calls.append(call)
            self.module_stack.append(name)
            # Snapshotting in a hook must not record detach/reshape as model operations.
            with self.pause_capture():
                call.inputs = [self.capture(t) for t in tensors_in((args, kwargs))]

        return hook

    def _leave(self, _module, _args, _kwargs, output):
        # An earlier user pre-hook can raise before our enter hook runs.
        # Its always-call exit must not pop the enclosing module's invocation.
        if not self.call_stack or self.call_modules[-1] is not _module:
            return
        call = self.call_stack.pop()
        self.call_modules.pop()
        self.module_stack.pop()
        call.end_index = len(self.trace.operations)
        if call.start_index == call.end_index:
            self.trace.module_calls.remove(call)
            return
        with self.pause_capture():
            call.outputs = [self.capture(t) for t in tensors_in(output)]

    def assignment(self, source):
        """The assignment whose value contains the expression being evaluated.

        Returns (value span, names, exact). `exact` means the expression is the
        whole assigned value, so its result is the variable itself. In
        `scores = q @ k.transpose(-2, -1)` only the matrix product is exact.
        """
        table = self.file_names.get(source.file, self.names)
        span = self.position[2] if self.position else None
        for value, names in table.get(source.line, []):
            if span is None:
                # No column information: fall back to the statement's first line.
                if value[0] == source.line:
                    return value, names, False
            elif within(span, value):
                return value, names, span == value
        return None

    def backfill_axes(self, kind, lesson, input_ids, output, value):
        """Carry a statement's `# axes:` names back through its intermediates.

        Same-shape elementwise steps keep their input's axes and a transpose
        or permute reorders them, so each such step's input is named too. The
        walk stays inside the statement and stops at any other operation.
        """
        frame = self.position[0] if self.position else None
        producers = {tensor_id: op for op in self.trace.operations for tensor_id in op.outputs}
        axes = output.axes
        while input_ids:
            source = self.trace.tensors[input_ids[0]]
            if kind in SAME_AXES and source.shape == output.shape:
                named = list(axes)
            elif lesson.axis_order and len(lesson.axis_order) == len(axes):
                named = [""] * len(axes)
                for position, origin in enumerate(lesson.axis_order):
                    named[origin] = axes[position]
            else:
                return
            producer = producers.get(source.id)
            here = self.positions[producer.index] if producer else None
            if (
                not producer
                or not here
                or here[0] != frame
                or here[3] != value
                or source.role != "intermediate"
                or not all(axis.startswith("axis ") for axis in source.axes)
            ):
                return
            source.axes = named
            kind, lesson, input_ids, output, axes = (
                producer.kind,
                producer.lesson,
                producer.inputs,
                source,
                named,
            )

    def name_intermediates(self):
        """Give a variable's name only to the operation that produced its value.

        An expression inside an assigned value keeps the name when nothing
        later in the same evaluation completes that value, such as the taken
        branch of `y = f(x) if flag else g(x)`.
        """
        operations = self.trace.operations
        count = min(len(operations), len(self.positions))
        # A library module can perform several operations at one user call
        # site. Their source spans are identical, but only the tensors the
        # module actually returns belong to the assignment at that call site.
        # Use invocation ranges as well as spans so repeated calls in a loop
        # retain their own bindings.
        calls = {call.id: call for call in self.trace.module_calls}
        for call in self.trace.module_calls:
            binding = self.module_assignments.get(call.id)
            if not binding or not call.outputs:
                continue
            position, names = binding
            parent = calls.get(call.parent_id)
            while parent:
                parent_binding = self.module_assignments.get(parent.id)
                if parent_binding and parent_binding[0] == position:
                    break
                parent = calls.get(parent.parent_id)
            if parent:
                # The enclosing module returns the value of this expression;
                # a child may return an intermediate value instead.
                continue
            if len(names) == len(call.outputs):
                assigned = dict(zip(call.outputs, names))
            elif len(names) == 1:
                assigned = {
                    tensor_id: names[0] if i == 0 else f"{names[0]}[{i}]"
                    for i, tensor_id in enumerate(call.outputs)
                }
            else:
                assigned = {}
            for i in range(call.start_index, min(call.end_index, count)):
                here = self.positions[i]
                if not here or here[:3] != position:
                    continue
                operation = operations[i]
                for j, tensor_id in enumerate(operation.outputs):
                    tensor = self.trace.tensors[tensor_id]
                    if tensor.role == "intermediate":
                        default = self.default_names.get(operation.id, operation.kind)
                        tensor.name = assigned.get(
                            tensor_id,
                            default if j == 0 else f"{default}[{j}]",
                        )
        for i in range(count):
            here = self.positions[i]
            if not here or here[3] is None or here[4]:
                continue
            frame, offset, _, value, _ = here
            for later in self.positions[i + 1 : count]:
                if not later or later[0] != frame:
                    continue
                if later[3] == value and later[1] > offset:
                    kind = self.default_names.get(operations[i].id, operations[i].kind)
                    for j, tensor_id in enumerate(operations[i].outputs):
                        tensor = self.trace.tensors[tensor_id]
                        if tensor.role == "intermediate":
                            tensor.name = kind if j == 0 else f"{kind}[{j}]"
                break
        self.positions = []
        self.final_default_names()

    def final_default_names(self):
        """Rebuild `x[:, 0]` and `~mask` names from their operands' final names.

        Operations run in order, so an operand named this way (`sum[:, 0]` of
        an unassigned `sum`) is settled before the names built on it.
        """
        for operation in self.trace.operations:
            template = self.default_templates.get(operation.id)
            if not template:
                continue
            prefix, operand, suffix = template
            stale = self.default_names[operation.id]
            fresh = f"{prefix}{self.trace.tensors[operand].name}{suffix}"
            self.default_names[operation.id] = fresh
            for j, tensor_id in enumerate(operation.outputs):
                tensor = self.trace.tensors[tensor_id]
                if tensor.name == (stale if j == 0 else f"{stale}[{j}]"):
                    tensor.name = fresh if j == 0 else f"{fresh}[{j}]"

    def close(self):
        self.name_intermediates()
        for hook in self.hooks:
            hook.remove()

    def source(self):
        frame = inspect.currentframe()
        constructing = False
        try:
            while frame:
                # Parameter initialization inside a module constructor is setup,
                # not a step of the traced computation.
                constructing = constructing or (
                    frame.f_code.co_name == "__init__"
                    and isinstance(frame.f_locals.get("self"), torch.nn.Module)
                )
                known = (
                    frame.f_code.co_filename in self.source_files
                    or frame.f_code.co_filename == self.filename
                )
                if known and constructing:
                    return None
                if known:
                    code = frame.f_code
                    if code not in self.code_positions:
                        self.code_positions[code] = list(code.co_positions())
                    index = frame.f_lasti // 2
                    spans = self.code_positions[code]
                    line, end_line, column, end_column = (
                        spans[index] if 0 <= index < len(spans) else (None,) * 4
                    )
                    span = (
                        None
                        if None in (line, end_line, column, end_column)
                        else (line, column, end_line, end_column)
                    )
                    self.position = (id(frame), frame.f_lasti, span)
                if frame.f_code.co_filename in self.source_files:
                    path, content = self.source_files[frame.f_code.co_filename]
                    line = frame.f_lineno
                    return SourceLocation(
                        line=line, file=path, text=content.splitlines()[line - 1].strip()
                    )
                if frame.f_code.co_filename == self.filename:
                    line = frame.f_lineno
                    return SourceLocation(
                        line=line,
                        text=self.component_lines.get(line, self.lines[line - 1]).strip(),
                    )
                frame = frame.f_back
        finally:
            del frame
        return None

    def capture(self, tensor, name="tensor", axes=None, role="intermediate", force=False):
        if (
            tensor.device.type not in ({"meta"} if self.shapes else {"cpu"})
            or tensor.layout != torch.strided
            or tensor.is_complex()
        ):
            raise TraceLimitError("This first version supports dense, real CPU tensors only.")
        previous = self.live.get(id(tensor))
        observation = TensorObservation.read(tensor)
        if previous and previous[1] == observation and not force:
            return previous[2]
        if previous and not force:
            warning = "A tensor changed outside a recorded PyTorch operation. Its new state is captured, but its mutation dependency cannot be attributed."
            if warning not in self.trace.warnings:
                self.trace.warnings.append(warning)
        count = tensor.numel()
        if count > 2**40:
            raise TraceLimitError("Shape capture supports up to 2^40 logical elements per tensor.")
        if not self.shapes and (
            count > MAX_TENSOR_ELEMENTS or self.captured_elements + count > MAX_CAPTURED_ELEMENTS
        ):
            raise TraceLimitError(
                "Value capture limit reached (8,388,608 elements per tensor; 32 million across snapshots). Use Shapes mode for larger tensors."
            )
        self.captured_elements += count
        storage = tensor.untyped_storage()
        address = storage._cdata
        if address not in self.storage_ids:
            self.storage_ids[address] = f"storage-{len(self.storage_ids) + 1}"
            self.storages.append(storage)
        tensor_id = f"t{len(self.trace.tensors)}"
        bare = self.shapes or not self.values
        paged = not bare and count > INLINE_ELEMENTS and self.snapshot_dir is not None
        if paged:
            save_snapshot(self.snapshot_dir, tensor_id, tensor)
        values = [] if bare or paged else tensor.detach().reshape(-1).tolist()
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        # Preserve integers that JavaScript cannot represent exactly, as well as NaN/Inf.
        safe_values = [
            str(v) if not math.isfinite(v) or isinstance(v, int) and abs(v) > 2**53 - 1 else v
            for v in values
        ]
        tensor_id = f"t{len(self.trace.tensors)}"
        if id(tensor) in self.parameters:
            name, role = self.parameters[id(tensor)], "parameter"
        elif isinstance(tensor, torch.nn.Parameter) and role == "intermediate":
            # A module built during the traced call has no registered path.
            name, role = "parameter", "parameter"
        if not axes or len(axes) != tensor.ndim:
            axes = [f"axis {i}" for i in range(tensor.ndim)]
        spread = None
        if not bare:
            # The distribution's own tensor operations are not steps.
            with self.pause_capture():
                spread = value_histogram(tensor)
        state = TensorState(
            id=tensor_id,
            name=name,
            shape=list(tensor.shape),
            axes=axes,
            dtype=str(tensor.dtype).removeprefix("torch."),
            strides=list(tensor.stride()),
            storage_id=self.storage_ids[address],
            storage_offset=tensor.storage_offset(),
            contiguous=tensor.is_contiguous(),
            numel=count,
            values=safe_values,
            value_source="shape" if bare else "paged" if paged else "inline",
            # Paged tensors send no values; their range comes from the histogram.
            minimum=min(finite) if finite else spread.low if spread and spread.counts else None,
            maximum=max(finite) if finite else spread.high if spread and spread.counts else None,
            histogram=spread,
            role=role,
        )
        self.trace.tensors[tensor_id] = state
        self.live[id(tensor)] = (tensor, observation, tensor_id)
        return tensor_id

    def knock_out(self, result, operands):
        """The step's result with the knockout applied. A step that writes in
        place has its tensor written over; any other gets a new tensor."""
        rule = self.knockout
        results = list(tensors_in(result))
        if not results:
            # Not a step: nothing will be recorded for it.
            return result
        self.knocked = True
        if rule.output >= len(results):
            raise ValueError(
                f"Step {rule.step + 1} makes {len(results)} "
                f"{'result' if len(results) == 1 else 'results'}, not {rule.output + 1}."
            )
        target = results[rule.output]
        with self.pause_capture():
            replaced = knocked_value(target, rule, self.patch)
            if any(target is operand for operand in operands):
                with torch.no_grad():
                    target.copy_(replaced)
                return result
        return replace_in(result, target, replaced)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.suspended:
            return func(*args, **kwargs)
        kind = getattr(func, "__name__", str(func))
        if kind == "__get__":
            # A property such as x.T or x.mT: name the step after the property.
            kind = getattr(getattr(func, "__self__", None), "__name__", kind)
        # Metadata queries are useful to Python, but are not tensor transformations.
        if kind in {
            "size",
            "dim",
            "ndimension",
            "numel",
            "stride",
            "is_contiguous",
            "__repr__",
            "__str__",
            "__format__",
            "item",
            "tolist",
            "__len__",
        }:
            return func(*args, **kwargs)
        source = self.source()
        if source is None:
            return func(*args, **kwargs)
        if len(self.trace.operations) >= MAX_OPERATIONS:
            raise TraceLimitError(
                "The run exceeded 256 recorded operations. Try a smaller example."
            )
        operands = list(operands_for(kind, args, kwargs))
        input_ids = [self.capture(t) for t in operands]
        # An unobserved native call may have changed a different live tensor
        # since the last intercepted operation. Never bless its old snapshot
        # with a new observation just because this call does not touch it.
        # One look at each live tensor serves twice: it finds such a change,
        # and it is that tensor's state before this call.
        # An analysis pass (values=False) attributes no writes: it reads only
        # the steps and the live tensors, so it skips this watch on every live
        # tensor, by far the costliest part of recording a pass.
        seen = {}
        if self.values:
            for key, (tensor, recorded, _) in list(self.live.items()):
                seen[key] = TensorObservation.read(tensor)
                if seen[key] != recorded:
                    self.capture(tensor)
        before = dict(self.live) if self.values else {}
        observations = {
            key: seen.get(key) or TensorObservation.read(item[0]) for key, item in before.items()
        }
        # Preserve duplicates: x @ x has two distinct argument positions.
        arguments = arguments_for(kind, args, kwargs)
        error = None
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            result, error = None, exc
        elapsed = time.perf_counter() - started
        if (
            error is None
            and self.knockout is not None
            and not self.knocked
            and len(self.trace.operations) == self.knockout.step
        ):
            result = self.knock_out(result, operands)
        after = {key: TensorObservation.read(item[0]) for key, item in before.items()}
        effects = affected_tensors(
            kind,
            observations,
            after,
            [id(t) for t in operands],
            [id(t) for t in tensors_in(kwargs.get("out"))],
            kwargs.get("inplace", False),
        )
        # Shared version bumps alone do not change an alias's layout or values.
        # Keep these cached snapshots rather than inventing an unproduced state.
        for key, (tensor, _, tensor_id) in before.items():
            if key not in effects:
                self.live[key] = (tensor, after[key], tensor_id)
        output_ids = []
        annotated = False
        if error is None:
            found = self.assignment(source)
            annotation = re.search(r"#\s*axes:\s*(.+)$", source.text)
            # `# axes:` names the statement's value. In
            # `q = q.reshape(...).transpose(1, 2)  # axes: …` the reshape is an
            # intermediate whose axes are in a different order.
            annotated = bool(annotation) and (found is None or found[2])
            axes = [s.strip() for s in annotation[1].split(",")] if annotated else None
            results = list(tensors_in(result))
            assigned = found[1] if found else None
            default = kind
            template = None
            if kind == "__getitem__" and input_ids:
                subscript = index_text(arguments.get("index"))
                if subscript:
                    template = ("", input_ids[0], subscript)
            elif kind in PREFIX_OPERATORS and len(input_ids) == 1:
                template = (PREFIX_OPERATORS[kind], input_ids[0], "")
            if template:
                # The operand's name may still change: on `y = ~mask[:3]` the
                # subscript is provisionally called y. `final_default_names`
                # rebuilds this once every name is settled.
                prefix, operand, suffix = template
                default = f"{prefix}{self.trace.tensors[operand].name}{suffix}"
                op_id = f"op{len(self.trace.operations)}"
                self.default_names[op_id] = default
                self.default_templates[op_id] = template
            for i, t in enumerate(results):
                if assigned and len(assigned) == len(results) > 1:
                    # a, b = x.chunk(2) names each part.
                    output_name = assigned[i]
                else:
                    name = assigned[0] if assigned and len(assigned) == 1 else default
                    output_name = name if i == 0 else f"{name}[{i}]"
                if id(t) in effects and not assigned:
                    output_name = self.trace.tensors[before[id(t)][2]].name
                output_ids.append(self.capture(t, name=output_name, axes=axes, force=True))
        mutations = []
        for key, effect in effects.items():
            tensor, _, before_id = before[key]
            previous = self.trace.tensors[before_id]
            # A returned tensor may already have been captured above. Aliases
            # with independent version counters (e.g. .data) still need a snapshot.
            after_id = self.live[key][2]
            if after_id == before_id:
                after_id = self.capture(
                    tensor, name=previous.name, axes=previous.axes, role=previous.role, force=True
                )
            elif effect != "metadata" and not re.search(r"#\s*axes:", source.text):
                updated = self.trace.tensors[after_id]
                if updated.shape == previous.shape:
                    updated.axes = previous.axes.copy()
            mutations.append(TensorMutation(before=before_id, after=after_id, kind=effect))
        if output_ids or mutations or error:
            inputs = [self.trace.tensors[t] for t in input_ids]
            outputs = [self.trace.tensors[t] for t in output_ids]
            lesson = describe_operation(kind, arguments, inputs, outputs)
            if lesson.interaction == "reduction" and not re.search(r"#\s*axes:", source.text):
                outputs[0].axes = reduction_axes(kind, arguments, inputs, outputs)
            if lesson.interaction == "inspect" and not self.shapes and inputs and outputs:
                with self.pause_capture():
                    replayed = replayed_selection(kind, func, args, kwargs, inputs[0], outputs[0])
                lesson = replayed or lesson
            if lesson.interaction == "tensor_assembly" and not re.search(r"#\s*axes:", source.text):
                for tensor, axes in zip(outputs, assembly_axes(kind, arguments, inputs, outputs)):
                    tensor.axes = axes
            if (
                lesson.interaction == "relation"
                and lesson.relation
                and outputs
                and not re.search(r"#\s*axes:", source.text)
            ):
                outputs[0].axes = relation_axes(lesson.relation, inputs, outputs[0])
            if (
                lesson.interaction in {"convolution", "patch_projection"}
                and len(inputs[0].shape) == len(outputs[0].shape)
                and not re.search(r"#\s*axes:", source.text)
            ):
                # A convolution keeps batch, channel, and spatial roles.
                outputs[0].axes = inputs[0].axes.copy()
            if lesson.interaction == "pooling" and not re.search(r"#\s*axes:", source.text):
                for tensor in outputs:
                    tensor.axes = inputs[0].axes.copy()
            if mutations:
                metadata_only = all(m.kind == "metadata" for m in mutations)
                lesson = Lesson(
                    title="Change a tensor's layout in place"
                    if metadata_only
                    else "Write into shared storage",
                    summary=(
                        "This changes the receiving tensor's layout or storage binding. Other views keep their own layout."
                        if metadata_only
                        else "This operation writes into existing storage. Recorded views of that storage are refreshed together."
                    ),
                    detail=(
                        "Before and after are immutable snapshots. A layout change does not imply that values were rearranged in memory. Resizing can expose new, uninitialized values."
                        if metadata_only
                        else "Before and after are immutable snapshots of the real execution. Shared-storage connections track the whole storage; disjoint slices or a write of the same value may have unchanged cells. Copies on separate storage are unaffected."
                    ),
                    category="memory",
                )
            if not annotated and lesson.axis_order and inputs and outputs:
                outputs[0].axes = [inputs[0].axes[i] for i in lesson.axis_order]
            if (
                not annotated
                and inputs
                and outputs
                and kind in SAME_AXES
                and inputs[0].shape == outputs[0].shape
            ):
                outputs[0].axes = inputs[0].axes.copy()
            if (
                not annotated
                and inputs
                and outputs
                and all(axis.startswith("axis ") for axis in outputs[0].axes)
            ):
                named = feature_axes(kind, inputs[0], outputs[0])
                if named:
                    outputs[0].axes = named
            index = len(self.trace.operations)
            # A failing step still marks its statement's position, so the
            # steps before it on `y = x.long() @ x.T` are not left named `y`,
            # a variable the failed line never assigned.
            found = self.assignment(source)
            if annotated and found and outputs:
                self.backfill_axes(kind, lesson, input_ids, outputs[0], found[0])
            self.positions.append(
                (*self.position, found[0] if found else None, bool(found and found[2]))
                if self.position
                else None
            )
            self.trace.operations.append(
                Operation(
                    id=f"op{index}",
                    index=index,
                    kind=kind,
                    function=f"{getattr(func, '__module__', 'torch')}.{kind}",
                    inputs=input_ids,
                    outputs=output_ids,
                    mutations=mutations,
                    arguments=arguments,
                    source=source,
                    module=" / ".join(self.module_stack),
                    lesson=lesson,
                    status="error" if error else "ok",
                    error=str(error) if error else None,
                    loops=self.loops.context(inspect.currentframe()),
                    duration_us=None if self.shapes else round(elapsed * 1e6, 2),
                )
            )
        if error:
            raise error
        return result
