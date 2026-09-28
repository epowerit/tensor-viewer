"""Record high-level eager PyTorch calls without rewriting user code."""

import ast
import inspect
import math
import re
from collections.abc import Iterable

import torch
from torch.overrides import TorchFunctionMode

from .models import Operation, SourceLocation, TensorState, Trace
from .operations import describe_operation

MAX_OPERATIONS = 256
MAX_TENSOR_ELEMENTS = 16384
MAX_CAPTURED_ELEMENTS = 250000


class TraceLimitError(RuntimeError):
    pass


def tensors_in(value) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors_in(item)


def plain(value):
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def arguments_for(kind, args, kwargs):
    result = {k: plain(v) for k, v in kwargs.items()}
    rest = list(args[1:])
    if kind in {"reshape", "view", "permute"}:
        key = "dims" if kind == "permute" else "shape"
        if rest:
            result[key] = plain(
                rest[0] if len(rest) == 1 and isinstance(rest[0], (tuple, list)) else rest
            )
    else:
        names = {
            "transpose": ["dim0", "dim1"],
            "softmax": ["dim", "dtype"],
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
        }.get(kind, [])
        for i, value in enumerate(rest):
            if not isinstance(value, torch.Tensor):
                result[names[i] if i < len(names) else f"arg{i + 1}"] = plain(value)
    return result


def operands_for(kind, args, kwargs):
    """Preserve semantic operand order even when keyword arguments are reordered."""
    names = {
        "matmul": ["input", "other"],
        "mm": ["input", "mat2"],
        "bmm": ["input", "mat2"],
        "linear": ["input", "weight", "bias"],
        "add": ["input", "other"],
        "sub": ["input", "other"],
        "mul": ["input", "other"],
        "div": ["input", "other"],
    }.get(kind, ["input"])
    ordered = list(args)
    ordered.extend(kwargs[name] for name in names[len(args) :] if name in kwargs)
    ordered.extend(value for name, value in kwargs.items() if name not in names)
    return tensors_in(ordered)


class Recorder(TorchFunctionMode):
    def __init__(self, code: str, filename: str, model: torch.nn.Module):
        super().__init__()
        self.trace = Trace()
        self.filename = filename
        self.lines = code.splitlines()
        self.live: dict[int, tuple[torch.Tensor, int, str]] = {}
        self.storage_ids: dict[int, str] = {}
        # Keep storages alive so allocator address reuse cannot imply false aliasing.
        self.storages: list = []
        self.captured_elements = 0
        self.module_stack: list[str] = []
        self.hooks = []
        self.parameters = {id(t): name for name, t in model.named_parameters()}
        self.parameters.update({id(t): name for name, t in model.named_buffers()})
        self.names = {}
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(target, ast.Name):
                    self.names[node.lineno] = target.id
        for name, module in model.named_modules():
            self.hooks.append(
                module.register_forward_pre_hook(self._enter(name or model.__class__.__name__))
            )
            self.hooks.append(module.register_forward_hook(self._leave, always_call=True))

    def _enter(self, name):
        def hook(_module, _args):
            self.module_stack.append(name)

        return hook

    def _leave(self, _module, _args, _output):
        if self.module_stack:
            self.module_stack.pop()

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def source(self):
        frame = inspect.currentframe()
        try:
            while frame:
                if frame.f_code.co_filename == self.filename:
                    line = frame.f_lineno
                    return SourceLocation(line=line, text=self.lines[line - 1].strip())
                frame = frame.f_back
        finally:
            del frame
        return None

    def capture(self, tensor, name="tensor", axes=None, role="intermediate", force=False):
        if tensor.device.type != "cpu" or tensor.layout != torch.strided or tensor.is_complex():
            raise TraceLimitError("This first version supports dense, real CPU tensors only.")
        previous = self.live.get(id(tensor))
        if previous and previous[1] == tensor._version and not force:
            return previous[2]
        count = tensor.numel()
        if count > MAX_TENSOR_ELEMENTS or self.captured_elements + count > MAX_CAPTURED_ELEMENTS:
            raise TraceLimitError(
                "Trace size limit reached. Use smaller tensors (16,384 elements per tensor; 250,000 total)."
            )
        self.captured_elements += count
        storage = tensor.untyped_storage()
        address = storage._cdata
        if address not in self.storage_ids:
            self.storage_ids[address] = f"storage-{len(self.storage_ids) + 1}"
            self.storages.append(storage)
        values = tensor.detach().reshape(-1).tolist()
        finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        # Preserve integers that JavaScript cannot represent exactly, as well as NaN/Inf.
        safe_values = [
            str(v) if not math.isfinite(v) or isinstance(v, int) and abs(v) > 2**53 - 1 else v
            for v in values
        ]
        tensor_id = f"t{len(self.trace.tensors)}"
        if id(tensor) in self.parameters:
            name, role = self.parameters[id(tensor)], "parameter"
        if not axes or len(axes) != tensor.ndim:
            axes = [f"axis {i}" for i in range(tensor.ndim)]
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
            minimum=min(finite) if finite else None,
            maximum=max(finite) if finite else None,
            role=role,
        )
        self.trace.tensors[tensor_id] = state
        self.live[id(tensor)] = (tensor, tensor._version, tensor_id)
        return tensor_id

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        kind = getattr(func, "__name__", str(func))
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
        input_ids = [self.capture(t) for t in operands_for(kind, args, kwargs)]
        # Preserve duplicates: x @ x has two distinct argument positions.
        arguments = arguments_for(kind, args, kwargs)
        error = None
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            result, error = None, exc
        output_ids = []
        if error is None:
            annotation = re.search(r"#\s*axes:\s*(.+)$", source.text)
            axes = [s.strip() for s in annotation[1].split(",")] if annotation else None
            name = self.names.get(source.line, kind)
            for i, t in enumerate(tensors_in(result)):
                output_ids.append(
                    self.capture(t, name=name if i == 0 else f"{name}[{i}]", axes=axes, force=True)
                )
        if output_ids or error:
            inputs = [self.trace.tensors[t] for t in input_ids]
            outputs = [self.trace.tensors[t] for t in output_ids]
            lesson = describe_operation(kind, arguments, inputs, outputs)
            if (
                not re.search(r"#\s*axes:", source.text)
                and lesson.axis_order
                and inputs
                and outputs
            ):
                outputs[0].axes = [inputs[0].axes[i] for i in lesson.axis_order]
            if (
                not re.search(r"#\s*axes:", source.text)
                and inputs
                and outputs
                and kind in {"contiguous", "clone", "softmax", "div", "mul", "add", "sub"}
                and inputs[0].shape == outputs[0].shape
            ):
                outputs[0].axes = inputs[0].axes.copy()
            index = len(self.trace.operations)
            self.trace.operations.append(
                Operation(
                    id=f"op{index}",
                    index=index,
                    kind=kind,
                    function=f"{getattr(func, '__module__', 'torch')}.{kind}",
                    inputs=input_ids,
                    outputs=output_ids,
                    arguments=arguments,
                    source=source,
                    module=" / ".join(self.module_stack),
                    lesson=lesson,
                    status="error" if error else "ok",
                    error=str(error) if error else None,
                )
            )
        if error:
            raise error
        return result
