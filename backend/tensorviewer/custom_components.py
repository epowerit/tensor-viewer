"""Explicit, process-isolated shape checks for saved local PyTorch components."""

import hashlib
import json
from collections import OrderedDict
from threading import Lock

from .models import ComponentSpec, CompositionRequest, InputSpec, ProjectDraft
from .runner import run_project


class NeedsShapeCheck(ValueError):
    pass


def component_source(component: ComponentSpec, index: int, preceding: str) -> tuple[str, str]:
    """Give each module its own globals, retaining source lines in the generated file.

    The source string is displayed in the generated program. Leading blank lines
    align its compiled line numbers with that program for the execution inspector.
    """
    custom = component.custom
    name = f"_tv_source_{index}"
    scope = f"_tv_scope_{index}"
    escaped = custom.code.replace("\\", "\\\\").replace('"', '\\"')
    prefix = f'\n\n{name} = """\n'
    offset = (preceding + prefix).count("\n")
    source = prefix + escaped + '\n"""\n'
    source += f'{scope} = {{"__name__": "tensorviewer_component_{index}"}}\n'
    source += (
        f'exec(compile("\\n" * {offset - 1} + {name}, "<tensorviewer-project>", "exec"), {scope})\n'
    )
    arguments = component.arguments if component.arguments is not None else custom.constructor
    # JSON is decoded at runtime so booleans, nulls and nested settings keep their types.
    module = (
        f'{scope}[{custom.class_name!r}](**__import__("json").loads({json.dumps(arguments)!r}))'
    )
    return source, module


class ComponentChecks:
    """Bounded cache, scoped to this backend session and exact input configuration.

    Browsing, saving and automatic compose requests only consult this cache.
    Only the explicit check endpoint is allowed to execute user code.
    """

    def __init__(self):
        self.cache = OrderedDict()
        self.sequences = OrderedDict()
        self.lock = Lock()

    def verify_sequence(self, plan, request, *, execute=False):
        """Check the real sequence too: preceding operations can change strides."""
        key = hashlib.sha256(
            json.dumps(
                [
                    plan.code,
                    request.input.model_dump(),
                    request.capture_mode,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if not execute:
            with self.lock:
                if key not in self.sequences:
                    plan.valid = False
                    plan.validation_required = True
                    plan.error = "Check custom shapes for this sequence before running."
                    return plan
                return self.sequences[key].model_copy(deep=True)
        trace = run_project(
            ProjectDraft(
                name="Check component sequence",
                code=plan.code,
                class_name="ComposedModel",
                constructor={},
                capture_mode="shapes",
                input=request.input,
            )
        )
        if trace.error:
            plan.valid = False
            plan.error = f"Sequence check: {trace.error.message}"
            failed_module = trace.operations[-1].module if trace.operations else ""
            failed_index = next(
                (i for i in range(len(plan.stages)) if f"stage_{i}" in failed_module.split(".")), 0
            )
            for stage in plan.stages[failed_index:]:
                stage.shape = None
                stage.error = plan.error
        with self.lock:
            self.sequences[key] = plan.model_copy(deep=True)
            self.sequences.move_to_end(key)
            while len(self.sequences) > 128:
                self.sequences.popitem(last=False)
        return plan

    def resolve(self, component, shape, axes, request: CompositionRequest, *, execute=False):
        custom = component.custom
        arguments = component.arguments if component.arguments is not None else custom.constructor
        key = hashlib.sha256(
            json.dumps(
                [
                    custom.code,
                    custom.class_name,
                    arguments,
                    shape,
                    axes,
                    request.input.model_dump(),
                    request.capture_mode,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if not execute:
            with self.lock:
                if key not in self.cache:
                    raise NeedsShapeCheck("Check custom shapes to preview this component's output.")
                result = self.cache[key]
                self.cache.move_to_end(key)
            if isinstance(result, str):
                raise ValueError(result)
            return result
        # Shape validation never allocates the requested tensor's numeric values.
        limit = 8_388_608 if request.capture_mode == "values" else 2**40
        source, module = component_source(component, 0, "import torch\nfrom torch import nn\n")
        code = (
            "import torch\nfrom torch import nn\n"
            + source
            + f"""
class _TVCheck(nn.Module):
    def __init__(self):
        super().__init__()
        self.component = {module}
        count = sum(t.numel() for t in self.component.parameters())
        count += sum(t.numel() for t in self.component.buffers())
        if count > {limit}:
            raise ValueError("Component weights exceed this recording mode's element limit.")

    def forward(self, x):
        y = self.component(x)
        if not isinstance(y, torch.Tensor):
            raise ValueError("Custom sequence components must return one tensor, not a tuple or dictionary.")
        if y.dtype != x.dtype:
            raise ValueError("Custom sequence components must preserve the input dtype in this version.")
        if not 1 <= y.ndim <= 6 or any(d < 1 for d in y.shape):
            raise ValueError("Custom outputs need one to six non-empty dimensions.")
        return y
"""
        )
        trace = run_project(
            ProjectDraft(
                name="Check custom component",
                code=code,
                class_name="_TVCheck",
                constructor={},
                capture_mode="shapes",
                input=InputSpec(
                    **{
                        **request.input.model_dump(),
                        "shape": shape,
                        "axis_names": axes,
                        "generator": "zeros",
                        "uploaded": None,
                    }
                ),
            )
        )
        if trace.error:
            result = f"{custom.name}: {trace.error.message}"
        elif not trace.output_ids:
            result = f"{custom.name}: No tensor was returned."
        elif max((t.numel for t in trace.tensors.values()), default=0) > limit:
            result = "A custom intermediate tensor exceeds this recording mode's element limit. Use Shapes only."
        else:
            output = trace.tensors[trace.output_ids[0]]
            result = (output.shape, output.axes)
        with self.lock:
            self.cache[key] = result
            self.cache.move_to_end(key)
            while len(self.cache) > 128:
                self.cache.popitem(last=False)
        if isinstance(result, str):
            raise ValueError(result)
        return result
