"""Small sequential component library. Shape rules and code share one registry."""

from math import prod

from .custom_components import NeedsShapeCheck, component_source
from .models import ComponentStage, CompositionPlan, CompositionRequest, ProjectDraft
from .templates import ATTENTION_CODE, PATCH_EMBEDDING_CODE
from .vision import VISION_CODE, transformer_parameters, vision_budget


def field(key, label, default, maximum=4096, minimum=1):
    return {"key": key, "label": label, "default": default, "min": minimum, "max": maximum}


CATALOG = [
    {
        "kind": "attention",
        "title": "Self-attention",
        "group": "Models",
        "description": "Explicit Q, K, V projections and attention heads.",
        "parameters": [field("heads", "Heads", 2, 16)],
    },
    {
        "kind": "transformer",
        "title": "Transformer blocks",
        "group": "Models",
        "description": "Pre-norm attention, residual connections and a feed-forward network.",
        "parameters": [
            field("heads", "Heads", 2, 16),
            field("blocks", "Blocks", 1, 4),
            field("expansion", "Feed-forward multiplier", 2, 4),
        ],
    },
    {
        "kind": "conv2d",
        "title": "2D convolution",
        "group": "Spatial",
        "description": "A spatial convolution over [batch, channels, height, width].",
        "parameters": [
            field("channels", "Output channels", 8, 256),
            field("kernel", "Kernel size", 3, 11),
            field("stride", "Stride", 1, 8),
        ],
    },
    {
        "kind": "rnn",
        "title": "Simple RNN",
        "group": "Sequence",
        "description": "An explicit tanh recurrent layer. Up to 32 time steps in this release.",
        "parameters": [field("hidden", "Hidden features", 8, 256)],
    },
    {
        "kind": "linear",
        "title": "Linear projection",
        "group": "Layers",
        "description": "Project the final dimension into a new feature space.",
        "parameters": [field("features", "Output features", 8)],
    },
    {
        "kind": "relu",
        "title": "ReLU",
        "group": "Activations",
        "description": "Keep positive values and replace negative values with zero.",
        "parameters": [],
    },
    {
        "kind": "tokens",
        "title": "Image to tokens",
        "group": "Shape adapters",
        "description": "[B, C, H, W] → [B, H×W, C]. Preserves each spatial position.",
        "parameters": [],
    },
    {
        "kind": "flatten",
        "title": "Flatten features",
        "group": "Shape adapters",
        "description": "Keep the batch axis; combine all remaining axes.",
        "parameters": [],
    },
]


def component(kind, title, group, description, *parameters):
    return dict(
        kind=kind, title=title, group=group, description=description, parameters=list(parameters)
    )


def axis_field(key="axis", label="Axis", default=-1):
    return field(key, label, default, 5, -6)


CATALOG += [
    component(
        "vit",
        "Vision Transformer",
        "Models",
        "Image patches → class token + positions → transformer blocks → class logits.",
        field("patch", "Patch size", 4, 32),
        field("features", "Embedding features", 8, 1024),
        field("heads", "Heads", 2, 16),
        field("blocks", "Blocks", 1, 4),
        field("expansion", "Feed-forward multiplier", 2, 4),
        field("classes", "Output classes", 10, 1000),
    ),
    component(
        "token_preparation",
        "Class token + positions",
        "Sequence",
        "Prepend one learned class token and add learned position vectors. [B, N, D] → [B, N+1, D].",
    ),
    component(
        "class_readout",
        "Class-token classifier",
        "Sequence",
        "Layer-normalize the sequence, select token 0, then project to class logits.",
        field("classes", "Output classes", 10, 1000),
    ),
    component(
        "patch_embedding",
        "Patch embedding",
        "Models",
        "Non-overlapping image patches projected into a token sequence.",
        field("patch", "Patch size", 4, 32),
        field("features", "Embedding features", 8),
    ),
    component(
        "mlp",
        "Feed-forward network",
        "Models",
        "Linear → GELU → Linear on the last dimension.",
        field("hidden", "Hidden features", 32),
        field("features", "Output features", 8),
    ),
    component(
        "conv1d",
        "1D convolution",
        "Sequence",
        "Convolution over [batch, channels, length]. Padding is kernel size // 2.",
        field("channels", "Output channels", 8, 256),
        field("kernel", "Kernel size", 3, 11),
        field("stride", "Stride", 1, 8),
    ),
    component(
        "maxpool2d",
        "Max pooling",
        "Spatial",
        "Keep the largest value in each spatial window; no padding.",
        field("kernel", "Window size", 2, 32),
        field("stride", "Stride", 2, 32),
    ),
    component(
        "avgpool2d",
        "Average pooling",
        "Spatial",
        "Average each spatial window; no padding.",
        field("kernel", "Window size", 2, 32),
        field("stride", "Stride", 2, 32),
    ),
    component(
        "adaptiveavgpool2d",
        "Adaptive average pooling",
        "Spatial",
        "Pool an image to a chosen height and width.",
        field("height", "Output height", 1, 256),
        field("width", "Output width", 1, 256),
    ),
    component(
        "layernorm",
        "Layer normalization",
        "Layers",
        "Normalize each vector along the final dimension.",
    ),
    component(
        "batchnorm2d",
        "Batch normalization",
        "Layers",
        "Image-channel normalization in evaluation mode, using initialized running statistics.",
    ),
    component("gelu", "GELU", "Activations", "Smooth Gaussian error linear activation."),
    component("sigmoid", "Sigmoid", "Activations", "Map each value into the interval (0, 1)."),
    component("tanh", "Tanh", "Activations", "Map each value into the interval (-1, 1)."),
    component(
        "softmax",
        "Softmax",
        "Activations",
        "Normalize values into probabilities along an axis.",
        axis_field(),
    ),
    component(
        "transpose",
        "Transpose axes",
        "Shape adapters",
        "Swap two axes without changing their values.",
        axis_field("axis_a", "First axis", -2),
        axis_field("axis_b", "Second axis", -1),
    ),
    component(
        "split_axis",
        "Split axis",
        "Shape adapters",
        "Reshape one dimension into two factors, for example features → heads × head features.",
        axis_field(),
        field("factor", "First factor", 2),
    ),
    component(
        "merge_axes",
        "Merge axes",
        "Shape adapters",
        "Reshape two adjacent dimensions into one.",
        axis_field(default=-2),
    ),
    component(
        "unsqueeze",
        "Insert axis",
        "Shape adapters",
        "Insert a dimension of size one.",
        field("axis", "Axis", 1, 6, -7),
    ),
    component(
        "squeeze",
        "Remove unit axis",
        "Shape adapters",
        "Remove a selected dimension of size one.",
        axis_field(default=1),
    ),
    component(
        "unfold",
        "Unfold windows",
        "Shape adapters",
        "Create sliding windows along an axis; overlapping windows share source elements.",
        axis_field(),
        field("window", "Window size", 2, 256),
        field("step", "Step", 1, 256),
    ),
    component(
        "contiguous",
        "Contiguous layout",
        "Shape adapters",
        "Return contiguous storage, copying only when necessary.",
    ),
    component(
        "mean",
        "Mean reduction",
        "Shape adapters",
        "Average values along a chosen axis, optionally retaining a dimension of size one.",
        axis_field(),
        field("keepdim", "Keep dimension (0 or 1)", 0, 1, 0),
    ),
]
REGISTRY = {item["kind"]: item for item in CATALOG}


def axis_index(value, rank, insertion=False):
    size = rank + int(insertion)
    if not -size <= value < size:
        raise ValueError(f"Axis must be between {-size} and {size - 1} for this tensor.")
    return value % size


SUPPORT = (
    PATCH_EMBEDDING_CODE
    + """

class TransformerBlock(nn.Module):
    def __init__(self, features, heads, expansion):
        super().__init__()
        self.norm1 = nn.LayerNorm(features)
        self.attention = Attention(features, heads)
        self.norm2 = nn.LayerNorm(features)
        self.feedforward = nn.Sequential(
            nn.Linear(features, features * expansion), nn.GELU(),
            nn.Linear(features * expansion, features))

    def forward(self, x):
        normalized = self.norm1(x)
        attended = self.attention(normalized)
        residual = x + attended
        normalized = self.norm2(residual)
        features = self.feedforward(normalized)
        return residual + features


class SimpleRNN(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.hidden = hidden
        self.input = nn.Linear(features, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        h = x.new_zeros(x.shape[0], self.hidden)
        states = []
        for t in range(x.shape[1]):
            h = torch.tanh(self.input(x[:, t, :]) + self.recurrent(h))
            states.append(h)
        return torch.stack(states, dim=1)
"""
)


def compose(request: CompositionRequest, resolve_custom=None) -> CompositionPlan:
    shape, axes = list(request.input.shape), list(request.input.axis_names)
    axes = axes or [f"axis {i}" for i in range(len(shape))]
    stages, init, forward = [], [], []
    validation_required = False
    prefix = (
        ATTENTION_CODE + SUPPORT
        if any(
            c.kind
            in {
                "attention",
                "transformer",
                "rnn",
                "patch_embedding",
                "vit",
                "token_preparation",
                "class_readout",
            }
            for c in request.blueprint.components
        )
        else "import torch\nfrom torch import nn\n"
    )
    if any(
        c.kind in {"vit", "token_preparation", "class_readout"}
        for c in request.blueprint.components
    ):
        prefix += VISION_CODE
    parameter_count = 0
    error = None if request.blueprint.has_input else "Add an input tensor to begin."
    if request.capture_mode == "values" and prod(shape) > 8_388_608:
        error = "Use Shapes only for inputs larger than 8,388,608 elements."
    for index, component in enumerate(request.blueprint.components):
        item = (
            {"title": component.custom.name, "parameters": []}
            if component.custom
            else REGISTRY.get(component.kind)
        )
        stage = ComponentStage(
            id=component.id,
            title=item["title"] if item else component.kind,
            input_shape=list(shape),
        )
        stages.append(stage)
        if error:
            stage.error = "Waiting for a valid preceding tensor."
            continue
        try:
            if item is None:
                raise ValueError("This component is not in the toolbox.")
            fields = {f["key"]: f for f in item["parameters"]}
            if component.parameters.keys() - fields.keys():
                raise ValueError("Unknown component setting.")
            p = {key: component.parameters.get(key, f["default"]) for key, f in fields.items()}
            for key, value in p.items():
                f = fields[key]
                if not f["min"] <= value <= f["max"]:
                    raise ValueError(f"{f['label']} must be between {f['min']} and {f['max']}.")
            kind, module = component.kind, None
            parameters = 0
            intermediate = prod(shape)
            expression = f"self.stage_{index}(x)"
            if kind == "custom":
                if resolve_custom is None:
                    raise NeedsShapeCheck("Check custom shapes to preview this component's output.")
                shape, axes = resolve_custom(component, list(shape), list(axes), request)
                shape, axes = list(shape), list(axes)
                source, module = component_source(component, index, prefix)
                prefix += source
            elif kind == "vit":
                if len(shape) != 4:
                    raise ValueError(
                        "ViT needs [batch, channels, height, width]. Choose an image input."
                    )
                if request.input.dtype == "int64":
                    raise ValueError("ViT needs a floating-point image input.")
                if any(d % p["patch"] for d in shape[2:]):
                    raise ValueError("Image height and width must be divisible by the patch size.")
                if p["features"] % p["heads"]:
                    raise ValueError("Embedding features must divide evenly into attention heads.")
                parameters, largest = vision_budget(shape, **p)
                intermediate = max(intermediate, largest)
                module = f"VisionTransformer({shape[1]}, {shape[2]}, {shape[3]}, patch={p['patch']}, features={p['features']}, heads={p['heads']}, blocks={p['blocks']}, expansion={p['expansion']}, classes={p['classes']})"
                shape, axes = [shape[0], p["classes"]], ["batch", "classes"]
            elif kind in {"token_preparation", "class_readout"}:
                if len(shape) != 3 or request.input.dtype == "int64":
                    raise ValueError("Needs floating-point [batch, tokens, features].")
                if kind == "token_preparation":
                    module = f"TokenPreparation({shape[1]}, {shape[2]})"
                    parameters = (shape[1] + 2) * shape[2]
                    shape[1] += 1
                    axes = ["batch", "tokens", "features"]
                else:
                    module = f"ClassTokenReadout({shape[2]}, {p['classes']})"
                    parameters = 2 * shape[2] + (shape[2] + 1) * p["classes"]
                    shape, axes = [shape[0], p["classes"]], ["batch", "classes"]
            elif kind in {"attention", "transformer", "rnn"}:
                if len(shape) != 3:
                    raise ValueError(
                        "Needs [batch, tokens, features]. Add Image to tokens for images."
                    )
                if request.input.dtype == "int64":
                    raise ValueError("This layer needs a floating-point input.")
                features = shape[-1]
                if kind == "rnn":
                    if shape[1] > 32:
                        raise ValueError("Simple RNN currently supports up to 32 time steps.")
                    module = f"SimpleRNN({features}, {p['hidden']})"
                    parameters = (features + p["hidden"] + 1) * p["hidden"]
                    shape[-1] = p["hidden"]
                    axes = ["batch", "time", "hidden"]
                else:
                    if features % p["heads"]:
                        raise ValueError(
                            f"Feature dimension {features} must divide evenly into {p['heads']} heads."
                        )
                    if kind == "attention":
                        module = f"Attention({features}, {p['heads']})"
                        parameters = 4 * features**2
                    else:
                        block = f"TransformerBlock({features}, {p['heads']}, {p['expansion']})"
                        module = f"nn.Sequential(*[{block} for _ in range({p['blocks']})])"
                        parameters = transformer_parameters(features, p["blocks"], p["expansion"])
                        intermediate = max(intermediate, prod(shape) * p["expansion"])
                    intermediate = max(intermediate, shape[0] * p["heads"] * shape[1] ** 2)
                    axes = ["batch", "tokens", "features"]
            elif kind == "conv2d":
                if len(shape) != 4:
                    raise ValueError(
                        "Needs [batch, channels, height, width]. Choose an image input."
                    )
                if request.input.dtype == "int64":
                    raise ValueError("Convolution needs a floating-point input.")
                padding = p["kernel"] // 2
                parameters = (shape[1] * p["kernel"] ** 2 + 1) * p["channels"]
                module = f"nn.Conv2d({shape[1]}, {p['channels']}, {p['kernel']}, stride={p['stride']}, padding={padding})"
                shape = [
                    shape[0],
                    p["channels"],
                    *[(d + 2 * padding - p["kernel"]) // p["stride"] + 1 for d in shape[2:]],
                ]
                axes = ["batch", "channels", "height", "width"]
            elif kind in {
                "conv1d",
                "patch_embedding",
                "maxpool2d",
                "avgpool2d",
                "adaptiveavgpool2d",
                "batchnorm2d",
            }:
                rank = 3 if kind == "conv1d" else 4
                if len(shape) != rank:
                    raise ValueError(
                        "Needs [batch, channels, length]."
                        if rank == 3
                        else "Needs [batch, channels, height, width]."
                    )
                if request.input.dtype == "int64":
                    raise ValueError("This layer needs a floating-point input.")
                if kind == "conv1d":
                    padding = p["kernel"] // 2
                    parameters = (shape[1] * p["kernel"] + 1) * p["channels"]
                    module = f"nn.Conv1d({shape[1]}, {p['channels']}, {p['kernel']}, stride={p['stride']}, padding={padding})"
                    shape = [
                        shape[0],
                        p["channels"],
                        (shape[2] + 2 * padding - p["kernel"]) // p["stride"] + 1,
                    ]
                    axes = ["batch", "channels", "length"]
                elif kind == "patch_embedding":
                    if any(d % p["patch"] for d in shape[2:]):
                        raise ValueError(
                            "Image height and width must be divisible by the patch size."
                        )
                    module = f"PatchEmbedding({shape[1]}, {p['features']}, {p['patch']})"
                    parameters = (shape[1] * p["patch"] ** 2 + 1) * p["features"]
                    shape = [
                        shape[0],
                        (shape[2] // p["patch"]) * (shape[3] // p["patch"]),
                        p["features"],
                    ]
                    axes = ["batch", "tokens", "features"]
                elif kind == "batchnorm2d":
                    module = f"nn.BatchNorm2d({shape[1]})"
                    parameters = 2 * shape[1]
                elif kind == "adaptiveavgpool2d":
                    module = f"nn.AdaptiveAvgPool2d(({p['height']}, {p['width']}))"
                    shape = [*shape[:2], p["height"], p["width"]]
                else:
                    if min(shape[2:]) < p["kernel"]:
                        raise ValueError("The pooling window must fit inside the image.")
                    cls = "MaxPool2d" if kind == "maxpool2d" else "AvgPool2d"
                    module = f"nn.{cls}({p['kernel']}, stride={p['stride']})"
                    shape = [*shape[:2], *[(d - p["kernel"]) // p["stride"] + 1 for d in shape[2:]]]
            elif kind in {"mlp", "layernorm"}:
                if not shape or request.input.dtype == "int64":
                    raise ValueError(
                        "This layer needs a floating-point tensor with a feature axis."
                    )
                if kind == "layernorm":
                    module = f"nn.LayerNorm({shape[-1]})"
                    parameters = 2 * shape[-1]
                else:
                    parameters = (shape[-1] + 1) * p["hidden"] + (p["hidden"] + 1) * p["features"]
                    intermediate = max(intermediate, prod(shape[:-1]) * p["hidden"])
                    module = f"nn.Sequential(nn.Linear({shape[-1]}, {p['hidden']}), nn.GELU(), nn.Linear({p['hidden']}, {p['features']}))"
                    shape[-1], axes[-1] = p["features"], "features"
            elif kind == "linear":
                if not shape:
                    raise ValueError("Linear projection needs a feature axis.")
                if request.input.dtype == "int64":
                    raise ValueError("Linear projection needs a floating-point input.")
                module = f"nn.Linear({shape[-1]}, {p['features']})"
                parameters = (shape[-1] + 1) * p["features"]
                shape[-1], axes[-1] = p["features"], "features"
            elif kind == "relu":
                module = "nn.ReLU()"
            elif kind in {"gelu", "sigmoid", "tanh", "softmax", "mean"}:
                if request.input.dtype == "int64":
                    raise ValueError("This operation needs a floating-point input.")
                if kind in {"softmax", "mean"}:
                    axis = axis_index(p["axis"], len(shape))
                    if kind == "softmax":
                        module = f"nn.Softmax(dim={axis})"
                    else:
                        expression = f"x.mean(dim={axis}, keepdim={bool(p['keepdim'])})"
                        if p["keepdim"]:
                            shape[axis] = 1
                        else:
                            shape.pop(axis)
                            axes.pop(axis)
                else:
                    module = {"gelu": "nn.GELU()", "sigmoid": "nn.Sigmoid()", "tanh": "nn.Tanh()"}[
                        kind
                    ]
            elif kind == "transpose":
                a, b = (axis_index(p[key], len(shape)) for key in ("axis_a", "axis_b"))
                shape[a], shape[b] = shape[b], shape[a]
                axes[a], axes[b] = axes[b], axes[a]
                expression = f"x.transpose({a}, {b})"
            elif kind in {"split_axis", "merge_axes", "squeeze", "unsqueeze"}:
                axis = axis_index(p["axis"], len(shape), kind == "unsqueeze")
                if kind == "split_axis":
                    if shape[axis] % p["factor"]:
                        raise ValueError(
                            "The selected dimension must divide evenly by the first factor."
                        )
                    shape[axis : axis + 1] = [p["factor"], shape[axis] // p["factor"]]
                    axes[axis : axis + 1] = [axes[axis] + " outer", axes[axis] + " inner"]
                    expression = f"x.reshape({shape})"
                elif kind == "merge_axes":
                    if axis + 1 >= len(shape):
                        raise ValueError("Select an axis with another axis immediately after it.")
                    shape[axis : axis + 2] = [shape[axis] * shape[axis + 1]]
                    axes[axis : axis + 2] = [" × ".join(axes[axis : axis + 2])]
                    expression = f"x.reshape({shape})"
                elif kind == "squeeze":
                    if shape[axis] != 1:
                        raise ValueError("Only an axis of size one can be removed.")
                    shape.pop(axis)
                    axes.pop(axis)
                    expression = f"x.squeeze({axis})"
                else:
                    shape.insert(axis, 1)
                    axes.insert(axis, "unit")
                    expression = f"x.unsqueeze({axis})"
            elif kind == "unfold":
                axis = axis_index(p["axis"], len(shape))
                if p["window"] > shape[axis]:
                    raise ValueError("The window must fit inside the selected dimension.")
                shape[axis] = (shape[axis] - p["window"]) // p["step"] + 1
                shape.append(p["window"])
                axes.append(axes[axis] + " window")
                axes[axis] += " position"
                expression = f"x.unfold({axis}, {p['window']}, {p['step']})"
            elif kind == "contiguous":
                expression = "x.contiguous()"
            elif kind == "flatten":
                if len(shape) < 2:
                    raise ValueError(
                        "Flatten features needs a batch axis and at least one feature axis."
                    )
                shape, axes, expression = (
                    [shape[0], prod(shape[1:])],
                    ["batch", "features"],
                    "x.flatten(1)",
                )
            elif kind == "tokens":
                if len(shape) != 4:
                    raise ValueError("Image to tokens needs [batch, channels, height, width].")
                shape, axes = (
                    [shape[0], shape[2] * shape[3], shape[1]],
                    ["batch", "tokens", "features"],
                )
                expression = "x.flatten(2).transpose(1, 2)"
            if len(shape) > 6:
                raise ValueError("The viewer currently supports up to six tensor dimensions.")
            limit = 8_388_608 if request.capture_mode == "values" else 2**40
            if max(prod(shape), intermediate) > limit:
                raise ValueError(
                    "An output or intermediate tensor exceeds this recording mode's element limit. Reduce the shape or use Shapes only."
                )
            parameter_count += parameters
            if parameter_count > limit:
                raise ValueError(
                    "Model weights exceed this recording mode's element limit. Reduce features or use Shapes only."
                )
            stage.shape, stage.axes = list(shape), list(axes)
            if module:
                init.append(f"        self.stage_{index} = {module}")
            # Labels here are generated, never arbitrary code from settings.
            labels = ", ".join(" ".join(axis.splitlines()) for axis in axes)
            forward.append(f"        x = {expression}  # axes: {labels}")
            if kind == "custom":
                forward.extend(
                    [
                        f"        if not isinstance(x, torch.Tensor) or list(x.shape) != {shape!r} or x.dtype != torch.{request.input.dtype}:",
                        f"            raise ValueError({(component.custom.name + ': output does not match the checked shape and dtype. Recheck this component.')!r})",
                    ]
                )
        except ValueError as exc:
            validation_required = isinstance(exc, NeedsShapeCheck)
            error = stage.error = str(exc)
    if error:
        forward = [f"        raise ValueError({error!r})"]
    if any(c.custom for c in request.blueprint.components):
        limit = 8_388_608 if request.capture_mode == "values" else 2**40
        init.extend(
            [
                f"        if sum(t.numel() for t in self.parameters()) + sum(t.numel() for t in self.buffers()) > {limit}:",
                "            raise ValueError('Combined model weights exceed this recording mode. Use Shapes only.')",
            ]
        )
    code = (
        prefix
        + "\n\nclass ComposedModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    )
    code += "\n".join(init) + "\n\n    def forward(self, x):\n"
    code += "\n".join(forward) + "\n        return x\n"
    return CompositionPlan(
        code=code,
        stages=stages,
        valid=error is None,
        error=error,
        validation_required=validation_required,
    )


def canonical_project(draft: ProjectDraft, resolve_custom=None) -> ProjectDraft:
    if draft.blueprint is None:
        return draft
    plan = compose(
        CompositionRequest(
            blueprint=draft.blueprint, input=draft.input, capture_mode=draft.capture_mode
        ),
        resolve_custom,
    )
    return draft.model_copy(
        update={"code": plan.code, "class_name": "ComposedModel", "constructor": {}}
    )
