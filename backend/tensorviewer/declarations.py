"""Read a whole project from its source: the model class, its inputs, and a name.

A pasted file, an uploaded file, and a library example all become projects
through `read_project`, so they behave the same. Everything the run needs is
written in the code itself, in comments that plain Python ignores:

    \"\"\"Vision Transformer. Patches become tokens; attention mixes them.\"\"\"

    # model: VisionTransformer
    # input image: batch=2, channels=3, height=16, width=16 | image
    # input tokens: text "the cat sat on the mat"
    # input mask (keyword): batch=2, tokens=6 | ones
    # constructor: {"depth": 2}
    # capture: shapes

`# input NAME: ...` takes axis sizes (`name=size`, or bare sizes) and, after
`|`, any of a dtype (float32, float64, int64), a value source (arange, random,
ones, zeros, image), and `seed=N`. `text "..."` feeds a sentence's token ids
as int64 `[1, tokens]`. Forward parameters without a declaration get the
default input. Nothing here imports or runs the code.
"""

import ast
import json
import re
from typing import Any

from pydantic import BaseModel

from .models import ForwardInput, InputSpec, ProjectDraft
from .samples import MAX_TOKENS, tokenize

DECLARATION = re.compile(r"^\s*#\s*(input|model|constructor|capture)\b(.*)$", re.MULTILINE)
INPUT = re.compile(
    r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<binding>keyword|positional)\))?\s*:\s*(?P<body>.+)$"
)
TEXT = re.compile(r"""^text\s+(["'])(?P<sentence>.*)\1\s*$""")
DTYPES = {"float32", "float64", "int64"}
GENERATORS = {"arange", "random", "ones", "zeros", "image"}
DEFAULT_AXES = ["batch", "tokens", "features"]
DEFAULT_SHAPE = [2, 4, 8]


class ReadProject(BaseModel):
    """A ready-to-run draft and anything the reader could not use."""

    draft: ProjectDraft
    title: str
    summary: str
    notes: list[str]


def _is_module(node: ast.ClassDef) -> bool:
    for base in node.bases:
        name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
        if name.endswith("Module"):
            return True
    return False


def _forward(node: ast.ClassDef) -> ast.FunctionDef | None:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            return item
    return None


def _default_input() -> InputSpec:
    return InputSpec(
        shape=list(DEFAULT_SHAPE),
        generator="random",
        random_stream="input",
        axis_names=list(DEFAULT_AXES),
    )


def _parse_input(body: str, notes: list[str], name: str) -> InputSpec | None:
    parts = [part.strip() for part in body.split("|")]
    text = TEXT.match(parts[0])
    if text:
        sentence = text.group("sentence")
        count = len(tokenize(sentence)[0])
        if not 1 <= count <= MAX_TOKENS:
            notes.append(f"Input {name}: use a sentence with 1 to {MAX_TOKENS} words and symbols.")
            return None
        return InputSpec(
            shape=[1, count],
            generator="text",
            text=sentence,
            dtype="int64",
            axis_names=["batch", "tokens"],
        )
    shape: list[int] = []
    axes: list[str] = []
    for item in [piece.strip() for piece in parts[0].split(",") if piece.strip()]:
        label, _, size = item.rpartition("=")
        try:
            value = int(size)
        except ValueError:
            notes.append(f"Input {name}: {item!r} is not a size; write axis=size or a number.")
            return None
        shape.append(value)
        axes.append(label.strip() or f"axis {len(axes)}")
    if not shape:
        notes.append(f"Input {name}: declare at least one axis size.")
        return None
    dtype, generator, seed = "float32", None, 7
    for option in parts[1:]:
        if option in DTYPES:
            dtype = option
        elif option in GENERATORS:
            generator = option
        elif option.startswith("seed="):
            try:
                seed = int(option[5:])
            except ValueError:
                notes.append(f"Input {name}: {option!r} needs a whole-number seed.")
        else:
            notes.append(f"Input {name}: ignored {option!r}.")
    if generator is None:
        # Models usually see noise; integer inputs count up instead.
        generator = "arange" if dtype == "int64" else "random"
    named = any(not label.startswith("axis ") for label in axes)
    try:
        return InputSpec(
            shape=shape,
            generator=generator,
            dtype=dtype,
            seed=seed,
            random_stream="input",
            axis_names=axes if named else [],
        )
    except ValueError as error:
        notes.append(f"Input {name}: {error}")
        return None


def read_project(code: str, name: str | None = None, model: str | None = None) -> ReadProject:
    """Build a draft from source alone; problems become notes, not errors.

    `model` names the class to run when the caller chose one; otherwise a
    `# model:` line, then the last nn.Module in the file, decides.
    """
    notes: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise ValueError(f"The code does not parse: line {error.lineno}: {error.msg}") from None

    docstring = ast.get_docstring(tree) or ""
    title, _, summary = docstring.strip().partition("\n")
    title = title.strip().rstrip(".")
    summary = " ".join(summary.split())

    declared: dict[str, str] = {}
    inputs: dict[str, tuple[str | None, str]] = {}
    for kind, rest in DECLARATION.findall(code):
        rest = rest.strip()
        if kind == "input":
            match = INPUT.match(rest)
            if not match:
                notes.append(
                    f"Could not read '# input {rest}'; write '# input name: axis=size, ...'."
                )
                continue
            inputs[match.group("name")] = (match.group("binding"), match.group("body").strip())
        else:
            declared[kind] = rest.lstrip(":").strip()

    if model:
        declared["model"] = model
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and _is_module(node)]
    chosen = None
    if "model" in declared:
        chosen = next((node for node in classes if node.name == declared["model"]), None)
        if chosen is None:
            notes.append(f"No nn.Module named {declared['model']} in the code.")
    if chosen is None and classes:
        chosen = classes[-1]
    if chosen is None:
        raise ValueError("No nn.Module class found. Define a class that subclasses nn.Module.")

    constructor: dict[str, Any] = {}
    if "constructor" in declared:
        try:
            value = json.loads(declared["constructor"])
            if not isinstance(value, dict):
                raise ValueError
            constructor = value
        except ValueError:
            notes.append('The "# constructor:" line must be a JSON object, such as {"dim": 8}.')

    parameters: list[tuple[str, str, bool]] = []
    forward = _forward(chosen)
    if forward is None:
        notes.append(f"{chosen.name} has no forward method here; it may inherit one.")
        parameters = [("x", "positional", False)]
    else:
        args = forward.args
        positional = [*args.posonlyargs, *args.args][1:]
        defaults = len(args.defaults)
        for index, arg in enumerate(positional):
            parameters.append((arg.arg, "positional", index >= len(positional) - defaults))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            parameters.append((arg.arg, "keyword", default is not None))

    forward_inputs: list[ForwardInput] = []
    for parameter, binding, optional in parameters:
        spec = None
        if parameter in inputs:
            override, body = inputs.pop(parameter)
            spec = _parse_input(body, notes, parameter)
            binding = override or binding
        elif optional:
            continue  # Its default applies.
        forward_inputs.append(
            ForwardInput(name=parameter, binding=binding, input=spec or _default_input())
        )
    for leftover in inputs:
        notes.append(f"'# input {leftover}' does not name a forward parameter of {chosen.name}.")
    if not forward_inputs:
        notes.append(f"{chosen.name}.forward takes no tensor inputs; one default input is used.")
        forward_inputs.append(ForwardInput(name="x", input=_default_input()))
    if len(forward_inputs) > 8:
        notes.append("Only the first eight forward inputs are used.")
        forward_inputs = forward_inputs[:8]

    capture = declared.get("capture", "values")
    if capture not in ("values", "shapes"):
        notes.append("'# capture:' takes values or shapes.")
        capture = "values"

    first, *rest = forward_inputs
    draft = ProjectDraft(
        name=(name or title or chosen.name)[:100],
        code=code,
        class_name=chosen.name,
        constructor=constructor,
        input=first.input,
        input_name=first.name,
        input_binding=first.binding,
        additional_inputs=rest,
        capture_mode=capture,
    )
    return ReadProject(draft=draft, title=title or chosen.name, summary=summary, notes=notes)
