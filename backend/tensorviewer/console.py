"""Wrap a few lines of tensor code in a module so the ordinary recorder can run them."""

import ast
import io
import tokenize

HEADER = """import math

import torch
import torch.nn.functional as F
from torch import nn


class Console(nn.Module):
"""
INDENT = "        "
# Script line n is generated line n + SCRIPT_LINE_OFFSET.
SCRIPT_LINE_OFFSET = HEADER.count("\n") + 1


def _names(target) -> list[str] | None:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = [_names(item) for item in target.elts]
        return [n for group in names for n in group] if all(names) else None
    return None


def _indented_lines(script: str) -> list[str]:
    # A literal's continuation lines are part of its value, not Python block
    # indentation. Keep them unchanged when nesting the script inside forward.
    literal_lines = set()
    fstring_start = None
    fstring_depth = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(script).readline):
            if token.type == tokenize.STRING:
                literal_lines.update(range(token.start[0], token.end[0]))
            elif token.type == getattr(tokenize, "FSTRING_START", -1):
                if not fstring_depth:
                    fstring_start = token.start[0]
                fstring_depth += 1
            elif token.type == getattr(tokenize, "FSTRING_END", -1):
                fstring_depth -= 1
                if not fstring_depth:
                    literal_lines.update(range(fstring_start, token.end[0]))
    except (tokenize.TokenError, SyntaxError):
        # Incomplete code still reaches the worker's ordinary syntax diagnostic.
        pass
    return [
        line if row in literal_lines or not line.strip() else INDENT + line
        for row, line in enumerate(script.split("\n"))
    ]


def console_code(script: str, positional: list[str], keywords: list[str]) -> str:
    """The final expression or assignment becomes the output; the script text is not rewritten."""
    parameters = ["self", *positional, *(["*", *keywords] if keywords else [])]
    lines = _indented_lines(script.rstrip() or "pass")
    result = (positional or keywords or ["None"])[0]
    try:
        body = ast.parse("def forward():\n" + "\n".join(lines)).body[0].body
    except SyntaxError:
        # Keep the text intact so the worker reports the error at the user's own line.
        body = []
    last = body[-1] if body else None
    if isinstance(last, ast.Return):
        result = None
    elif isinstance(last, ast.Expr) and not isinstance(last.value, ast.Constant):
        row = last.lineno - 2
        lines[row] = lines[row][: last.col_offset] + "return " + lines[row][last.col_offset :]
        result = None
    elif isinstance(last, ast.Assign) and _names(last.targets[0]):
        result = ", ".join(_names(last.targets[0]))
    elif isinstance(last, (ast.AnnAssign, ast.AugAssign)) and isinstance(last.target, ast.Name):
        result = last.target.id
    if result:
        lines.append(f"{INDENT}return {result}")
    return f"{HEADER}    def forward({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
