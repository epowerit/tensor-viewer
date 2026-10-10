"""Which loop iteration recorded each operation.

A trace hook watches only the user's own frames and counts how often each loop
header runs: the header executes once before every iteration (and once more
when the loop ends), so the count is the iteration the body is in. Operations
read the counts of every user frame on the stack, so a loop that calls a module
(``for block in self.blocks: x = block(x)``) tags the operations inside the
block too.
"""

import ast
import functools
import sys
from dataclasses import dataclass, field
from itertools import count

from .models import LoopStep


@dataclass(frozen=True)
class Loop:
    header: int
    body_start: int
    end: int
    text: str
    # The header line also belongs to the body when both share one line.
    inline: bool


def scope_of(code) -> int:
    """The key of a code object's function: its first line, or 0 for a module."""
    return 0 if code.co_name == "<module>" else code.co_firstlineno


@dataclass
class FileLoops:
    path: str | None
    # Loops keyed by the function that runs them (see ``scope_of``).
    loops: list[tuple[int, Loop]]
    headers: set[int]

    def enclosing(self, scope: int, line: int) -> list[Loop]:
        """Loops of one function around ``line``, outermost first."""
        found = [
            loop
            for owner, loop in self.loops
            if owner == scope
            and (loop.body_start <= line <= loop.end if line != loop.header else loop.inline)
        ]
        return sorted(found, key=lambda loop: (loop.header, -loop.end))

    def nested(self, header: int) -> list[int]:
        outer = next(loop for _, loop in self.loops if loop.header == header)
        return [
            loop.header
            for _, loop in self.loops
            if loop.header != header and outer.body_start <= loop.header <= outer.end
        ]


@functools.lru_cache(maxsize=32)
def file_loops(source: str, path: str | None = None) -> FileLoops | None:
    # Cached: an analysis records the same code tens of times, and the result
    # is only ever read.
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines()
    loops: list[tuple[int, Loop]] = []

    def visit(node: ast.AST, scope: int):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                # co_firstlineno counts decorators as part of the function.
                decorators = getattr(child, "decorator_list", [])
                visit(child, min([child.lineno, *(d.lineno for d in decorators)]))
                continue
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)) and child.body:
                header = child.lineno
                first = child.body[0].lineno
                text = lines[header - 1].strip() if header <= len(lines) else ""
                loops.append(
                    (
                        scope,
                        Loop(
                            header=header,
                            body_start=first if first > header else header,
                            end=child.body[-1].end_lineno or first,
                            text=text.split(":", 1)[0] if first == header else text.rstrip(":"),
                            inline=first == header,
                        ),
                    )
                )
            visit(child, scope)

    visit(tree, 0)
    if not loops:
        return None
    return FileLoops(path, loops, {loop.header for _, loop in loops})


@dataclass
class FrameState:
    serial: int
    counts: dict[int, int] = field(default_factory=dict)


class LoopTracker:
    """Counts loop headers in user frames while a forward pass runs."""

    def __init__(self, files: dict[str, FileLoops]):
        self.files = files
        self.frames: dict[int, FrameState] = {}
        self.serials = count()
        self.previous = None
        self.active = False
        self.paused = False

    def start(self):
        # Another tracer (a debugger or coverage) keeps priority; operations
        # then simply carry no loop information.
        if not self.files or sys.gettrace() is not None:
            return
        self.active = True
        sys.settrace(self._call)

    def stop(self):
        if self.active:
            sys.settrace(None)
            self.active = False
        self.paused = False
        self.frames.clear()

    # The hook runs on every Python call, the recorder's own included: tens
    # of thousands a run against a handful in user code. The recorder turns
    # it off while it works and back on for the call it records, which may
    # reach user code.
    def pause(self) -> bool:
        if not self.active or self.paused:
            return False
        sys.settrace(None)
        self.paused = True
        return True

    def resume(self, paused: bool):
        if paused:
            self.paused = False
            sys.settrace(self._call)

    def call(self, func, args, kwargs):
        """An intercepted call, with the hook on as it is in user code."""
        paused = self.paused
        self.resume(paused)
        try:
            return func(*args, **kwargs)
        finally:
            if paused:
                sys.settrace(None)
                self.paused = True

    def _call(self, frame, event, _arg):
        if event != "call" or frame.f_code.co_filename not in self.files:
            return None
        self.frames[id(frame)] = FrameState(next(self.serials))
        return self._local

    def _local(self, frame, event, _arg):
        state = self.frames.get(id(frame))
        if state is None:
            return None
        if event == "line":
            loops = self.files[frame.f_code.co_filename]
            line = frame.f_lineno
            if line in loops.headers:
                state.counts[line] = state.counts.get(line, 0) + 1
                # A new pass of an outer loop starts its inner loops afresh.
                for inner in loops.nested(line):
                    state.counts.pop(inner, None)
        elif event == "return":
            self.frames.pop(id(frame), None)
        return self._local

    def context(self, frame) -> list[LoopStep]:
        """Loop iterations around the current operation, outermost first."""
        if not self.active:
            return []
        user = []
        while frame:
            if frame.f_code.co_filename in self.files and id(frame) in self.frames:
                user.append(frame)
            frame = frame.f_back
        steps: list[LoopStep] = []
        path = ""
        for frame in reversed(user):
            state = self.frames[id(frame)]
            loops = self.files[frame.f_code.co_filename]
            for loop in loops.enclosing(scope_of(frame.f_code), frame.f_lineno):
                iteration = state.counts.get(loop.header, 0)
                if iteration < 1:
                    continue
                path += f"/{state.serial}:{loop.header}"
                steps.append(
                    LoopStep(
                        id=path,
                        line=loop.header,
                        file=loops.path,
                        text=loop.text,
                        iteration=iteration,
                    )
                )
                path += f"#{iteration}"
        return steps
