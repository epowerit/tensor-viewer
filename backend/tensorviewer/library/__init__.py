"""The project library: ordinary PyTorch files, from tensor shapes to CLIP.

Each `NN_name.py` beside this file is a complete project. It is read by the
same `read_project` that reads pasted and uploaded code, so a library project
and the same file pasted by a user are indistinguishable. Files are data:
they are never imported here.
"""

import re
from functools import cache
from pathlib import Path

from pydantic import BaseModel

from ..declarations import read_project

FOLDER = Path(__file__).parent
TRACK = re.compile(r"^\s*#\s*track:\s*(.+?)\s*$", re.MULTILINE)


class LibraryEntry(BaseModel):
    id: str
    number: int
    title: str
    summary: str
    track: str
    code: str


@cache
def entries() -> tuple[LibraryEntry, ...]:
    found = []
    for path in sorted(FOLDER.glob("[0-9][0-9]_*.py")):
        code = path.read_text(encoding="utf-8")
        read = read_project(code)
        track = TRACK.search(code)
        found.append(
            LibraryEntry(
                id=path.stem,
                number=int(path.stem[:2]),
                title=read.title,
                summary=read.summary,
                track=track.group(1) if track else "Projects",
                code=code,
            )
        )
    return tuple(found)


def project_name(entry: LibraryEntry) -> str:
    return f"{entry.number:02d} · {entry.title}"
