"""Bounded source snapshots. Importing a repository never imports its Python code."""

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

MAX_FILES = 128
MAX_SOURCE_BYTES = 2_000_000
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini"}


def valid_path(value: str, directory=False):
    if directory and value == ".":
        return value
    path = PurePosixPath(value)
    if (
        not value
        or len(value) > 240
        or "\\" in value
        or any(ord(c) < 32 for c in value)
        or path.is_absolute()
        or any(part in {"", ".", "..", ".git"} for part in value.split("/"))
    ):
        raise ValueError("Use a relative project path without empty segments, .., or .git.")
    if not directory and path.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError("Source files must be Python or supported text/configuration files.")
    return value


def validate_files(files):
    if len(files) > MAX_FILES:
        raise ValueError(f"Use at most {MAX_FILES} source files. Import a smaller subdirectory.")
    if sum(len(text.encode()) for text in files.values()) > MAX_SOURCE_BYTES:
        raise ValueError("Project sources must fit within 2 MB. Import a smaller subdirectory.")
    for path, text in files.items():
        valid_path(path)
        if len(text.encode()) > 500_000 or "\0" in text:
            raise ValueError(
                "Each source file must be UTF-8 text smaller than 500 KB, without null bytes."
            )


def source_hash(files):
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def import_git(repository, revision="HEAD", subdirectory="."):
    valid_path(subdirectory, directory=True)
    remote = urlsplit(repository)
    if remote.scheme:
        if (
            remote.scheme != "https"
            or not remote.hostname
            or remote.username
            or remote.password
            or remote.query
            or remote.fragment
        ):
            raise ValueError(
                "Use an HTTPS Git URL without credentials, or an absolute local repository path."
            )
    elif not Path(repository).is_absolute() or not Path(repository).is_dir():
        raise ValueError("Use an HTTPS Git URL or an existing absolute local repository path.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", revision):
        raise ValueError("Choose HEAD, a branch, a tag, or a commit SHA.")
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    deadline = time.monotonic() + 90
    with tempfile.TemporaryDirectory(prefix="tensorviewer-git-") as directory:
        root = Path(directory) / "repo"
        base = [
            "git",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "protocol.ext.allow=never",
            "-c",
            "credential.helper=",
            "-c",
            "protocol.file.allow=always" if not remote.scheme else "protocol.file.allow=never",
        ]

        def git(*args):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Git import exceeded its 90-second limit.")
            try:
                with tempfile.TemporaryFile() as output:
                    process = subprocess.Popen(
                        [*base, *args],
                        cwd=directory,
                        env=env,
                        stdout=output,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    try:
                        process.wait(timeout=min(30, remaining))
                    finally:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    if process.returncode:
                        raise ValueError(
                            "Git could not read that repository/revision. Check its address, visibility, and revision."
                        )
                    if output.tell() > 8_000_000:
                        raise ValueError(
                            "Repository listing is too large. Import a smaller repository."
                        )
                    output.seek(0)
                    return output.read()
            except subprocess.TimeoutExpired:
                raise ValueError(
                    "Git import timed out. Try a smaller repository or a local clone."
                ) from None
            except OSError:
                raise ValueError(
                    "Git could not start. Check that Git is installed and local temporary storage is available."
                ) from None

        git("init", "--bare", str(root))
        git("-C", str(root), "fetch", "--depth=1", "--no-tags", "--", repository, revision)
        commit = git("-C", str(root), "rev-parse", "FETCH_HEAD^{commit}").decode().strip()
        entries = git("-C", str(root), "ls-tree", "-r", "-l", "-z", commit).split(b"\0")
        files = {}
        skipped = 0
        prefix = "" if subdirectory == "." else subdirectory + "/"
        for entry in entries:
            if not entry:
                continue
            meta, raw_path = entry.split(b"\t", 1)
            path = raw_path.decode("utf-8", "replace")
            if not path.startswith(prefix):
                continue
            path = path[len(prefix) :]
            mode, kind, sha, size = meta.split()
            if (
                mode not in {b"100644", b"100755"}
                or kind != b"blob"
                or PurePosixPath(path).suffix.lower() not in TEXT_SUFFIXES
            ):
                skipped += 1
                continue
            valid_path(path)
            if int(size) > 500_000:
                raise ValueError(
                    f"{path} exceeds the 500 KB file limit. Import a smaller subdirectory."
                )
            blob = git("-C", str(root), "cat-file", "blob", sha.decode())
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                skipped += 1
                continue
            if "\0" in text:
                skipped += 1
                continue
            files[path] = text
            validate_files(files)
        if not any(p.endswith(".py") for p in files):
            raise ValueError(
                "No Python files were found in the selected revision and subdirectory."
            )
        return {
            "files": files,
            "repository": {
                "url": repository,
                "revision": commit,
                "subdirectory": subdirectory,
                "sha256": source_hash(files),
            },
            "skipped": skipped,
        }


@contextmanager
def project_namespace(project):
    """Materialize a private source tree only for an explicitly requested execution."""
    import importlib
    import importlib.util
    import sys

    if not project.files and project.entry_path == "model.py" and project.import_root == ".":
        namespace = {"__name__": "tensorviewer_user_project"}
        exec(compile(project.code, "<tensorviewer-project>", "exec"), namespace)
        yield namespace, {}
        return
    with tempfile.TemporaryDirectory(prefix="tensorviewer-source-") as directory:
        root = Path(directory)
        files = {**project.files, project.entry_path: project.code}
        sources = {}
        for path, code in files.items():
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(code)
            if path.endswith(".py"):
                sources[str(destination)] = (path, code)
        import_dir = root / project.import_root
        if not import_dir.is_dir():
            raise ValueError("The selected import root is not a directory in this project.")
        entry = root / project.entry_path
        try:
            relative = entry.relative_to(import_dir)
        except ValueError:
            raise ValueError("The entry file must be inside the selected import root.") from None
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts or not all(part.isidentifier() for part in parts):
            raise ValueError("Entry file and package names must be valid Python identifiers.")
        name = ".".join(parts)
        previous_path = list(sys.path)
        previous_cwd = Path.cwd()
        previous_bytecode = sys.dont_write_bytecode
        previous_modules = dict(sys.modules)
        try:
            sys.path.insert(0, str(import_dir))
            sys.dont_write_bytecode = True
            os.chdir(root)
            # Import parents normally so package-relative imports have normal Python semantics.
            if len(parts) > 1:
                importlib.import_module(".".join(parts[:-1]))
            spec = importlib.util.spec_from_file_location(name, entry)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            yield vars(module), sources
        except Exception as exc:
            exc._tensorviewer_source_files = sources
            raise
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode
            for key, module in list(sys.modules.items()):
                filename = getattr(module, "__file__", None)
                if filename and str(filename).startswith(str(root) + os.sep):
                    if key in previous_modules:
                        sys.modules[key] = previous_modules[key]
                    else:
                        sys.modules.pop(key, None)
