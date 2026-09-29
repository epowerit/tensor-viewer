"""Explicit, isolated dependency setup for trusted local source projects."""

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class EnvironmentRequest(BaseModel):
    requirements: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("requirements")
    @classmethod
    def pinned_wheels(cls, values):
        if any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.+_-]{0,79}", s)
            for s in values
        ):
            raise ValueError(
                "Use pinned PyPI requirements such as einops==0.8.1. URLs, flags, paths, and source builds are not supported."
            )
        return sorted(set(values))


class RuntimeEnvironment(BaseModel):
    id: str
    requirements: list[str]
    python: str
    packages: dict[str, str]


def environment_python(root: Path, asset_id: str | None):
    if asset_id is None:
        return None
    if not re.fullmatch("[a-f0-9]{64}", asset_id):
        raise ValueError("Invalid environment ID.")
    directory = root / asset_id
    if not (directory / "environment.json").is_file() or not (directory / "bin/python").is_file():
        raise ValueError("The selected Python environment is missing. Set it up again.")
    return directory / "bin/python"


def list_environments(root: Path):
    if not root.exists():
        return []
    return [
        RuntimeEnvironment.model_validate_json(p.read_text())
        for p in sorted(root.glob("*/environment.json"))
    ]


def create_environment(root: Path, request: EnvironmentRequest):
    identity = json.dumps([sys.executable, sys.version, request.requirements], sort_keys=True)
    asset_id = hashlib.sha256(identity.encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    target = root / asset_id
    if (target / "environment.json").is_file():
        return RuntimeEnvironment.model_validate_json((target / "environment.json").read_text())
    # Build at the final path: venv scripts embed absolute paths.
    target.mkdir(exist_ok=True)
    try:

        def command(args, timeout=180):
            with tempfile.TemporaryFile() as log:
                process = subprocess.Popen(
                    args,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=timeout)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if process.returncode:
                    log.seek(max(0, log.tell() - 6000))
                    raise ValueError(
                        "Environment setup failed: " + log.read().decode(errors="replace")
                    )

        command([sys.executable, "-m", "venv", "--system-site-packages", str(target)], 30)
        python = target / "bin/python"
        # A venv created from another venv inherits the *base* Python's packages,
        # not the backend venv. Expose the backend's dependencies explicitly;
        # packages installed into this environment still take precedence.
        site_dir = Path(
            subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
                timeout=10,
            ).strip()
        )
        (site_dir / "_tensorviewer_backend.pth").write_text(sysconfig.get_path("purelib") + "\n")
        if request.requirements:
            command(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--only-binary=:all:",
                    "--no-input",
                    *request.requirements,
                ]
            )
        info = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-c",
                    'import json,platform,importlib.metadata as m; print(json.dumps({"python":platform.python_version(),"packages":{d.metadata["Name"]:m.version(d.metadata["Name"]) for d in m.distributions() if d.metadata.get("Name")}}))',
                ],
                timeout=20,
            )
        )
        result = RuntimeEnvironment(id=asset_id, requirements=request.requirements, **info)
        (target / "environment.json").write_text(result.model_dump_json())
        return result
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        shutil.rmtree(target, ignore_errors=True)
        if isinstance(exc, ValueError):
            raise
        raise ValueError(
            "Environment setup failed or timed out. Check package versions and network access."
        ) from None
