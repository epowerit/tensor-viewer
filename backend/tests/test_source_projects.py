import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tensorviewer.app import create_app
from tensorviewer.environments import EnvironmentRequest, create_environment, environment_python
from tensorviewer.models import ProjectDraft
from tensorviewer.runner import run_project
from tensorviewer.source_projects import import_git, source_hash
from tensorviewer.worker import execute

ENTRY = """from torch import nn
from .helper import transform
class Example(nn.Module):
    def forward(self, x):
        y = transform(x)
        return y + 2
"""
HELPER = """def transform(x):
    transposed = x.transpose(1, 2) # axes: batch, features, tokens
    return transposed * 3
"""


def project(**changes):
    return ProjectDraft.model_validate(
        dict(
            name="Source project",
            code=ENTRY,
            entry_path="src/demo/model.py",
            import_root="src",
            class_name="Example",
            constructor={},
            input={"shape": [2, 3, 4], "axis_names": []},
            files={"src/demo/__init__.py": "", "src/demo/helper.py": HELPER},
            **changes,
        )
    )


def test_multifile_relative_imports_trace_names_lines_and_cleanup():
    old_path = list(sys.path)
    trace = execute(project())
    assert trace.error is None
    assert [(o.kind, o.source.file, o.source.line) for o in trace.operations] == [
        ("transpose", "src/demo/helper.py", 2),
        ("mul", "src/demo/helper.py", 3),
        ("add", "src/demo/model.py", 6),
    ]
    transposed = trace.tensors[trace.operations[0].outputs[0]]
    assert transposed.name == "transposed"
    assert transposed.axes == ["batch", "features", "tokens"]
    assert trace.tensors[trace.output_ids[0]].values[:6] == [2, 14, 26, 5, 17, 29]
    assert trace.runtime["torch"]
    assert sys.path == old_path and "demo.helper" not in sys.modules
    assert run_project(project()).error is None


@pytest.mark.parametrize(
    "code,line",
    [
        ("def transform(x):\n    return x.bad_method()", 2),
        ("raise RuntimeError('import failed')", 1),
        ("def transform(x)\n    return x", 1),
    ],
)
def test_helper_failures_keep_file_locations(code, line):
    draft = project()
    draft.files["src/demo/helper.py"] = code
    trace = execute(draft)
    assert trace.error and trace.error.file == "src/demo/helper.py"
    assert trace.error.line == line


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "/tmp/escape.py",
        "a/../b.py",
        "a//b.py",
        "a\\b.py",
        ".git/config.py",
        "a\x00.py",
    ],
)
def test_source_paths_reject_traversal(path):
    with pytest.raises(ValidationError):
        ProjectDraft(name="Invalid", code="pass", files={path: "pass"})


def test_source_budgets_and_entry_contract():
    for files in [
        {f"p{i}.py": "" for i in range(128)},
        {"huge.py": "x" * 500001},
        {f"p{i}.py": "x" * 450000 for i in range(5)},
        {"model.py": "pass"},
    ]:
        with pytest.raises(ValidationError):
            ProjectDraft(name="Invalid", code="pass", files=files)
    bad_root = project().model_copy(update={"import_root": "src/demo/other"})
    assert "import root" in execute(bad_root).error.message


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()

    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args]
            )
            .decode()
            .strip()
        )

    git("init", "-q")
    git("config", "user.name", "TensorViewer test")
    git("config", "user.email", "test@example.invalid")
    pkg = root / "example" / "src" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "model.py").write_text(ENTRY)
    (pkg / "helper.py").write_text(HELPER)
    (pkg / "__init__.py").write_text("")
    (root / "example" / "requirements.txt").write_text("einops==0.8.1\n")
    (root / "example" / "escape.py").symlink_to("/etc/passwd")
    (root / "example" / "setup.py").write_text("raise RuntimeError('must not execute on import')")
    git("add", ".")
    git("commit", "-qm", "fixture")
    sha = git("rev-parse", "HEAD")
    (pkg / "helper.py").write_text("uncommitted content")
    return root, sha


def test_git_import_pins_committed_tree_without_executing_or_following_links(repository):
    root, sha = repository
    snapshot = import_git(str(root), sha, "example")
    assert snapshot["repository"]["revision"] == sha
    assert snapshot["files"]["src/demo/helper.py"] == HELPER
    assert "escape.py" not in snapshot["files"]
    assert snapshot["skipped"] == 1
    assert snapshot["repository"]["sha256"] == source_hash(snapshot["files"])
    with pytest.raises(ValueError):
        import_git(str(root), "--upload-pack=bad")
    with pytest.raises(ValueError):
        import_git(str(root), "does-not-exist")


def test_git_api_and_run_source_snapshots_survive_project_edits(tmp_path, repository):
    client = TestClient(create_app(tmp_path / "data"))
    root, _ = repository
    result = client.post(
        "/api/v1/sources/git", json={"repository": str(root), "subdirectory": "example"}
    )
    assert result.status_code == 200
    snapshot = result.json()
    files = snapshot["files"]
    code = files.pop("src/demo/model.py")
    draft = project().model_dump() | {
        "code": code,
        "files": files,
        "repository": snapshot["repository"],
    }
    saved = client.post("/api/v1/projects", json=draft).json()
    run = client.post(f"/api/v1/projects/{saved['id']}/runs").json()
    assert run["trace"]["error"] is None
    draft["files"]["src/demo/helper.py"] = "raise ValueError('later edit')"
    assert client.put(f"/api/v1/projects/{saved['id']}", json=draft).status_code == 200
    restarted = TestClient(create_app(tmp_path / "data"))
    old = restarted.get(f"/api/v1/runs/{run['id']}").json()
    assert old["project"]["files"]["src/demo/helper.py"] == HELPER
    assert old["project"]["repository"] == snapshot["repository"]
    assert old["trace"]["runtime"]["Python"]
    draft["environment"] = "a" * 64
    assert client.post("/api/v1/projects", json=draft).status_code == 409


@pytest.mark.parametrize(
    "requirement",
    ["-e .", "torch", "x>=1", "https://example.com/a.whl", "x @ file:///tmp/a", "x==1 --bad"],
)
def test_dependencies_require_explicit_pinned_wheels(requirement):
    with pytest.raises(ValidationError):
        EnvironmentRequest(requirements=[requirement])


def test_separate_environment_runs_without_installing_any_packages(tmp_path):
    request = EnvironmentRequest(requirements=[])
    result = create_environment(tmp_path, request)
    python = environment_python(tmp_path, result.id)
    assert python.is_file() and result.packages["torch"]
    assert create_environment(tmp_path, request) == result
    assert run_project(project(), python_executable=python).error is None
    with pytest.raises(ValueError):
        environment_python(tmp_path, "../bin/python")


def test_pinned_wheel_install_is_local_to_selected_environment(tmp_path, monkeypatch):
    import importlib.util
    import zipfile

    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = wheel_dir / "tensorviewer_test_addon-0.1-py3-none-any.whl"
    dist = "tensorviewer_test_addon-0.1.dist-info"
    files = {
        "tensorviewer_test_addon.py": "def transform(x):\n    return x + 7\n",
        f"{dist}/METADATA": "Metadata-Version: 2.1\nName: tensorviewer-test-addon\nVersion: 0.1\n",
        f"{dist}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
        archive.writestr(
            f"{dist}/RECORD", "\n".join(f"{name},," for name in [*files, f"{dist}/RECORD"])
        )
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(wheel_dir))
    root = tmp_path / "environments"
    result = create_environment(
        root, EnvironmentRequest(requirements=["tensorviewer-test-addon==0.1"])
    )
    assert result.packages["tensorviewer-test-addon"] == "0.1"
    draft = ProjectDraft(
        name="Dependency test",
        constructor={},
        class_name="Example",
        code=(
            "from torch import nn\nfrom tensorviewer_test_addon import transform\n"
            "class Example(nn.Module):\n    def forward(self,x):\n        return transform(x)\n"
        ),
    )
    trace = run_project(draft, python_executable=environment_python(root, result.id))
    assert trace.error is None
    assert trace.runtime["tensorviewer-test-addon"] == "0.1"
    assert trace.tensors[trace.output_ids[0]].values[:3] == [7, 8, 9]
    assert importlib.util.find_spec("tensorviewer_test_addon") is None
    assert list(root.iterdir()) == [root / result.id]
    with pytest.raises(ValueError, match="setup failed"):
        create_environment(
            root, EnvironmentRequest(requirements=["tensorviewer-test-addon==999.0"])
        )
    assert list(root.iterdir()) == [root / result.id]


def test_unimported_invalid_python_does_not_prevent_the_selected_model_running():
    draft = project()
    draft.files["old_example.py"] = "print 'Python 2 example'"
    trace = execute(draft)
    assert trace.error is None
    assert len(trace.operations) == 3
