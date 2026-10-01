import pytest
import torch

from tensorviewer.console import SCRIPT_LINE_OFFSET, console_code
from tensorviewer.models import ForwardInput, InputSpec, ProjectDraft
from tensorviewer.worker import execute


def draft(script, **fields):
    return ProjectDraft(
        name="Console",
        code="pass",
        script=script,
        input=InputSpec(shape=[2, 3, 4], axis_names=["batch", "rows", "columns"]),
        **fields,
    )


def test_final_assignment_is_the_output_and_lines_map_back_to_the_script():
    project = draft("y = x.reshape(2, 12)\nz = y.permute(1, 0)")
    assert project.class_name == "Console" and project.constructor == {}
    trace = execute(project)
    assert trace.error is None
    assert [op.kind for op in trace.operations] == ["reshape", "permute"]
    assert [op.source.line - SCRIPT_LINE_OFFSET for op in trace.operations] == [1, 2]
    assert [trace.tensors[op.outputs[0]].name for op in trace.operations] == ["y", "z"]
    output = trace.tensors[trace.output_ids[0]]
    expected = torch.arange(24.0).reshape(2, 12).permute(1, 0)
    assert output.shape == [12, 2] and output.values == expected.reshape(-1).tolist()


def test_final_expression_and_explicit_return_and_tuple_targets():
    assert execute(draft("x.transpose(0, 2)")).tensors["t1"].shape == [4, 3, 2]
    trace = execute(draft("y = x + 1\nreturn y.sum(\n    dim=0\n)"))
    assert trace.error is None and trace.tensors[trace.output_ids[0]].shape == [3, 4]
    trace = execute(draft("a, b = x.chunk(2, dim=0)"))
    assert [trace.tensors[t].shape for t in trace.output_ids] == [[1, 3, 4], [1, 3, 4]]
    # A comment-only script returns its input unchanged.
    trace = execute(draft("# nothing yet"))
    assert trace.error is None and trace.output_ids == trace.input_ids


@pytest.mark.parametrize(
    "script",
    [
        'text = """first\nsecond"""\ny = x * len(text)',
        'text = """first\n  second\n\nthird\n"""\ny = x * len(text)',
        'text = r"""first\\n\nsecond"""\ny = x * len(text)',
        'text = f"""first {2 + 3}\nsecond {4}"""\ny = x * len(text)',
        'text = b"""first\nsecond"""\ny = x * len(text)',
        "text = 'first\\\nsecond'\ny = x * len(text)",
        'if True:\n    text = """first\nsecond"""\ny = x * len(text)',
        'y = x * len("""first\nsecond""")',
    ],
)
def test_wrapping_preserves_multiline_literal_values(script):
    # Execute the original script independently of the wrapper as the oracle.
    namespace = {"x": torch.arange(24.0).reshape(2, 3, 4)}
    exec(script, namespace)
    trace = execute(draft(script))
    assert trace.error is None, trace.error
    assert trace.tensors[trace.output_ids[0]].values == namespace["y"].flatten().tolist()


def test_multiline_literals_keep_final_expressions_and_error_lines():
    script = 'x * len("""first\nsecond""")'
    trace = execute(draft(script))
    assert trace.error is None
    assert trace.tensors[trace.output_ids[0]].values == (torch.arange(24.0) * 12).tolist()
    for ending, error_type in [("y = x @ x", "RuntimeError"), ("y = (", "SyntaxError")]:
        trace = execute(draft('text = """first\nsecond"""\n' + ending))
        assert trace.error.type == error_type
        assert trace.error.line - SCRIPT_LINE_OFFSET == 3


def test_errors_report_the_script_line_and_keep_earlier_steps():
    trace = execute(draft("y = x.reshape(6, 4)\nz = y @ y"))
    assert trace.error and trace.error.line - SCRIPT_LINE_OFFSET == 2
    assert trace.operations[0].status == "ok" and trace.operations[-1].status == "error"
    trace = execute(draft("y = x.reshape(6, 4)\nz = (y +"))
    assert trace.error.type == "SyntaxError"
    assert trace.error.line - SCRIPT_LINE_OFFSET == 2


def test_named_inputs_shape_mode_and_rejected_combinations():
    project = draft(
        "scores = q @ k.transpose(-2, -1)\ntorch.softmax(scores + mask, dim=-1)",
        input_name="q",
        additional_inputs=[
            ForwardInput(name="k", input=InputSpec(shape=[2, 3, 4])),
            ForwardInput(
                name="mask", binding="keyword", input=InputSpec(shape=[3, 3], axis_names=[])
            ),
        ],
    )
    assert "def forward(self, q, k, *, mask):" in project.code
    trace = execute(project)
    assert trace.error is None and trace.tensors[trace.output_ids[0]].shape == [2, 3, 3]
    shapes = execute(draft("x.permute(2, 0, 1)", capture_mode="shapes"))
    assert shapes.error is None and shapes.tensors[shapes.output_ids[0]].shape == [4, 2, 3]
    try:
        draft("x", files={"helper.py": ""})
    except ValueError as exc:
        assert "Console projects" in str(exc)
    else:
        raise AssertionError("A console project accepted extra files.")


def test_generated_code_keeps_script_text_and_survives_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from tensorviewer.app import create_app

    script = "y = x.flatten(1)  # axes: batch, cells"
    assert script in console_code(script, ["x"], [])
    client = TestClient(create_app(tmp_path))
    project = client.post("/api/v1/projects", json=draft(script).model_dump()).json()
    assert project["script"] == script
    run = client.post(f"/api/v1/projects/{project['id']}/runs").json()
    assert run["project"]["script"] == script
    assert run["trace"]["tensors"][run["trace"]["output_ids"][0]]["axes"] == ["batch", "cells"]
    revised = {**project, "script": "x * 2"}
    saved = client.put(f"/api/v1/projects/{project['id']}", json=revised).json()
    assert "return x * 2" in saved["code"]


def test_module_construction_inside_a_script_is_not_a_recorded_step():
    trace = execute(draft("layer = nn.Linear(4, 5)\ny = layer(x)"))
    assert trace.error is None
    assert [op.kind for op in trace.operations] == ["linear"]
    weight = trace.tensors[trace.operations[0].inputs[1]]
    assert weight.role == "parameter" and weight.shape == [5, 4]


def test_only_a_statements_final_result_takes_the_variable_name():
    def names(script):
        trace = execute(draft(script))
        assert trace.error is None, trace.error
        return [(op.kind, [trace.tensors[t].name for t in op.outputs]) for op in trace.operations]

    assert names("scores = x @ x.transpose(-2, -1)") == [
        ("transpose", ["transpose"]),
        ("matmul", ["scores"]),
    ]
    assert names("first, second = x.chunk(2, dim=0)") == [("chunk", ["first", "second"])]
    # A loop runs the same line again; each visit keeps its own final name.
    assert names("h = x\nfor _ in range(2):\n    h = torch.tanh(h)") == [
        ("tanh", ["h"]),
        ("tanh", ["h"]),
    ]
    assert (
        names("h = x\nfor _ in range(2):\n    h = torch.tanh(h) + 1")
        == [
            ("tanh", ["tanh"]),
            ("add", ["h"]),
        ]
        * 2
    )
    # Two statements on one line, a multi-line value, and a conditional value.
    assert names("a = x + 1; a * 2") == [("add", ["a"]), ("mul", ["mul"])]
    assert names("s = (\n    x.flatten(1)\n    .sum(-1)\n)") == [
        ("flatten", ["flatten"]),
        ("sum", ["s"]),
    ]
    assert names("y = torch.relu(x) if True else torch.tanh(x)") == [("relu", ["y"])]
    assert names("def helper(v):\n    return torch.tanh(v)\ny = torch.relu(x) + helper(x)") == [
        ("relu", ["relu"]),
        ("tanh", ["tanh"]),
        ("add", ["y"]),
    ]
    # Statements without a plain variable target keep operation names.
    assert names("x.flatten(1).sum(-1)") == [("flatten", ["flatten"]), ("sum", ["sum"])]


def test_shape_check_is_an_unsaved_metadata_dry_run(tmp_path):
    from fastapi.testclient import TestClient

    from tensorviewer.app import create_app

    client = TestClient(create_app(tmp_path))
    body = draft("y = x.reshape(6, 4)\nz = y @ y").model_dump()
    checked = client.post("/api/v1/shape-check", json=body)
    assert checked.status_code == 200
    result = checked.json()
    assert result["project"]["capture_mode"] == "shapes"
    trace = result["trace"]
    assert trace["error"]["type"] == "RuntimeError"
    assert [op["status"] for op in trace["operations"]] == ["ok", "error"]
    assert trace["tensors"][trace["operations"][0]["outputs"][0]]["shape"] == [6, 4]
    assert all(t["value_source"] == "shape" and not t["values"] for t in trace["tensors"].values())
    # Nothing is stored: no project, no run, no snapshot.
    assert client.get("/api/v1/projects").json() == []
    assert not any((tmp_path / "snapshots").glob("*"))
    # A shape far beyond the value limit is still checked.
    large = draft("x.permute(2, 0, 1).flatten(1)").model_dump()
    large["input"]["shape"] = [1024, 1024, 1024]
    large["capture_mode"] = "shapes"
    trace = client.post("/api/v1/shape-check", json=large).json()["trace"]
    assert trace["error"] is None
    assert trace["tensors"][trace["output_ids"][0]]["shape"] == [1024, 1024 * 1024]
    canvas = {**body, "script": None, "blueprint": {"has_input": False, "components": []}}
    assert client.post("/api/v1/shape-check", json=canvas).status_code == 422
