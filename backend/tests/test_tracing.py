import json

import pytest
import torch
from pydantic import ValidationError

from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.templates import TEMPLATES
from tensorviewer.worker import execute


def project(body, shape=(1, 2, 4)):
    return ProjectDraft(
        name="Test",
        class_name="Example",
        constructor={},
        input=InputSpec(shape=list(shape), axis_names=[]),
        code="import torch\nfrom torch import nn\nclass Example(nn.Module):\n    def forward(self, x):\n"
        + "\n".join("        " + line for line in body.splitlines()),
    )


def test_attention_matches_uninstrumented_pytorch_and_keeps_branches():
    draft = TEMPLATES[0].project
    trace = execute(draft)
    assert trace.error is None
    namespace = {}
    exec(draft.code, namespace)
    torch.manual_seed(draft.input.seed)
    model = namespace["Attention"](**draft.constructor).eval()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    with torch.no_grad():
        expected = model(x)
    actual = trace.tensors[trace.output_ids[0]]
    torch.testing.assert_close(torch.tensor(actual.values).reshape(actual.shape), expected)
    assert [o.kind for o in trace.operations].count("matmul") == 2
    assert len(trace.operations) == 18
    assert {o.inputs[0] for o in trace.operations[:3]} == {trace.input_ids[0]}
    assert all(o.source and o.source.line > 0 for o in trace.operations)
    assert trace.operations[3].lesson.interaction == "mapping"
    for operation in trace.operations:
        for reference in operation.inputs + operation.outputs:
            assert reference in trace.tensors


def test_reshape_permute_and_contiguous_have_exact_element_mappings():
    trace = execute(TEMPLATES[1].project)
    assert trace.error is None
    for operation in trace.operations:
        before, after = (trace.tensors[ids[0]] for ids in (operation.inputs, operation.outputs))
        assert operation.lesson.mapping is not None
        assert [before.values[i] for i in operation.lesson.mapping] == after.values
    reshape, permute, contiguous = trace.operations
    a = trace.tensors[reshape.inputs[0]]
    b = trace.tensors[permute.outputs[0]]
    c = trace.tensors[contiguous.outputs[0]]
    assert a.storage_id == b.storage_id
    assert not b.contiguous
    assert b.storage_id != c.storage_id
    assert c.contiguous
    assert b.values == c.values


def test_contiguous_noop_is_recorded_without_inventing_a_copy():
    trace = execute(project("return x.contiguous()"))
    assert trace.error is None
    operation = trace.operations[0]
    assert operation.kind == "contiguous"
    before, after = (trace.tensors[ids[0]] for ids in (operation.inputs, operation.outputs))
    assert before.storage_id == after.storage_id
    assert "no copy" in operation.lesson.detail


def test_noncontiguous_reshape_detects_real_copy():
    trace = execute(project("y = x.transpose(1, 2)\nreturn y.reshape(1, -1)"))
    assert trace.error is None
    operation = trace.operations[-1]
    before, after = (trace.tensors[ids[0]] for ids in (operation.inputs, operation.outputs))
    assert before.storage_id != after.storage_id
    assert after.values == before.values


def test_snapshots_survive_inplace_mutation_and_alias_updates():
    trace = execute(project("y = x.view(-1)\nx.add_(10)\nreturn y * 2"))
    assert trace.error is None
    original = trace.tensors[trace.input_ids[0]]
    view = trace.tensors[trace.operations[0].outputs[0]]
    updated_alias = trace.tensors[trace.operations[-1].inputs[0]]
    assert original.values == list(range(8))
    assert view.values == list(range(8))
    assert updated_alias.values == list(range(10, 18))
    assert updated_alias.storage_id == original.storage_id


def test_unfold_mapping_preserves_overlapping_membership():
    trace = execute(project("return x.unfold(0, 3, 1)", (5,)))
    assert trace.error is None
    op = trace.operations[0]
    assert op.lesson.mapping == [0, 1, 2, 1, 2, 3, 2, 3, 4]
    assert trace.tensors[op.outputs[0]].values == op.lesson.mapping


def test_failure_keeps_previous_steps_and_failing_source():
    trace = execute(project("y = x.transpose(1, 2)\nreturn y.view(1, -1)"))
    assert trace.error and trace.error.type == "RuntimeError"
    assert trace.error.line == 6
    assert len(trace.operations) == 2
    assert trace.operations[-1].status == "error"
    assert trace.operations[-1].outputs == []


def test_data_dependent_branch_and_generic_operation():
    trace = execute(project("if x.sum() > 0:\n    return torch.erf(x)\nreturn torch.cos(x)"))
    assert trace.error is None
    assert any(o.kind == "erf" and o.lesson.category == "generic" for o in trace.operations)
    assert not any(o.kind == "cos" for o in trace.operations)


def test_duplicate_operands_and_multiple_outputs_are_preserved():
    trace = execute(project("y = x @ x\nreturn torch.chunk(y, 2, dim=-1)", (1, 2, 2)))
    assert trace.error is None
    matmul = trace.operations[0]
    assert len(matmul.inputs) == 2 and matmul.inputs[0] == matmul.inputs[1]
    assert len(trace.output_ids) == 2


def test_nonfinite_values_are_json_safe_and_input_limits_are_validated():
    trace = execute(project("return x / 0"))
    assert trace.error is None
    values = trace.tensors[trace.output_ids[0]].values
    assert "nan" in values and "inf" in values
    json.dumps(trace.model_dump(), allow_nan=False)
    with pytest.raises(ValidationError):
        InputSpec(shape=[2**41], axis_names=[])


def test_tensor_size_limit_is_explained():
    trace = execute(project("return x.expand(2000000, -1, -1)"))
    assert trace.error and trace.error.type == "TraceLimitError"


def test_billion_element_shape_trace_is_bounded_and_preserves_layout():
    draft = project("return x")
    draft.capture_mode = "shapes"
    draft.input = InputSpec(shape=[1024, 1024, 1024], axis_names=["batch", "tokens", "features"])
    draft.code = draft.code.replace(
        "return x", "y = x.permute(0, 2, 1)\n        return y.reshape(1024, -1)"
    )
    trace = execute(draft)
    assert trace.error is None
    assert trace.tensors[trace.output_ids[0]].shape == [1024, 1048576]
    assert all(t.value_source == "shape" and t.values == [] for t in trace.tensors.values())
    assert trace.operations[0].lesson.mapping_rule == "permutation"
    assert trace.operations[0].lesson.mapping is None
    assert not trace.tensors[trace.operations[0].outputs[0]].contiguous
    assert len(trace.model_dump_json()) < 10000


def test_value_budget_requires_explicit_shape_mode():
    with pytest.raises(ValidationError, match="Shapes mode"):
        ProjectDraft(
            name="Too big", code="pass", input=InputSpec(shape=[1024, 1024, 1024], axis_names=[])
        )


def test_shape_mode_reports_data_dependent_operations_honestly():
    draft = project("if x.sum().item() > 0:\n    return x\nreturn x + 1")
    draft.capture_mode = "shapes"
    trace = execute(draft)
    assert trace.error is not None
    assert all(t.values == [] for t in trace.tensors.values())


def test_syntax_error_and_invalid_class_are_reported():
    draft = project("return x")
    draft.code = "def broken("
    assert execute(draft).error.type == "SyntaxError"
    draft.code = "class Example: pass"
    assert "nn.Module" in execute(draft).error.message


def test_keyword_operand_order_and_tensor_properties_are_recorded():
    trace = execute(
        project("y = x.transpose(-2, -1)\nreturn torch.matmul(other=y, input=x)", (1, 2, 3))
    )
    assert trace.error is None
    op = trace.operations[-1]
    assert trace.tensors[op.inputs[0]].shape == [1, 2, 3]
    assert trace.tensors[op.inputs[1]].shape == [1, 3, 2]
    transposed = execute(project("return x.T", (2, 3)))
    assert transposed.error is None
    assert len(transposed.operations) == 1
    assert transposed.tensors[transposed.output_ids[0]].shape == [3, 2]


def test_dtype_views_are_not_described_as_value_preserving():
    trace = execute(project("return x.view(torch.int32)"))
    assert trace.error is None
    op = trace.operations[0]
    assert op.lesson.interaction == "inspect"
    assert op.lesson.mapping is None
    assert trace.tensors[op.outputs[0]].values != trace.tensors[op.inputs[0]].values


def test_large_integers_and_scalar_softmax_are_display_safe():
    trace = execute(project("return torch.tensor([9007199254740993], dtype=torch.int64)"))
    assert trace.error is None
    assert trace.tensors[trace.output_ids[0]].values == ["9007199254740993"]
    trace = execute(project("return torch.softmax(x.sum(), dim=0)"))
    assert trace.error is None
    assert trace.operations[-1].lesson.interaction == "normalization"
    output = trace.tensors[trace.output_ids[0]]
    assert output.shape == [] and output.values == [1.0]


def test_library_module_binding_names_only_its_returned_tensor_on_each_call():
    draft = project("for _ in range(2):\n    y = self.layer(x)\n    x = y\nreturn y")
    draft.code = draft.code.replace(
        "    def forward(self, x):",
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.layer = nn.Sequential(\n"
        "            nn.Linear(4, 4), nn.Sequential(nn.ReLU(), nn.Linear(4, 4)))\n"
        "    def forward(self, x):",
    )
    trace = execute(draft)
    assert trace.error is None
    assert [(op.kind, trace.tensors[op.outputs[0]].name) for op in trace.operations] == [
        ("linear", "linear"),
        ("relu", "relu"),
        ("linear", "y"),
    ] * 2
    # The module hierarchy still identifies the otherwise unnamed intermediates.
    assert trace.operations[0].module == "Example / layer / layer.0"
    assert trace.operations[1].module == "Example / layer / layer.1 / layer.1.0"
    assert trace.tensors[trace.output_ids[0]].name == "y"

    # If the module call is only part of the assigned expression, its return
    # is an intermediate too. The following addition alone produces y.
    draft.code = draft.code.replace("y = self.layer(x)", "y = self.layer(x) + 1")
    trace = execute(draft)
    assert trace.error is None
    assert [(op.kind, trace.tensors[op.outputs[0]].name) for op in trace.operations] == [
        ("linear", "linear"),
        ("relu", "relu"),
        ("linear", "linear"),
        ("add", "y"),
    ] * 2


def test_library_module_tuple_binding_follows_actual_outputs_after_internal_transpose():
    draft = project("y, weights = self.layer(x, x, x)\nreturn y, weights")
    draft.code = draft.code.replace(
        "    def forward(self, x):",
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.layer = nn.MultiheadAttention(4, 2, batch_first=True)\n"
        "    def forward(self, x):",
    )
    trace = execute(draft)
    assert trace.error is None
    assert [trace.tensors[t].name for t in trace.output_ids] == ["y", "weights"]
    for op in trace.operations:
        for tensor_id in op.outputs:
            if tensor_id not in trace.output_ids:
                assert trace.tensors[tensor_id].name not in {"y", "weights"}


def test_user_module_inner_assignments_keep_their_own_source_names():
    draft = project("y = self.layer(x)\nreturn y")
    draft.code = draft.code.replace(
        "class Example(nn.Module):",
        "class Block(nn.Module):\n"
        "    def forward(self, x):\n"
        "        intermediate = x + 1\n"
        "        result = intermediate * 2\n"
        "        return result\n"
        "class Example(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.layer = Block()",
    )
    trace = execute(draft)
    assert trace.error is None
    assert [trace.tensors[op.outputs[0]].name for op in trace.operations] == [
        "intermediate",
        "result",
    ]
