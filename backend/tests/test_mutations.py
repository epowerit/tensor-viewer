"""Mutation provenance must match snapshots, not merely a shared version counter."""

import pytest

from tensorviewer.models import InputSpec, ProjectDraft, Trace
from tensorviewer.snapshots import read_snapshot
from tensorviewer.worker import execute


def run(body, shape=(2, 3), *, shapes=False, directory=None):
    return execute(
        ProjectDraft(
            name="Shared storage",
            class_name="Example",
            constructor={},
            input=InputSpec(shape=list(shape), axis_names=[]),
            capture_mode="shapes" if shapes else "values",
            code="import torch\nfrom torch import nn\nclass Example(nn.Module):\n    def forward(self, x):\n"
            + "\n".join("        " + line for line in body.splitlines()),
        ),
        snapshot_dir=directory,
    )


def assert_chronology(trace):
    assert not trace.warnings
    available = set(trace.input_ids)
    for op in trace.operations:
        for effect in op.mutations:
            assert effect.before in available
            assert effect.after not in available
            assert effect.before != effect.after
        available.update(op.outputs)
        available.update(m.after for m in op.mutations)
    assert set(trace.output_ids) <= available


def test_updates_are_produced_by_the_actual_write_and_keep_historical_values():
    trace = run("y = x.view(-1)\nx.add_(10)\nreturn y * 2")
    assert trace.error is None
    view, write, multiply = trace.operations
    direct, alias = write.mutations
    assert (direct.before, direct.after, direct.kind) == (
        trace.input_ids[0],
        write.outputs[0],
        "write",
    )
    assert (alias.before, alias.after, alias.kind) == (view.outputs[0], multiply.inputs[0], "alias")
    assert trace.tensors[alias.before].values == list(range(6))
    assert trace.tensors[alias.after].values == list(range(10, 16))
    assert trace.tensors[alias.after].name == "y"
    assert_chronology(trace)


def test_index_assignment_is_recorded_despite_returning_none():
    trace = run("y = x.view(-1)\nx[0, 0] = 42\nreturn y")
    assert trace.error is None
    write = trace.operations[1]
    assert write.kind == "__setitem__" and not write.outputs
    assert len(write.mutations) == 2
    assert trace.output_ids == [write.mutations[1].after]
    assert trace.tensors[trace.output_ids[0]].values == [42, 1, 2, 3, 4, 5]
    assert_chronology(trace)


@pytest.mark.parametrize("write", ["torch.add(x, 10, out=x)", "x.copy_(x + 10)", "x.add_(10)"])
def test_out_copy_and_inplace_writes_preserve_both_operand_and_alias_provenance(write):
    trace = run("y = x.view(-1)\n" + write + "\nreturn y")
    assert trace.error is None
    mutation = next(op for op in trace.operations if op.mutations)
    assert trace.tensors[trace.output_ids[0]].values == list(range(10, 16))
    assert trace.output_ids[0] in [m.after for m in mutation.mutations]
    assert [m.kind for m in mutation.mutations] == ["write", "alias"]
    assert_chronology(trace)


@pytest.mark.parametrize("view", ["x.detach()", "x.data", "x.view(torch.int32)"])
def test_aliases_with_separate_versions_or_dtypes_follow_written_storage(view):
    trace = run(f"y = {view}\nx.zero_()\nreturn y")
    assert trace.error is None
    alias = next(m for m in trace.operations[-1].mutations if m.kind == "alias")
    assert trace.output_ids == [alias.after]
    assert trace.tensors[alias.after].values == [0] * 6
    assert trace.tensors[alias.before].values != [0] * 6
    assert_chronology(trace)


def test_repeated_writes_chain_every_view_and_do_not_touch_copies():
    trace = run("y = x.view(-1)\nz = x.clone()\nx.add_(10)\ny.mul_(2)\nreturn x, y, z")
    assert trace.error is None
    first, second = trace.operations[-2:]
    assert {m.after for m in first.mutations} == {m.before for m in second.mutations}
    assert not any(
        m.before == trace.operations[1].outputs[0] for op in trace.operations for m in op.mutations
    )
    assert [trace.tensors[t].values for t in trace.output_ids] == [
        list(range(20, 32, 2)),
        list(range(20, 32, 2)),
        list(range(6)),
    ]
    assert_chronology(trace)


@pytest.mark.parametrize(
    "change",
    ["x.transpose_(0, 1)", "x.unsqueeze_(0)", "x.as_strided_((3,2),(2,1))", "x.resize_(3, 2)"],
)
def test_layout_changes_do_not_invent_value_writes_to_other_views(change):
    trace = run("y = x.view(-1)\n" + change + "\nreturn y * 2")
    assert trace.error is None
    view, layout, multiply = trace.operations
    assert len(layout.mutations) == 1
    assert layout.mutations[0].kind == "metadata"
    assert multiply.inputs[0] == view.outputs[0]
    assert trace.tensors[trace.output_ids[0]].values == list(range(0, 12, 2))
    assert_chronology(trace)


def test_rebinding_storage_leaves_old_and_new_base_tensors_untouched():
    trace = run("y = x.view(-1)\nz = x + 10\nx.set_(z)\nother = z + 1\nreturn x, y, z")
    assert trace.error is None
    # set_ bypasses TorchFunctionMode in this PyTorch build. Do not attribute it
    # to the previous add, and explain why the new binding is a captured root.
    assert not any(op.mutations for op in trace.operations)
    assert trace.warnings and "cannot be attributed" in trace.warnings[0]
    assert [trace.tensors[t].values for t in trace.output_ids] == [
        list(range(10, 16)),
        list(range(6)),
        list(range(10, 16)),
    ]
    assert (
        trace.tensors[trace.output_ids[1]].storage_id
        == trace.tensors[trace.operations[0].outputs[0]].storage_id
    )
    assert trace.output_ids[2] == trace.operations[1].outputs[0]
    assert trace.output_ids[0] not in {i for op in trace.operations for i in op.outputs}


def test_disjoint_slices_are_honestly_refreshed_at_storage_granularity():
    trace = run("left = x[:1]\nright = x[1:]\nleft.add_(10)\nreturn right")
    assert trace.error is None
    right = next(
        m for m in trace.operations[-1].mutations if m.before == trace.operations[1].outputs[0]
    )
    assert right.kind == "alias"
    assert trace.tensors[right.before].values == trace.tensors[right.after].values == [3, 4, 5]
    assert "disjoint" in trace.operations[-1].lesson.detail
    assert_chronology(trace)


def test_rebound_tensor_version_counter_does_not_imply_shared_storage():
    trace = run("y = x.view(-1)\nz = x + 10\nx.set_(z)\nx.add_(10)\nreturn x, y, z")
    assert trace.error is None
    write = trace.operations[-1]
    assert write.kind == "add_"
    assert len(write.mutations) == 2
    assert all(
        trace.tensors[m.before].storage_id
        == trace.tensors[trace.operations[1].outputs[0]].storage_id
        for m in write.mutations
    )
    assert [trace.tensors[t].values for t in trace.output_ids] == [
        list(range(20, 26)),
        list(range(6)),
        list(range(20, 26)),
    ]


def test_shape_only_aliases_are_bounded_and_failed_writes_do_not_invent_updates():
    trace = run("y = x.view(-1)\nx.add_(10)\nreturn y", (1024, 1024, 1024), shapes=True)
    assert trace.error is None
    assert len(trace.operations[1].mutations) == 2
    assert all(t.value_source == "shape" and t.values == [] for t in trace.tensors.values())
    assert len(trace.model_dump_json()) < 15000
    assert_chronology(trace)
    failed = run("y = x.view(-1)\nx.copy_(torch.ones(4))\nreturn y")
    assert failed.error is not None
    assert failed.operations[-1].status == "error"
    assert failed.operations[-1].mutations == []


def test_paged_alias_updates_use_immutable_snapshots(tmp_path):
    trace = run("y = x.view(-1)\nx.add_(10)\nreturn y", (128, 128), directory=tmp_path)
    assert trace.error is None
    alias = trace.operations[1].mutations[1]
    before = read_snapshot(tmp_path, alias.before, [0, 16383])
    after = read_snapshot(tmp_path, alias.after, [0, 16383])
    assert before == [0, 16383]
    assert after == [10, 16393]
    assert_chronology(trace)


def test_older_traces_have_no_inferred_mutations():
    trace = run("return x + 1")
    data = trace.model_dump()
    for op in data["operations"]:
        del op["mutations"]
    assert Trace.model_validate(data).operations[0].mutations == []


def test_inplace_functional_call_and_axis_labels():
    trace = run("y = x.view(-1)\nreturn torch.nn.functional.relu(x, inplace=True), y")
    assert trace.error is None
    assert len(trace.operations[-1].mutations) == 2
    assert trace.operations[-1].mutations[0].kind == "write"
    assert_chronology(trace)
    draft = ProjectDraft(
        name="Named axes",
        class_name="Example",
        constructor={},
        input=InputSpec(shape=[2, 3], axis_names=["tokens", "features"]),
        code="from torch import nn\nclass Example(nn.Module):\n def forward(self,x):\n  x.add_(10)\n  return x",
    )
    trace = execute(draft)
    after = trace.tensors[trace.output_ids[0]]
    assert after.name == "x" and after.axes == ["tokens", "features"]


def test_no_return_module_keeps_its_write_and_later_alias_consumer():
    draft = ProjectDraft(
        name="Side effects",
        class_name="Example",
        constructor={},
        input=InputSpec(shape=[2, 3], axis_names=[]),
        code="""from torch import nn
class Write(nn.Module):
    def forward(self, x):
        x.add_(10)
        x.mul_(2)
class Example(nn.Module):
    def __init__(self):
        super().__init__()
        self.write = Write()
    def forward(self, x):
        y = x.view(-1)
        self.write(x)
        return y + 1
""",
    )
    trace = execute(draft)
    assert trace.error is None
    call = trace.module_calls[1]
    assert call.outputs == [] and (call.start_index, call.end_index) == (1, 3)
    alias = next(m for m in trace.operations[2].mutations if m.kind == "alias")
    assert trace.operations[-1].inputs == [alias.after]
    assert trace.tensors[trace.output_ids[0]].values == [21, 23, 25, 27, 29, 31]
    assert_chronology(trace)
