from tensorviewer.models import InputSpec, ProjectDraft, Trace
from tensorviewer.worker import execute


def run(code):
    return execute(
        ProjectDraft(
            name="Stages",
            code=code,
            class_name="Model",
            constructor={},
            input=InputSpec(shape=[2, 3], axis_names=[]),
        )
    )


def test_repeated_module_calls_have_distinct_boundaries_without_snapshot_operations():
    trace = run("""from torch import nn
class Block(nn.Module):
    def forward(self, x):
        y = x + 1
        return y * 2
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = Block()
    def forward(self, x):
        return self.shared(self.shared(x))
""")
    assert trace.error is None
    assert [op.kind for op in trace.operations] == ["add", "mul", "add", "mul"]
    root, first, second = trace.module_calls
    assert root.start_index == 0 and root.end_index == 4
    assert first.id != second.id
    assert first.parent_id == second.parent_id == root.id
    assert first.path == second.path == "shared"
    assert (first.start_index, first.end_index) == (0, 2)
    assert (second.start_index, second.end_index) == (2, 4)
    assert first.inputs == trace.input_ids
    assert second.inputs == first.outputs
    assert second.outputs == root.outputs == trace.output_ids
    assert trace.tensors[root.outputs[0]].values == [6, 10, 14, 18, 22, 26]


def test_keyword_inputs_and_tuple_returns_keep_real_tensor_order_and_duplicates():
    trace = run("""from torch import nn
class Block(nn.Module):
    def forward(self, *, right, left):
        a = left + right
        b = a * 2
        return left, b, b
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x):
        return self.block(right=x, left=x)
""")
    assert trace.error is None
    call = trace.module_calls[1]
    assert call.inputs == trace.input_ids * 2
    assert call.outputs == trace.output_ids
    assert call.outputs[0] == trace.input_ids[0]
    assert call.outputs[1] == call.outputs[2] == trace.operations[-1].outputs[0]
    assert len(trace.operations) == 2


def test_failed_nested_call_keeps_successful_steps_and_no_invented_return():
    trace = run("""from torch import nn
class Block(nn.Module):
    def forward(self, x):
        y = x + 1
        return y.reshape(11)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()
    def forward(self, x):
        return self.block(x)
""")
    assert trace.error is not None
    assert len(trace.operations) == 2 and trace.operations[-1].status == "error"
    assert len(trace.module_calls) == 2
    assert all(c.end_index == 2 and not c.outputs for c in trace.module_calls)


def test_empty_calls_do_not_accumulate_metadata_or_invent_operations():
    trace = run("""from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
    def forward(self, x):
        for _ in range(1000):
            x = self.identity(x)
        return x + 1
""")
    assert trace.error is None
    assert len(trace.module_calls) == len(trace.operations) == 1
    assert trace.module_calls[0].outputs == trace.output_ids


def test_older_saved_traces_remain_valid():
    assert Trace.model_validate({"schema_version": "1", "operations": []}).module_calls == []


def test_failure_in_an_earlier_pre_hook_does_not_close_the_enclosing_call():
    trace = run("""from torch import nn
def fail(module, args):
    raise ValueError('stop child')
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Identity()
        self.child.register_forward_pre_hook(fail)
    def forward(self, x):
        y = x + 1
        try:
            self.child(y)
        except ValueError:
            pass
        return y * 2
""")
    assert trace.error is None
    assert len(trace.module_calls) == 1
    assert trace.module_calls[0].end_index == 2
    assert trace.module_calls[0].outputs == trace.output_ids
    assert all(op.module == "Model" for op in trace.operations)
