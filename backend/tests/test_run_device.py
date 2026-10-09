from tensorviewer.models import InputSpec, ProjectDraft
from tensorviewer.worker import execute

# Reports the function modes around the forward pass, and where a tensor the
# code makes lands.
CODE = """import torch
from torch import nn


class Probe(nn.Module):
    def forward(self, x):
        modes = torch.overrides._get_current_function_mode_stack()
        print(sorted(type(mode).__name__ for mode in modes))
        print(torch.zeros(2).device)
        return x + 1
"""


def probe(capture_mode: str) -> ProjectDraft:
    return ProjectDraft(
        name="Probe",
        code=CODE,
        class_name="Probe",
        constructor={},
        input_name="x",
        input=InputSpec(shape=[2, 3], axis_names=[], generator="arange"),
        capture_mode=capture_mode,
    )


def test_a_cpu_run_records_without_a_device_mode_around_every_call():
    trace = execute(probe("values"))
    assert trace.error is None
    # A device mode would send every tensor call, the recorder's included,
    # through Python; new tensors land on the CPU without one.
    assert "DeviceContext" not in trace.stdout
    assert "cpu" in trace.stdout


def test_a_shapes_only_run_still_makes_new_tensors_on_the_meta_device():
    trace = execute(probe("shapes"))
    assert trace.error is None
    assert "DeviceContext" in trace.stdout
    assert "meta" in trace.stdout
