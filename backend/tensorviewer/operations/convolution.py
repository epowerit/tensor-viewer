"""Validate cross-correlation geometry without allocating a receptive-field map."""

from math import prod

from ..models import Lesson


def spatial_argument(value, dimensions, minimum):
    values = [value] * dimensions if isinstance(value, int) else value
    if (
        not isinstance(values, (list, tuple))
        or len(values) != dimensions
        or any(type(v) is not int or v < minimum for v in values)
    ):
        raise ValueError("Invalid convolution geometry.")
    return list(values)


def convolution_spec(kind, args, inputs, outputs):
    if kind not in {"conv1d", "conv2d"} or len(inputs) not in {2, 3} or len(outputs) != 1:
        raise ValueError("Expected an input, kernel, optional bias, and one output.")
    dimensions = 1 if kind == "conv1d" else 2
    image, weight = inputs[:2]
    output = outputs[0]
    if (
        len(image.shape) not in {dimensions + 1, dimensions + 2}
        or len(weight.shape) != dimensions + 2
        or len(output.shape) != len(image.shape)
        or any(not t.numel or t.dtype != image.dtype for t in [*inputs, output])
        or image.dtype not in {"float16", "bfloat16", "float32", "float64"}
    ):
        raise ValueError("Unsupported convolution tensor layout or dtype.")
    channel_axis = len(image.shape) - dimensions - 1
    channels, features = image.shape[channel_axis], weight.shape[0]
    groups = args.get("groups", 1)
    if (
        type(groups) is not int
        or groups < 1
        or channels % groups
        or features % groups
        or weight.shape[1] != channels // groups
        or (len(inputs) == 3 and inputs[2].shape != [features])
    ):
        raise ValueError("Invalid channel groups or bias shape.")
    kernel = weight.shape[2:]
    stride = spatial_argument(args.get("stride", 1), dimensions, 1)
    dilation = spatial_argument(args.get("dilation", 1), dimensions, 1)
    effective = [d * (k - 1) + 1 for k, d in zip(kernel, dilation)]
    padding = args.get("padding", 0)
    if isinstance(padding, str):
        if padding == "valid":
            before = after = [0] * dimensions
        elif padding == "same" and all(s == 1 for s in stride):
            before = [(k - 1) // 2 for k in effective]
            after = [k - 1 - p for k, p in zip(effective, before)]
        else:
            raise ValueError("Unsupported string padding.")
    else:
        before = after = spatial_argument(padding, dimensions, 0)
    spatial = image.shape[-dimensions:]
    expected = [
        (n + a + b - k) // s + 1 for n, a, b, k, s in zip(spatial, before, after, effective, stride)
    ]
    if output.shape != [*image.shape[:channel_axis], features, *expected]:
        raise ValueError("Recorded shapes do not match convolution geometry.")
    return {
        "dimensions": dimensions,
        "groups": groups,
        "channels_per_group": channels // groups,
        "kernel": kernel,
        "stride": stride,
        "dilation": dilation,
        "before": before,
        "after": after,
        "terms": channels // groups * prod(kernel),
    }


def describe_convolution(kind, args, inputs, outputs):
    p = convolution_spec(kind, args, inputs, outputs)
    return Lesson(
        title=f"Slide a {p['dimensions']}D kernel",
        summary=f"Each output combines {p['terms']:,} input × weight terms from its channel group, plus bias when present.",
        detail=(
            "PyTorch computes cross-correlation: the kernel is not flipped. "
            "Input position = output position × stride − padding before + kernel position × dilation. "
            "Positions outside the recorded input read virtual zero padding. "
            "Channel groups restrict which input channels contribute to each output channel. "
            "Select an output or kernel cell to inspect its exact contributors."
        ),
        category="compute",
        interaction="convolution",
    )
