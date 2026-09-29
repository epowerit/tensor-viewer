"""Pooling descriptions based on exact recorded window geometry."""

from ..models import Lesson
from .convolution import spatial_argument

POOL_KINDS = {"max_pool2d", "max_pool2d_with_indices", "avg_pool2d", "adaptive_avg_pool2d"}
POOL_ARGUMENTS = {
    "max_pool2d": ["kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"],
    "max_pool2d_with_indices": [
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "ceil_mode",
        "return_indices",
    ],
    "avg_pool2d": [
        "kernel_size",
        "stride",
        "padding",
        "ceil_mode",
        "count_include_pad",
        "divisor_override",
    ],
    "adaptive_avg_pool2d": ["output_size"],
}


def pair(value, minimum):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value * 2
    return spatial_argument(value, 2, minimum)


def pooling_spec(kind, args, inputs, outputs):
    indexed = kind == "max_pool2d_with_indices"
    if kind not in POOL_KINDS or len(inputs) != 1 or len(outputs) != (2 if indexed else 1):
        raise ValueError("Expected one pooling input and its recorded output(s).")
    image, output = inputs[0], outputs[0]
    if (
        len(image.shape) not in {3, 4}
        or len(image.shape) != len(output.shape)
        or image.shape[:-2] != output.shape[:-2]
        or not image.numel
        or not output.numel
        or image.dtype not in {"float16", "bfloat16", "float32", "float64"}
        or output.dtype != image.dtype
        or (indexed and (outputs[1].dtype != "int64" or outputs[1].shape != output.shape))
    ):
        raise ValueError("Unsupported pooling tensors.")
    if kind == "adaptive_avg_pool2d":
        size = args.get("output_size")
        size = [size, size] if type(size) is int else size
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("Invalid adaptive output size.")
        expected = [n if s is None else s for s, n in zip(size, image.shape[-2:])]
        if any(type(s) is not int or s < 1 for s in expected) or expected != output.shape[-2:]:
            raise ValueError("Adaptive sizes must match the recorded output.")
        return {"mode": "adaptive", "output_size": expected}
    maximum = kind.startswith("max_")
    kernel = pair(args.get("kernel_size"), 1)
    stride_value = args.get("stride")
    stride = kernel if stride_value is None or stride_value == [] else pair(stride_value, 1)
    padding = pair(args.get("padding", 0), 0)
    dilation = pair(args.get("dilation", 1), 1) if maximum else [1, 1]
    ceil = args.get("ceil_mode", False)
    include = args.get("count_include_pad", True)
    divisor = args.get("divisor_override")
    if (
        type(ceil) is not bool
        or type(include) is not bool
        or any(p > k // 2 for p, k in zip(padding, kernel))
        or (divisor is not None and (type(divisor) is not int or divisor == 0))
    ):
        raise ValueError("Invalid pooling settings.")
    expected = []
    for n, k, s, p, d in zip(image.shape[-2:], kernel, stride, padding, dilation):
        size = (n + 2 * p - d * (k - 1) - 1 + (s - 1 if ceil else 0)) // s + 1
        if ceil and (size - 1) * s >= n + p:
            size -= 1
        expected.append(size)
    if expected != output.shape[-2:]:
        raise ValueError("Window settings do not match the recorded shape.")
    return {
        "mode": "max" if maximum else "average",
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "ceil": ceil,
        "include_padding": include,
        "divisor": divisor,
    }


def describe_pooling(kind, args, inputs, outputs):
    p = pooling_spec(kind, args, inputs, outputs)
    if p["mode"] == "adaptive":
        title = "Average adaptive regions"
        summary = "Choose each region from the input and requested output sizes, then average its real cells."
        detail = "Region start = floor(output index × input size / output size); end = ceil((output index + 1) × input size / output size). Adjacent regions can overlap and differ in size. Batch and channel stay unchanged."
    elif p["mode"] == "max":
        title = "Keep each window's maximum"
        summary = "Select the largest input value in each spatial window, independently for every batch and channel."
        detail = "Stride moves the window and dilation spaces its samples. Out-of-bounds samples read negative infinity, not zero. Recorded indices, when returned, identify the selected input; equal maxima alone do not prove which tied cell PyTorch chose."
    else:
        title = "Average each spatial window"
        summary = "Sum the real input values, then divide by the configured window count or explicit divisor."
        detail = "Declared padding contributes zeros. count_include_pad chooses whether those positions enter the divisor. Ceil-mode overhang beyond the declared padding is excluded even when padding is counted. divisor_override replaces the count when provided. Batch and channel stay unchanged."
    return Lesson(
        title=title, summary=summary, detail=detail, category="compute", interaction="pooling"
    )
