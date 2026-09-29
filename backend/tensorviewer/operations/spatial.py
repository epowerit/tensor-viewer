"""Spatial lessons validated against recorded operands and convolution arguments."""

from ..models import Lesson, TensorState


def pair(value):
    return [value, value] if isinstance(value, int) else list(value)


def describe_spatial(kind: str, args: dict, inputs: list[TensorState], outputs: list[TensorState]):
    image, weight = inputs[:2]
    output = outputs[0]
    if all(len(t.shape) == 4 for t in (image, weight, output)):
        b, c, h, w = image.shape
        d, channels, ph, pw = weight.shape
        # Only a complete, unpadded, ungrouped tiling can be described as this
        # patch-to-token projection. Overlapping/general convolutions stay generic.
        if (
            ph > 0
            and pw > 0
            and c == channels
            and args.get("groups", 1) == 1
            and pair(args.get("stride", 1)) == [ph, pw]
            and pair(args.get("padding", 0)) == [0, 0]
            and pair(args.get("dilation", 1)) == [1, 1]
            and h % ph == 0
            and w % pw == 0
            and output.shape == [b, d, h // ph, w // pw]
            and all(n > 0 for n in image.shape + output.shape)
        ):
            return Lesson(
                title="Project spatial patches",
                summary=f"Each {ph} × {pw} patch across {c} channels becomes a vector of {d} features.",
                detail=(
                    "The convolution uses a stride equal to its kernel size, with no padding, "
                    "dilation or channel groups. Each output location uses exactly one spatial "
                    "patch. Each feature sums input × weight over every channel and patch "
                    "position, plus bias when present. Flattening the spatial grid and then "
                    "transposing can arrange those vectors as token rows without changing values."
                ),
                category="compute",
                interaction="patch_projection",
                patch_size=[ph, pw],
            )
    return Lesson(
        title="Apply a spatial convolution",
        summary="Combine spatial neighborhoods with the recorded kernel weights.",
        detail=(
            "This convolution is not a complete, non-overlapping patch projection. "
            "Inspect its recorded tensors and arguments; padding, overlap, dilation or "
            "channel groups may affect its receptive fields."
        ),
        category="compute",
    )
