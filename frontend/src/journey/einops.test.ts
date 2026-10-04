import { expect, test } from "vitest";
import type { AxisPart } from "../tensors/axisLineage";
import { einopsPattern } from "./einops";

const whole = (axis: number, size: number): AxisPart => ({ axis, size });
const piece = (
  axis: number,
  size: number,
  index: number,
  of: number,
): AxisPart => ({
  axis,
  size,
  piece: { index, of },
});

test("a head split reads as einops writes it, sizes for all but the last piece", () => {
  // [batch 2, tokens 5, features 8] → [batch, heads 2, tokens, head_dim 4]
  const found = einopsPattern(
    [2, 5, 8],
    ["batch", "tokens", "features"],
    [2, 2, 5, 4],
    ["batch", "heads", "tokens", "head_dim"],
    [[whole(0, 2)], [piece(2, 2, 0, 2)], [whole(1, 5)], [piece(2, 4, 1, 2)]],
  )!;
  expect(found.pattern).toBe(
    "batch tokens (heads head_dim) -> batch heads tokens head_dim",
  );
  expect(found.call("x")).toBe(
    "rearrange(x, 'batch tokens (heads head_dim) -> batch heads tokens head_dim', heads=2)",
  );
});

test("merges, unnamed axes, and size-1 axes", () => {
  // [1, 3, 4, 4] flattened to [1, 3, 16] and an axis of 1 dropped
  const flat = einopsPattern(
    [1, 3, 4, 4],
    [null, "channels", "height", "width"],
    [3, 16],
    [null, null],
    [[whole(1, 3)], [whole(2, 4), whole(3, 4)]],
  )!;
  expect(flat.pattern).toBe(
    "1 channels height width -> channels (height width)",
  );
  expect(flat.sizes).toEqual([]);
  // An axis of 1 added: "1" on the right.
  const added = einopsPattern(
    [5],
    [null],
    [1, 5],
    [null, null],
    [[], [whole(0, 5)]],
  )!;
  expect(added.pattern).toBe("a0 -> 1 a0");
  // An axis of more than one left out is not a rearrangement.
  expect(
    einopsPattern([2, 3], [null, null], [3], [null], [[whole(1, 3)]]),
  ).toBeNull();
});
