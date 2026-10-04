import { expect, test } from "vitest";
import type { Evaluation } from "../api/client";
import { watchText } from "./WatchPanel";

test("a watch reads as a shape and first values, a scalar, or text", () => {
  const tensor = {
    kind: "tensor",
    shape: [1, 2, 13],
    dtype: "float32",
    numel: 26,
    values: Array(26).fill(1),
    names: [],
  } as unknown as Evaluation;
  expect(watchText(tensor)).toBe(
    "[1, 2, 13] float32 · 1, 1, 1, 1, 1, 1, 1, 1, …",
  );
  expect(watchText({ ...tensor, shape: [], numel: 1, values: [1.0021] })).toBe(
    "1.002 (float32)",
  );
  expect(
    watchText({
      kind: "value",
      text: "torch.Size([1, 2])",
      names: [],
    } as Evaluation),
  ).toBe("torch.Size([1, 2])");
});
