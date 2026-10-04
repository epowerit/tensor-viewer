import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { broadcastReuse, reuseText } from "./broadcast";

const tensor = (
  id: string,
  name: string,
  shape: number[],
  axes: string[] = [],
) =>
  ({
    id,
    name,
    shape,
    axes,
    numel: shape.reduce((a, b) => a * b, 1),
  }) as unknown as Tensor;
const op = (kind: string, inputs: string[], output: string) =>
  ({ id: "op", kind, inputs, outputs: [output] }) as unknown as Operation;

test("a bias added at every token is reused once per token", () => {
  const tensors = {
    x: tensor("x", "x", [2, 4, 8], ["batch", "tokens", "features"]),
    b: tensor("b", "bias", [8]),
    y: tensor("y", "y", [2, 4, 8], ["batch", "tokens", "features"]),
  };
  const reuse = broadcastReuse(op("add", ["x", "b"], "y"), tensors)!;
  expect(reuse.factor).toBe(8);
  expect(reuseText(reuse)).toBe("×8 bias over batch, tokens");
});

test("a size-1 axis repeats too, an axis of 1 in the result does not; same shapes, scalars and other steps do not count", () => {
  const tensors = {
    s: tensor(
      "s",
      "scores",
      [1, 2, 4, 4],
      ["batch", "heads", "queries", "keys"],
    ),
    m: tensor("m", "mask", [4, 4]),
    k: tensor("k", "keep", [1, 1, 4, 4]),
    c: tensor("c", "scale", []),
    o: tensor("o", "out", [1, 2, 4, 4], ["batch", "heads", "queries", "keys"]),
  };
  expect(
    reuseText(broadcastReuse(op("masked_fill", ["s", "m"], "o"), tensors)!),
  ).toBe("×2 mask over heads");
  expect(broadcastReuse(op("mul", ["s", "k"], "o"), tensors)?.axes).toEqual([
    "heads",
  ]);
  expect(broadcastReuse(op("add", ["s", "s"], "o"), tensors)).toBeNull();
  expect(broadcastReuse(op("div", ["s", "c"], "o"), tensors)).toBeNull();
  expect(broadcastReuse(op("matmul", ["s", "m"], "o"), tensors)).toBeNull();
});
