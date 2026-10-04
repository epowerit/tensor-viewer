import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { bytesText, flopsText, stepBytes, stepFlops } from "./cost";

const tensor = (id: string, shape: number[], storage = id) =>
  ({
    id,
    shape,
    numel: shape.reduce((a, b) => a * b, 1),
    dtype: "float32",
    storage_id: storage,
  }) as unknown as Tensor;
const op = (kind: string, inputs: string[], outputs: string[]) =>
  ({ kind, inputs, outputs }) as unknown as Operation;

const tensors = {
  x: tensor("x", [2, 13, 16]),
  w: tensor("w", [64, 16]),
  b: tensor("b", [64]),
  y: tensor("y", [2, 13, 64]),
  v: tensor("v", [2, 13, 4, 16], "y"),
  image: tensor("image", [1, 3, 8, 8]),
  kernel: tensor("kernel", [6, 3, 3, 3]),
  conv: tensor("conv", [1, 6, 8, 8]),
};

test("matrix products count a multiply and an add per term; layout is free", () => {
  // 2 × 13 × 64 results, each summing 16 products, plus the bias.
  expect(stepFlops(op("linear", ["x", "w", "b"], ["y"]), tensors)).toBe(
    2 * 2 * 13 * 64 * 16 + 2 * 13 * 64,
  );
  expect(stepFlops(op("matmul", ["x", "w"], ["y"]), tensors)).toBe(
    2 * 2 * 13 * 64 * 16,
  );
  expect(stepFlops(op("view", ["y"], ["v"]), tensors)).toBe(0);
  // A 3 × 3 window over 3 channels for each of 6 × 8 × 8 results.
  expect(stepFlops(op("conv2d", ["image", "kernel"], ["conv"]), tensors)).toBe(
    2 * 6 * 64 * 27,
  );
  expect(stepFlops(op("softmax", ["y"], ["y"]), tensors)).toBe(2 * 13 * 64 * 6);
});

test("a view takes no new memory", () => {
  expect(stepBytes(op("linear", ["x", "w"], ["y"]), tensors)).toBe(
    2 * 13 * 64 * 4,
  );
  expect(stepBytes(op("view", ["y"], ["v"]), tensors)).toBe(0);
  expect(flopsText(53_248)).toBe("53 kFLOP");
  expect(flopsText(0)).toBe("0 FLOP");
  expect(bytesText(6656)).toBe("6.5 KB");
});
