import { expect, test } from "vitest";
import type { Tensor } from "../api/client";
import { changedCells } from "./mutation";

const tensor = (values: (number | string)[], extra: Partial<Tensor> = {}) =>
  ({
    shape: [values.length],
    numel: values.length,
    dtype: "float32",
    values,
    ...extra,
  }) as Tensor;

test("a write is summarized by the cells whose values differ", () => {
  expect(changedCells(tensor([1, 2, 3, 4]), tensor([1, 99, 3, 98]))).toEqual({
    indices: [1, 3],
    count: 2,
  });
  // A view on a disjoint slice shares storage but sees no change.
  expect(changedCells(tensor([1, 2]), tensor([1, 2]))).toEqual({
    indices: [],
    count: 0,
  });
  expect(changedCells(tensor(["nan", 1]), tensor(["nan", 2]))?.indices).toEqual(
    [1],
  );
  expect(changedCells(tensor([1, 2, 3]), tensor([4, 5, 6]), 2)).toEqual({
    indices: [0, 1],
    count: 3,
  });
});

test("snapshots that cannot be compared are not summarized", () => {
  const paged = tensor([], { numel: 9000, shape: [9000] });
  expect(changedCells(paged, paged)).toBeNull();
  expect(
    changedCells(tensor([1, 2]), tensor([1, 2], { dtype: "int64" })),
  ).toBeNull();
  expect(
    changedCells(tensor([1, 2, 3, 4]), tensor([1, 2, 3, 4], { shape: [2, 2] })),
  ).toBeNull();
  expect(changedCells(tensor([]), tensor([]))).toBeNull();
});
