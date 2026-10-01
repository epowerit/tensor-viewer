import { describe, expect, it } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { presenters } from "./presenters";

const tensor: Tensor = {
  id: "t",
  name: "x",
  shape: [1, 2],
  axes: ["rows", "columns"],
  values: [1, 2],
  strides: [2, 1],
  storage_offset: 0,
  storage_id: "s",
  contiguous: true,
  numel: 2,
  dtype: "torch.float32",
  role: "intermediate",
  minimum: 1,
  maximum: 2,
};
const operation: Operation = {
  id: "op",
  index: 0,
  kind: "contiguous",
  function: "Tensor.contiguous",
  inputs: ["t"],
  outputs: ["out"],
  arguments: {},
  source: null,
  module: "Example",
  status: "ok",
  error: null,
  lesson: {
    title: "Copy",
    summary: "Copy",
    detail: "Copy",
    category: "memory",
    interaction: "mapping",
    mapping: [0, 1],
    axis_order: null,
  },
};

describe("accurate tensor explanations", () => {
  it("keeps scalar softmax tensor details free of an invented empty group", () => {
    const scalar = { ...tensor, shape: [], numel: 1, values: [1] };
    const result = presenters.normalization(
      { ...operation, kind: "softmax", arguments: { dim: 0 } },
      [scalar],
      scalar,
      0,
    );
    expect(result.title).toBe("Inspect an element");
    expect(result.expression).toBeUndefined();
  });
  it("does not apply the old softmax presenter to ambiguous axes or dtype conversions", () => {
    for (const args of [
      {},
      { dim: 2 },
      { dim: -3 },
      { dim: true },
      { dim: 1, dtype: "torch.float64" },
    ]) {
      const result = presenters.normalization(
        { ...operation, kind: "softmax", arguments: args },
        [tensor],
        tensor,
        0,
      );
      expect(result.title).toBe("Inspect an element");
      expect(result.expression).toBeUndefined();
    }
  });
  it("does not claim that copying a tensor changes its coordinates", () => {
    const result = presenters.mapping(
      operation,
      [tensor],
      { ...tensor, id: "out", storage_id: "new" },
      1,
    );
    expect(result.text).toContain("coordinates are unchanged");
    expect(result.expression).toBe("[0, 1] → [0, 1]");
  });
  it("does not claim non-finite softmax scores form a finite probability group", () => {
    const result = presenters.normalization(
      { ...operation, kind: "softmax", arguments: { dim: -1 } },
      [{ ...tensor, values: ["Infinity", 0] }],
      { ...tensor, values: ["NaN", "NaN"] },
      0,
    );
    expect(result.leftHighlights).toEqual([0, 1]);
    expect(result.text).toContain("non-finite");
    expect(result.expression).toBeUndefined();
  });
  it("labels displayed rounded arithmetic as approximate and handles an empty contraction", () => {
    const result = presenters.dot_product(
      operation,
      [
        { ...tensor, shape: [1, 0], values: [], numel: 0 },
        { ...tensor, shape: [0, 1], values: [], numel: 0 },
      ],
      { ...tensor, shape: [1, 1], values: [0], numel: 1 },
      0,
    );
    expect(result.leftHighlights).toEqual([]);
    expect(result.rightHighlights).toEqual([]);
    expect(result.expression).toBe("0 ≈ 0");
  });
});
