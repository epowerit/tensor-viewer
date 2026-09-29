import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { ravel } from "../tensors/coordinates";
import {
  contributionSum,
  finiteProduct,
  linearInputSelection,
  linearProjection,
  linearSelection,
  linearWeightSelection,
  linearWindow,
} from "./linear";

function tensor(id: string, shape: number[]): Tensor {
  return {
    id,
    name: id,
    shape,
    axes: [],
    values: [],
    strides: [],
    storage_offset: 0,
    storage_id: id,
    contiguous: true,
    dtype: "torch.float32",
    numel: shape.reduce((a, b) => a * b, 1),
    minimum: null,
    maximum: null,
    role: "intermediate",
    value_source: "shape",
  };
}
function fixture(shape = [2, 3, 4], features = 5, bias = true) {
  const tensors = Object.fromEntries(
    [
      tensor("x", shape),
      tensor("w", [features, shape.at(-1)!]),
      tensor("b", [features]),
      tensor("y", [...shape.slice(0, -1), features]),
    ].map((t) => [t.id, t]),
  );
  const op = {
    kind: "linear",
    status: "ok",
    inputs: ["x", "w", ...(bias ? ["b"] : [])],
    outputs: ["y"],
  } as Operation;
  const run = { trace: { tensors, operations: [op] } } as Run;
  return { run, op, p: linearProjection(run, op)! };
}

describe("linear projection coordinates", () => {
  it("uses weight rows, keeps batch/token coordinates, and supports older runs", () => {
    const { p } = fixture();
    const selected = linearSelection(p, ravel([1, 2, 3], p.output.shape), 2);
    expect(selected.inputCoordinates).toEqual([1, 2, 2]);
    expect(selected.weightCoordinates).toEqual([3, 2]);
    expect(selected.input).toBe(22);
    expect(selected.weight).toBe(14);
    expect(selected.coordinates).toEqual([1, 2, 3]);
  });
  it("selects an input vector while retaining the chosen output feature", () => {
    const { p } = fixture();
    expect(linearInputSelection(p, 23, 3)).toEqual({ output: 28, term: 3 });
  });
  it("selects a weight row while retaining the chosen batch and token", () => {
    const { p } = fixture();
    expect(linearWeightSelection(p, 18, 28)).toEqual({ output: 29, term: 2 });
  });
  it("handles a rank-one vector without inventing leading dimensions", () => {
    const { p } = fixture([4], 5, false);
    expect(p.bias).toBeUndefined();
    expect(linearSelection(p, 3, 2).inputCoordinates).toEqual([2]);
    expect(linearInputSelection(p, 1, 3)).toEqual({ output: 3, term: 1 });
    expect(linearWeightSelection(p, 18, 3)).toEqual({ output: 4, term: 2 });
  });
  it("maps logical indices independently of noncontiguous storage", () => {
    const { p } = fixture();
    p.input.contiguous = false;
    p.input.strides = [4, 8, 1];
    p.input.storage_offset = 2;
    expect(linearSelection(p, 28, 2).input).toBe(22);
  });
  it("bounds the final window for enormous tensors", () => {
    const { p } = fixture([1024, 1024, 999999], 768);
    const window = linearWindow(p, p.output.numel - 1, 999998);
    expect(window).toHaveLength(7);
    expect(window.at(-1)!.inputCoordinates).toEqual([1023, 1023, 999998]);
    expect(window.at(-1)!.weightCoordinates).toEqual([767, 999998]);
    expect(window.at(-1)!.input).toBe(p.input.numel - 1);
  });
  it("rejects unsupported shapes, missing operands, and failed operations", () => {
    const { run, op } = fixture();
    expect(linearProjection(run, { ...op, status: "error" })).toBeNull();
    run.trace.tensors.b.shape = [];
    expect(linearProjection(run, op)).toBeNull();
    delete run.trace.tensors.b;
    expect(linearProjection(run, op)).toBeNull();
    op.inputs = ["x", "w"];
    run.trace.tensors.w.shape = [4];
    expect(linearProjection(run, op)).toBeNull();
    run.trace.tensors.w.shape = [5, 4];
    run.trace.tensors.y.shape = [3, 2, 5];
    expect(linearProjection(run, op)).toBeNull();
    run.trace.tensors.y.shape = [2, 3, 5];
    run.trace.tensors.x.numel = 0;
    expect(linearProjection(run, op)).toBeNull();
  });
});

describe("recorded contribution arithmetic", () => {
  it("reconstructs a feature using captured inputs, weights, and bias", () => {
    const x = [2, -3, 4],
      w = [0.5, 2, -1],
      bias = 7;
    const products = x.map((value, k) => finiteProduct(value, w[k]));
    expect(products).toEqual([1, -6, -4]);
    expect(contributionSum(products)! + bias).toBe(-2);
  });
  it("never turns unavailable, non-finite, or overflowing data into a result", () => {
    expect(finiteProduct(undefined, 2)).toBeUndefined();
    expect(finiteProduct("Infinity", 2)).toBeUndefined();
    expect(finiteProduct(1e308, 2)).toBeUndefined();
    expect(contributionSum([2, undefined])).toBeUndefined();
    expect(contributionSum([NaN])).toBeUndefined();
    expect(contributionSum([1e308, 1e308])).toBeUndefined();
    expect(contributionSum([])).toBeUndefined();
  });
});
