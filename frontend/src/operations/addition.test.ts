import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { ravel } from "../tensors/coordinates";
import {
  additionInputSelection,
  additionSelection,
  broadcastAxes,
  finiteAddition,
  tensorAddition,
} from "./addition";

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
function fixture(left = [2, 3, 4], right = [1, 3, 4], output = [2, 3, 4]) {
  const tensors = Object.fromEntries(
    [tensor("a", left), tensor("b", right), tensor("y", output)].map((t) => [
      t.id,
      t,
    ]),
  );
  const op = {
    kind: "add",
    status: "ok",
    inputs: ["a", "b"],
    outputs: ["y"],
    arguments: {},
  } as Operation;
  const run = { trace: { tensors, operations: [op] } } as Run;
  return { op, run, p: tensorAddition(run, op)! };
}

describe("broadcast addition", () => {
  it("maps positional contributions without mixing batches or tokens", () => {
    const { p } = fixture();
    expect(additionSelection(p, 23)).toEqual({
      coordinates: [1, 2, 3],
      left: { coordinates: [1, 2, 3], index: 23 },
      right: { coordinates: [0, 2, 3], index: 11 },
    });
    expect(broadcastAxes(p.right, p.output)).toEqual([0]);
    expect(broadcastAxes(p.left, p.output)).toEqual([]);
    expect(additionInputSelection(p, "right", 6, 23)).toBe(18);
  });
  it("matches both operands at the same coordinates for a residual sum", () => {
    const { p } = fixture([2, 3, 4], [2, 3, 4]);
    const selected = additionSelection(p, 23);
    expect(selected.left).toEqual(selected.right);
    expect(additionInputSelection(p, "left", 2, 23)).toBe(2);
    expect(additionInputSelection(p, "right", 5, 23)).toBe(5);
  });
  it("right-aligns different ranks and handles independent broadcast axes", () => {
    const { p } = fixture([2, 1, 4], [3, 1], [2, 3, 4]);
    expect(additionSelection(p, 23)).toEqual({
      coordinates: [1, 2, 3],
      left: { coordinates: [1, 0, 3], index: 7 },
      right: { coordinates: [2, 0], index: 2 },
    });
    expect(broadcastAxes(p.left, p.output)).toEqual([1]);
    expect(broadcastAxes(p.right, p.output)).toEqual([0, 2]);
    expect(additionInputSelection(p, "left", 2, 23)).toBe(
      ravel([0, 2, 2], p.output.shape),
    );
    expect(additionInputSelection(p, "right", 0, 23)).toBe(
      ravel([1, 0, 3], p.output.shape),
    );
  });
  it("handles scalar tensors with no invented axes", () => {
    const { p } = fixture([2, 3, 4], []);
    expect(additionSelection(p, 23).right).toEqual({
      coordinates: [],
      index: 0,
    });
    expect(additionInputSelection(p, "right", 0, 23)).toBe(23);
    const scalar = fixture([], [], []).p;
    expect(additionSelection(scalar, 0).coordinates).toEqual([]);
    expect(broadcastAxes(scalar.left, scalar.output)).toEqual([]);
  });
  it("retains alpha and supports older traces without the lesson capability", () => {
    const { op, run } = fixture();
    op.arguments.alpha = -2;
    expect(tensorAddition(run, op)?.alpha).toBe(-2);
    op.arguments.alpha = 0;
    expect(tensorAddition(run, op)?.alpha).toBe(0);
    op.inputs = ["a", "a"];
    expect(tensorAddition(run, op)?.left.id).toBe("a");
  });
  it("stays bounded at a billion elements and follows logical noncontiguous indices", () => {
    const { p } = fixture([1024, 1024, 768], [1, 1024, 768], [1024, 1024, 768]);
    p.left.contiguous = false;
    p.left.strides = [768, 1024 * 768, 1];
    p.left.storage_offset = 100;
    const selected = additionSelection(p, p.output.numel - 1);
    expect(selected.left.index).toBe(p.output.numel - 1);
    expect(selected.right.index).toBe(1024 * 768 - 1);
    expect(selected.right.coordinates).toEqual([0, 1023, 767]);
  });
  it("rejects mutations, incompatible outputs, missing operands, and empty tensors", () => {
    const { op, run } = fixture();
    expect(tensorAddition(run, { ...op, status: "error" })).toBeNull();
    expect(tensorAddition(run, { ...op, kind: "sub" })).toBeNull();
    expect(
      tensorAddition(run, {
        ...op,
        mutations: [{ before: "a", after: "y", kind: "write" }],
      }),
    ).toBeNull();
    expect(tensorAddition(run, { ...op, inputs: ["a"] })).toBeNull();
    expect(tensorAddition(run, { ...op, outputs: [] })).toBeNull();
    expect(
      tensorAddition(run, { ...op, arguments: { alpha: "NaN" } }),
    ).toBeNull();
    expect(
      tensorAddition(run, { ...op, arguments: { alpha: Infinity } }),
    ).toBeNull();
    run.trace.tensors.y.shape = [2, 4, 3];
    expect(tensorAddition(run, op)).toBeNull();
    run.trace.tensors.y.shape = [2, 3, 4];
    run.trace.tensors.b.shape = [2, 2, 4];
    expect(tensorAddition(run, op)).toBeNull();
    run.trace.tensors.b.shape = [1, 3, 4];
    run.trace.tensors.a.numel = 0;
    expect(tensorAddition(run, op)).toBeNull();
    delete run.trace.tensors.a;
    expect(tensorAddition(run, op)).toBeNull();
  });
  it("does not manufacture numbers from missing, non-finite, or exact-integer string values", () => {
    expect(finiteAddition(3, 4, -2)).toBe(-5);
    expect(finiteAddition(3, 4, 0)).toBe(3);
    expect(finiteAddition(1, undefined, 0)).toBeUndefined();
    expect(finiteAddition("9007199254740993", 1, 1)).toBeUndefined();
    expect(finiteAddition(NaN, 4, 1)).toBeUndefined();
    expect(finiteAddition(4, Infinity, 0)).toBeUndefined();
    expect(finiteAddition(1e308, 1e308, 1)).toBeUndefined();
  });
});
