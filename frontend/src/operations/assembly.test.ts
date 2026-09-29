import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import {
  locatePart,
  locateWhole,
  tensorAssembly,
  visibleParts,
} from "./assembly";
import { product } from "../tensors/coordinates";

function tensor(id: string, shape: number[], values: number[] = []): Tensor {
  return {
    id,
    name: id,
    shape,
    values,
    axes: [],
    strides: [],
    storage_id: id,
    storage_offset: 0,
    contiguous: true,
    dtype: "torch.float32",
    numel: product(shape),
    minimum: null,
    maximum: null,
    role: "intermediate",
    value_source: values.length ? "inline" : "shape",
  };
}
function fixture(
  kind: string,
  axis: number,
  wholeShape: number[],
  partShapes: number[][],
  wholeValues: number[] = [],
  partValues: number[][] = [],
) {
  const whole = tensor("whole", wholeShape, wholeValues);
  const parts = partShapes.map((s, i) => tensor(`part${i}`, s, partValues[i]));
  const joining = ["cat", "concat", "concatenate", "stack"].includes(kind);
  const op: Operation = {
    id: "assembly",
    index: 0,
    function: `torch.${kind}`,
    source: null,
    module: "",
    error: null,
    lesson: {
      title: kind,
      summary: "",
      detail: "",
      category: "layout",
      interaction: "tensor_assembly",
      mapping: null,
      axis_order: null,
    },
    kind,
    status: "ok",
    arguments: { dim: axis },
    inputs: joining ? parts.map((p) => p.id) : [whole.id],
    outputs: joining ? [whole.id] : parts.map((p) => p.id),
  };
  const run = {
    trace: {
      tensors: Object.fromEntries([whole, ...parts].map((t) => [t.id, t])),
      operations: [op],
    },
  } as Run;
  return { op, run, p: tensorAssembly(run, op)! };
}
const examples: [string, number, number[], number[][], number[], number[][]][] =
  [
    [
      "cat",
      1,
      [2, 3],
      [
        [2, 2],
        [2, 1],
      ],
      [0, 1, 10, 2, 3, 11],
      [
        [0, 1, 2, 3],
        [10, 11],
      ],
    ],
    [
      "stack",
      1,
      [2, 2, 2],
      [
        [2, 2],
        [2, 2],
      ],
      [0, 1, 10, 11, 2, 3, 12, 13],
      [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
      ],
    ],
    [
      "stack",
      -1,
      [2, 2, 2],
      [
        [2, 2],
        [2, 2],
      ],
      [0, 10, 1, 11, 2, 12, 3, 13],
      [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
      ],
    ],
    [
      "split",
      1,
      [2, 3],
      [
        [2, 1],
        [2, 2],
      ],
      [0, 1, 2, 3, 4, 5],
      [
        [0, 3],
        [1, 2, 4, 5],
      ],
    ],
    [
      "chunk",
      1,
      [2, 6],
      [
        [2, 2],
        [2, 2],
        [2, 2],
      ],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      [
        [0, 1, 6, 7],
        [2, 3, 8, 9],
        [4, 5, 10, 11],
      ],
    ],
    [
      "unbind",
      1,
      [2, 3, 2],
      [
        [2, 2],
        [2, 2],
        [2, 2],
      ],
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      [
        [0, 1, 6, 7],
        [2, 3, 8, 9],
        [4, 5, 10, 11],
      ],
    ],
  ];

describe("ordered tensor joins and partitions", () => {
  it.each(examples)(
    "maps every recorded cell for %s at axis %i",
    (kind, axis, shape, parts, values, partValues) => {
      const { p } = fixture(kind, axis, shape, parts, values, partValues);
      expect(p).not.toBeNull();
      for (let i = 0; i < p.whole.numel; i++) {
        const found = locatePart(p, i);
        expect(p.parts[found.part].values[found.index]).toBe(p.whole.values[i]);
        expect(locateWhole(p, found.part, found.index).index).toBe(i);
      }
      p.parts.forEach((part, member) => {
        for (let index = 0; index < part.numel; index++) {
          const found = locatePart(p, locateWhole(p, member, index).index);
          expect([found.part, found.index]).toEqual([member, index]);
        }
      });
    },
  );
  it("skips empty ranges and retains scalar stack/unbind coordinates", () => {
    const { p } = fixture(
      "cat",
      1,
      [2, 3],
      [
        [2, 0],
        [2, 1],
        [2, 0],
        [2, 2],
        [2, 0],
      ],
    );
    expect(locatePart(p, 0)).toEqual({
      part: 1,
      index: 0,
      coordinates: [0, 0],
    });
    expect(locatePart(p, 5)).toEqual({
      part: 3,
      index: 3,
      coordinates: [1, 1],
    });
    for (const kind of ["stack", "unbind"]) {
      const scalar = fixture(kind, 0, [2], [[], []]).p;
      expect(locatePart(scalar, 1)).toEqual({
        part: 1,
        index: 0,
        coordinates: [],
      });
      expect(locateWhole(scalar, 1, 0)).toEqual({ index: 1, coordinates: [1] });
    }
  });
  it("treats repeated operands as separate positions", () => {
    const { op, run } = fixture(
      "stack",
      1,
      [2, 2, 3],
      [
        [2, 3],
        [2, 3],
      ],
    );
    op.inputs = ["part0", "part0"];
    const p = tensorAssembly(run, op)!;
    expect(locatePart(p, 11).part).toBe(1);
    expect(locateWhole(p, 0, 5).index).toBe(8);
    expect(locateWhole(p, 1, 5).index).toBe(11);
  });
  it("uses logical coordinates regardless of strides, with bounded billion-element navigation", () => {
    const { p } = fixture(
      "cat",
      -1,
      [1024, 1024, 1536],
      [
        [1024, 1024, 768],
        [1024, 1024, 768],
      ],
    );
    p.parts[1].contiguous = false;
    p.parts[1].strides = [1, 786432, 1024];
    p.parts[1].storage_offset = 30;
    expect(locatePart(p, p.whole.numel - 1)).toEqual({
      part: 1,
      index: 1024 * 1024 * 768 - 1,
      coordinates: [1023, 1023, 767],
    });
    expect(locateWhole(p, 1, p.parts[1].numel - 1).index).toBe(
      p.whole.numel - 1,
    );
    expect(p.offsets).toEqual([0, 768]);
    expect(visibleParts(1000000, 512)).toEqual([0, 1, 511, 512, 513, 999999]);
    expect(visibleParts(4, 2)).toEqual([0, 1, 2, 3]);
  });
  it("accepts aliases and legacy positional dimensions without a guessed shape axis", () => {
    for (const kind of ["cat", "concat", "concatenate"]) {
      const { op, run } = fixture(
        kind,
        -1,
        [2, 3],
        [
          [2, 1],
          [2, 2],
        ],
      );
      op.arguments = { arg1: -1 };
      expect(tensorAssembly(run, op)?.axis).toBe(1);
      op.arguments = {};
      expect(tensorAssembly(run, op)).toBeNull(); // default dim 0 cannot explain these shapes
    }
    const { op, run } = fixture(
      "chunk",
      1,
      [2, 6],
      [
        [2, 2],
        [2, 2],
        [2, 2],
      ],
    );
    op.arguments = { arg1: 4, arg2: 1 };
    expect(tensorAssembly(run, op)?.parts).toHaveLength(3);
  });
  it("rejects failed/mutating, promoted-dtype, incomplete and mismatched traces", () => {
    const { op, run } = fixture(
      "cat",
      1,
      [2, 3],
      [
        [2, 1],
        [2, 2],
      ],
    );
    for (const patch of [
      { status: "error" },
      { kind: "add" },
      { inputs: ["missing"] },
      { outputs: [] },
      { arguments: { dim: 2 } },
      { arguments: { dim: 0.5 } },
      { arguments: { dim: 1, out: "tensor" } },
      { mutations: [{ before: "part0", after: "whole", kind: "write" }] },
    ])
      expect(tensorAssembly(run, { ...op, ...patch } as Operation)).toBeNull();
    run.trace.tensors.part0.dtype = "torch.int64";
    expect(tensorAssembly(run, op)).toBeNull();
    run.trace.tensors.part0.dtype = "torch.float32";
    run.trace.tensors.part0.shape = [2, 0];
    run.trace.tensors.part0.numel = 0;
    expect(tensorAssembly(run, op)).toBeNull();
    const stacked = fixture("stack", 0, [3, 2], [[2], [2]]);
    expect(tensorAssembly(stacked.run, stacked.op)).toBeNull();
    const mixed = fixture("cat", 0, [2, 3], [[0], [2, 3]]);
    expect(tensorAssembly(mixed.run, mixed.op)).toBeNull();
  });
});
