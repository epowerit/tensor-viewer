import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import {
  findPatchJourney,
  patchCell,
  patchGridWindow,
  patchPosition,
} from "./patches";

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
function operation(
  id: string,
  kind: string,
  inputs: string[],
  output: string,
  lesson: Partial<Operation["lesson"]>,
): Operation {
  return {
    id,
    index: 0,
    kind,
    inputs,
    outputs: [output],
    arguments: {},
    function: kind,
    module: "PatchEmbedding",
    source: null,
    status: "ok",
    error: null,
    lesson: {
      title: kind,
      summary: "",
      detail: "",
      category: "layout",
      interaction: "inspect",
      mapping: null,
      axis_order: null,
      ...lesson,
    },
  };
}
function fixture(image = [2, 3, 6, 8], patch = [2, 4], features = 5) {
  const [batch, channels, height, width] = image;
  const rows = height / patch[0],
    columns = width / patch[1];
  const tensors = Object.fromEntries(
    [
      tensor("x", image),
      tensor("w", [features, channels, ...patch]),
      tensor("b", [features]),
      tensor("p", [batch, features, rows, columns]),
      tensor("f", [batch, features, rows * columns]),
      tensor("t", [batch, rows * columns, features]),
    ].map((t) => [t.id, t]),
  );
  const operations = [
    operation("project", "conv2d", ["x", "w", "b"], "p", {
      interaction: "patch_projection",
      patch_size: patch,
    }),
    operation("flatten", "flatten", ["p"], "f", { mapping_rule: "identity" }),
    operation("tokens", "transpose", ["f"], "t", {
      mapping_rule: "permutation",
      axis_order: [0, 2, 1],
    }),
  ];
  return { trace: { tensors, operations } } as Run;
}

describe("exact patch membership", () => {
  it("links the same recorded sequence from any of its three operations", () => {
    const run = fixture();
    for (const op of run.trace.operations) {
      const journey = findPatchJourney(run, op)!;
      expect(journey.image.id).toBe("x");
      expect(journey.tokens?.id).toBe("t");
      expect(journey.patch).toEqual([2, 4]);
    }
  });
  it("maps rectangular patches across batch, channel and embedding axes", () => {
    const run = fixture(),
      j = findPatchJourney(run, run.trace.operations[0])!;
    expect(patchPosition(j, 3)).toEqual({ row: 1, column: 1, y: 2, x: 4 });
    expect(patchCell(j, 1, 3, 2, 1, 1, 2)).toEqual({
      input: 222,
      inputCoordinates: [1, 1, 3, 6],
      weight: 62,
      weightCoordinates: [2, 1, 1, 2],
      projected: 45,
      token: 47,
      flattened: 45,
      patchVectorIndex: 14,
    });
    const all = Array.from(
      { length: 24 },
      (_, i) =>
        patchCell(j, 1, 3, 2, Math.floor(i / 8), Math.floor((i % 8) / 4), i % 4)
          .input,
    );
    expect(new Set(all).size).toBe(24);
    expect(all).toEqual([
      164, 165, 166, 167, 172, 173, 174, 175, 212, 213, 214, 215, 220, 221, 222,
      223, 260, 261, 262, 263, 268, 269, 270, 271,
    ]);
  });
  it("requires real dependencies and axis permutations, not matching shapes", () => {
    const run = fixture();
    run.trace.operations[2].inputs = ["unrelated"];
    expect(findPatchJourney(run, run.trace.operations[2])).toBeNull();
    expect(
      findPatchJourney(run, run.trace.operations[0])?.tokens,
    ).toBeUndefined();
    run.trace.operations[2].inputs = ["f"];
    run.trace.operations[2].lesson.axis_order = [0, 1, 2];
    expect(findPatchJourney(run, run.trace.operations[2])).toBeNull();
    run.trace.operations[0].lesson.patch_size = null;
    expect(findPatchJourney(run, run.trace.operations[0])).toBeNull();
  });
  it("does not confuse an unrelated reshape branch with the token branch", () => {
    const run = fixture();
    run.trace.tensors.branch = tensor("branch", [2, 5, 6]);
    run.trace.operations.splice(
      1,
      0,
      operation("branch", "reshape", ["p"], "branch", {
        mapping_rule: "identity",
      }),
    );
    expect(findPatchJourney(run, run.trace.operations[0])?.tokens?.id).toBe(
      "t",
    );
    expect(
      findPatchJourney(run, run.trace.operations[1])?.tokens,
    ).toBeUndefined();
  });
  it("keeps huge shape-only grids bounded and reaches the final image element", () => {
    const run = fixture([1024, 3, 1024, 1024], [16, 16], 768);
    const j = findPatchJourney(run, run.trace.operations[0])!;
    const grid = patchGridWindow(j, 4095);
    expect(grid.tokens).toHaveLength(64);
    expect(grid.startRow).toBe(56);
    expect(grid.startColumn).toBe(56);
    expect(grid.tokens.at(-1)).toBe(4095);
    const end = patchCell(j, 1023, 4095, 767, 2, 15, 15);
    expect(end.input).toBe(j.image.numel - 1);
    expect(end.token).toBe(j.tokens!.numel - 1);
    expect(end.weight).toBe(j.weight.numel - 1);
    expect(
      patchGridWindow(
        findPatchJourney(fixture(), fixture().trace.operations[0])!,
        5,
      ).tokens,
    ).toHaveLength(6);
  });
});
