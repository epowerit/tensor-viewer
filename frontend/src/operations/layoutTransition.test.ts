import { describe, expect, it } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { ravel } from "../tensors/coordinates";
import {
  CELL_SIZE,
  layoutTransition,
  transitionPoint,
  transitionScene,
  transitionStep,
  transitionWindow,
} from "./layoutTransition";

function tensor(shape: number[], overrides: Partial<Tensor> = {}): Tensor {
  return {
    id: "t",
    name: "x",
    shape,
    axes: [],
    values: [],
    dtype: "torch.float32",
    numel: shape.reduce((a, b) => a * b, 1),
    storage_id: "s",
    storage_offset: 0,
    strides: shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1)),
    contiguous: true,
    minimum: null,
    maximum: null,
    role: "intermediate",
    value_source: "shape",
    ...overrides,
  };
}
function op(
  kind: string,
  rule: "identity" | "permutation" | "unfold" = "identity",
  order: number[] | null = null,
): Operation {
  return {
    kind,
    status: "ok",
    lesson: {
      interaction: "mapping",
      mapping_rule: rule,
      axis_order: order,
      mapping: null,
    },
  } as Operation;
}

describe("value-preserving layout replay", () => {
  it("decodes the same logical index using new dimension boundaries", () => {
    const m = layoutTransition(
      op("reshape"),
      tensor([2, 3, 8]),
      tensor([2, 3, 2, 4]),
      39,
    )!;
    expect(m.before).toEqual([1, 1, 7]);
    expect(m.after).toEqual([1, 1, 1, 3]);
    expect(m.source).toBe(39);
    expect(m.sourceStorage).toBe(m.targetStorage);
  });
  it("permutes coordinates without moving the element in shared storage", () => {
    const input = tensor([2, 3, 4]);
    const output = tensor([2, 4, 3], {
      strides: [12, 1, 4],
      contiguous: false,
    });
    const m = layoutTransition(
      op("permute", "permutation", [0, 2, 1]),
      input,
      output,
      ravel([1, 3, 1], output.shape),
    )!;
    expect(m.before).toEqual([1, 1, 3]);
    expect(m.after).toEqual([1, 3, 1]);
    expect(m.source).toBe(19);
    expect(m.target).toBe(22);
    expect(m.sharedStorage).toBe(true);
    expect(m.sourceStorage).toBe(19);
    expect(m.targetStorage).toBe(19);
  });
  it("keeps logical order separate from noncontiguous storage on a copy", () => {
    const m = layoutTransition(
      op("contiguous"),
      tensor([4, 3], { strides: [1, 4], contiguous: false, storage_offset: 2 }),
      tensor([4, 3], { storage_id: "copy" }),
      7,
    )!;
    expect(m.before).toEqual(m.after);
    expect(m.source).toBe(m.target);
    expect(m.sharedStorage).toBe(false);
    expect(m.sourceStorage).toBe(8);
    expect(m.targetStorage).toBe(7);
  });
  it("handles unit-axis insertion and scalar tensors", () => {
    const m = layoutTransition(op("unsqueeze"), tensor([]), tensor([1]), 0)!;
    expect(m.before).toEqual([]);
    expect(m.after).toEqual([0]);
    expect(transitionWindow(m.input, 0).cells).toEqual([
      { index: 0, coordinates: [], row: 0, column: 0 },
    ]);
  });
  it("supports compatible older explicit mappings", () => {
    const operation = op("reshape");
    operation.lesson.mapping_rule = null;
    operation.lesson.mapping = [0, 1, 2, 3, 4, 5];
    expect(
      layoutTransition(operation, tensor([2, 3]), tensor([3, 2]), 4)?.before,
    ).toEqual([1, 1]);
    operation.lesson.mapping[4] = 3;
    expect(
      layoutTransition(operation, tensor([2, 3]), tensor([3, 2]), 4),
    ).toBeNull();
  });
  it("does not animate computation, dtype reinterpretation, or one-to-many windows", () => {
    expect(
      layoutTransition(op("linear"), tensor([3]), tensor([3]), 1),
    ).toBeNull();
    expect(
      layoutTransition(op("unfold", "unfold"), tensor([3]), tensor([3]), 1),
    ).toBeNull();
    expect(
      layoutTransition(
        op("view"),
        tensor([3]),
        tensor([3], { dtype: "torch.int32" }),
        1,
      ),
    ).toBeNull();
    expect(
      layoutTransition(
        { ...op("view"), status: "error" },
        tensor([3]),
        tensor([3]),
        1,
      ),
    ).toBeNull();
  });
  it("rejects empty tensors, invalid indices, and unverified axis orders", () => {
    for (const index of [-1, 6, NaN, 0.5])
      expect(
        layoutTransition(op("reshape"), tensor([2, 3]), tensor([3, 2]), index),
      ).toBeNull();
    expect(
      layoutTransition(op("reshape"), tensor([0, 3]), tensor([3, 0]), 0),
    ).toBeNull();
    expect(
      layoutTransition(
        op("transpose", "permutation", [0, 0]),
        tensor([3, 3]),
        tensor([3, 3]),
        0,
      ),
    ).toBeNull();
    expect(
      layoutTransition(
        op("transpose", "permutation", [1, 0]),
        tensor([2, 3]),
        tensor([2, 3]),
        0,
      ),
    ).toBeNull();
  });
});

describe("bounded teaching windows and animation positions", () => {
  it("shows complete small axes without gaps or phantom cells", () => {
    const w = transitionWindow(tensor([2, 3]), 5);
    expect(w.cells).toHaveLength(6);
    expect(w.cells.map((c) => c.index)).toEqual([0, 1, 2, 3, 4, 5]);
    expect([w.selectedRow, w.selectedColumn]).toEqual([1, 2]);
  });
  it("includes the exact final element of a billion-element tensor in a bounded window", () => {
    const t = tensor([1024, 1024, 1024]);
    const w = transitionWindow(t, t.numel - 1);
    expect(w.cells.length).toBeLessThanOrEqual(32);
    expect(w.prefix).toEqual([1023]);
    expect(w.cells.at(-1)?.coordinates).toEqual([1023, 1023, 1023]);
    expect(w.cells.at(-1)?.index).toBe(t.numel - 1);
    expect(w.startRow).toBe(1020);
    expect(w.startColumn).toBe(1016);
  });
  it("handles vectors with no fictional row coordinate", () => {
    const w = transitionWindow(tensor([9]), 8);
    expect(w.rows).toBe(1);
    expect(w.cells.map((c) => c.coordinates)).toEqual([[8]]);
  });
  it("places endpoints inside their actual selected cells in both orientations", () => {
    const m = layoutTransition(
      op("reshape"),
      tensor([2, 3, 8]),
      tensor([2, 3, 2, 4]),
      39,
    )!;
    for (const vertical of [true, false]) {
      const s = transitionScene(m, vertical);
      const inputCell = s.before.cells.find((c) => c.index === m.source)!;
      const outputCell = s.after.cells.find((c) => c.index === m.target)!;
      expect(s.start.x).toBe(
        s.inputOrigin.x + (inputCell.column + 0.5) * CELL_SIZE,
      );
      expect(s.end.y).toBe(
        s.outputOrigin.y + (outputCell.row + 0.5) * CELL_SIZE,
      );
      for (const p of [s.start, s.middle, s.end]) {
        expect(p.x).toBeGreaterThan(0);
        expect(p.x).toBeLessThan(s.width);
        expect(p.y).toBeLessThan(s.height);
      }
      expect(transitionPoint(s.start, s.middle, s.end, 0)).toEqual(s.start);
      expect(transitionPoint(s.start, s.middle, s.end, 0.5)).toEqual(s.middle);
      expect(transitionPoint(s.start, s.middle, s.end, 1)).toEqual(s.end);
    }
  });
  it("steps in either direction from arbitrary scrub positions", () => {
    expect(transitionStep(0.1, 1)).toBe(0.5);
    expect(transitionStep(0.5, 1)).toBe(1);
    expect(transitionStep(0.9, -1)).toBe(0.5);
    expect(transitionStep(0.5, -1)).toBe(0);
    expect(transitionStep(0, -1)).toBe(0);
    expect(transitionStep(1, 1)).toBe(1);
  });
});
