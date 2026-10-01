import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { tensorRelation } from "./relations";
import {
  cellLayout,
  layoutMorph,
  relationMorph,
  MORPH_CELL,
  morphPosition,
  morphScene,
} from "./layoutMorph";

const tensor = (shape: number[], extra: Partial<Tensor> = {}) =>
  ({
    id: shape.join("x"),
    shape,
    numel: shape.reduce((a, b) => a * b, 1),
    dtype: "float32",
    ...extra,
  }) as Tensor;
const operation = (kind: string, lesson: object, extra: object = {}) =>
  ({
    kind,
    status: "ok",
    arguments: {},
    mutations: [],
    lesson: { interaction: "mapping", mapping: null, ...lesson },
    ...extra,
  }) as unknown as Operation;

test("cells follow row-major order with blocks for leading axes", () => {
  const grid = cellLayout([2, 3]);
  expect(grid.points[0]).toEqual({ x: 0, y: 0 });
  expect(grid.points[2]).toEqual({ x: 2 * MORPH_CELL, y: 0 });
  expect(grid.points[3]).toEqual({ x: 0, y: MORPH_CELL });
  expect([grid.columns, grid.rows]).toEqual([3, 2]);
  const blocks = cellLayout([2, 2, 3]);
  // The second block starts to the right of the first, after a gap.
  expect(blocks.points[6].x).toBeGreaterThan(3 * MORPH_CELL);
  expect(blocks.points[6].y).toBe(0);
  expect(cellLayout([]).points).toEqual([{ x: 0, y: 0 }]);
  expect(cellLayout([4]).width).toBe(4 * MORPH_CELL);
  const positions = cellLayout([2, 2, 2, 3]).points.map((p) => `${p.x},${p.y}`);
  expect(new Set(positions).size).toBe(24);
});

test("a permutation maps every output cell to one distinct input cell", () => {
  const morph = layoutMorph(
    operation("permute", { mapping_rule: "permutation", axis_order: [1, 0] }),
    tensor([2, 3]),
    tensor([3, 2]),
  )!;
  expect(morph.sources).toEqual([0, 3, 1, 4, 2, 5]);
  const scene = morphScene(morph, false);
  const start = morphPosition(morph, scene, 1, 0),
    end = morphPosition(morph, scene, 1, 1);
  expect(start).toEqual({
    x: scene.inputOrigin.x + morph.before.points[3].x,
    y: scene.inputOrigin.y + morph.before.points[3].y,
  });
  expect(end).toEqual({
    x: scene.outputOrigin.x + morph.after.points[1].x,
    y: scene.outputOrigin.y + morph.after.points[1].y,
  });
  expect(morphScene(morph, true).vertical).toBe(true);
});

test("reshape keeps logical order and roll wraps around", () => {
  const reshape = layoutMorph(
    operation("reshape", { mapping_rule: "identity" }),
    tensor([2, 6]),
    tensor([3, 4]),
  )!;
  expect(reshape.sources).toEqual([...Array(12).keys()]);
  const roll = layoutMorph(
    operation(
      "roll",
      { mapping_rule: "roll" },
      { arguments: { shifts: 1, dims: -1 } },
    ),
    tensor([4]),
    tensor([4]),
  )!;
  expect(roll.sources).toEqual([3, 0, 1, 2]);
});

test("unverified, oversized, or non-bijective layouts are not animated", () => {
  const identity = { mapping_rule: "identity" };
  const small = tensor([2, 3]);
  expect(
    layoutMorph(operation("reshape", identity), small, tensor([7])),
  ).toBeNull();
  expect(
    layoutMorph(operation("reshape", identity), tensor([300]), tensor([300])),
  ).toBeNull();
  // 64 cells in one row is too wide to read.
  expect(
    layoutMorph(operation("reshape", identity), tensor([64]), tensor([8, 8])),
  ).toBeNull();
  expect(
    layoutMorph(operation("unfold", { mapping_rule: "unfold" }), small, small),
  ).toBeNull();
  expect(
    layoutMorph(
      operation("reshape", { mapping: [0, 0, 1, 2, 3, 4] }),
      small,
      small,
    ),
  ).toBeNull();
  expect(
    layoutMorph(
      operation("view", identity),
      small,
      tensor([2, 3], { dtype: "int64" }),
    ),
  ).toBeNull();
  expect(
    layoutMorph(
      operation("reshape", identity, { status: "error" }),
      small,
      small,
    ),
  ).toBeNull();
  expect(layoutMorph(operation("add", identity), small, small)).toBeNull();
});

test("selection copies cells and reduction merges each group", () => {
  const input = tensor([2, 3]);
  const pick = operation("__getitem__", {
    interaction: "relation",
    relation: {
      rule: "index",
      operand: 0,
      axes: [
        { kind: "int", index: 1 },
        { kind: "slice", start: 1, step: 1, out: 0 },
      ],
    },
  });
  const row = tensor([2]);
  const selected = relationMorph(
    pick,
    tensorRelation(pick, [input], row),
    input,
    row,
  )!;
  expect(selected.kind).toBe("select");
  expect(selected.movers).toEqual([
    { source: 4, target: 0 },
    { source: 5, target: 1 },
  ]);
  const sum = operation("sum", {
    interaction: "relation",
    relation: { rule: "reduce", operand: 0, axes: [1], keepdim: false },
  });
  const totals = tensor([2]);
  const collapsed = relationMorph(
    sum,
    tensorRelation(sum, [input], totals),
    input,
    totals,
  )!;
  expect(collapsed.kind).toBe("collapse");
  expect(collapsed.movers.map((m) => m.target)).toEqual([0, 0, 0, 1, 1, 1]);
  const scene = morphScene(collapsed, false);
  // Three different cells arrive at the same output position.
  expect(morphPosition(collapsed, scene, 0, 1)).toEqual(
    morphPosition(collapsed, scene, 2, 1),
  );
  const compare = operation("gt", {
    interaction: "relation",
    relation: { rule: "elementwise", operand: 0, roles: ["input"] },
  });
  expect(
    relationMorph(
      compare,
      tensorRelation(compare, [input], input),
      input,
      input,
    ),
  ).toBeNull();
  expect(relationMorph(sum, null, input, totals)).toBeNull();
});
