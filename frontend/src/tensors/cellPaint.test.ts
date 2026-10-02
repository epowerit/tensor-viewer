import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { INK_PALETTE } from "./axisInk";
import { cellPaintOf, cellPaintOfAll, sourcePosition } from "./cellPaint";
import { ravel } from "./coordinates";
import recorded from "./fixtures/lineage-trace.json";

// x [2, 3, 8]: batch, tokens, features. Cells are painted by their feature.
const trace = recorded as unknown as Run["trace"];
const byName = (name: string) =>
  Object.values(trace.tensors).find((tensor) => tensor.name === name)!;
const paint = cellPaintOf(trace);
const at = (name: string, coords: number[]) =>
  paint(byName(name), ravel(coords, byName(name).shape));

test("a value keeps its color wherever it moves", () => {
  // Feature 5 of x is head 1, position 1 of heads [batch, head, token, dim].
  expect(at("heads", [0, 1, 2, 1])).toBe(at("x", [0, 2, 5]));
  expect(at("merged", [1, 2, 5])).toBe(at("x", [0, 2, 5]));
  // Different features differ; the same feature in another token does not.
  expect(at("x", [0, 0, 5])).toBe(at("x", [1, 2, 5]));
  expect(at("x", [0, 0, 5])).not.toBe(at("x", [0, 0, 6]));
  // Features are inked violet: the third input axis.
  expect(at("x", [0, 0, 0])).toMatch(/^hsl\(25\d /);
});

test("pieces recombine into the original position", () => {
  const stories = [
    {
      size: 2,
      terms: [
        {
          label: "x.f",
          size: 2,
          role: "input" as const,
          part: { index: 0, of: 2, sizes: [2, 4] },
        },
      ],
      note: null,
    },
    {
      size: 4,
      terms: [
        {
          label: "x.f",
          size: 4,
          role: "input" as const,
          part: { index: 1, of: 2, sizes: [2, 4] },
        },
      ],
      note: null,
    },
  ];
  expect(sourcePosition(stories, [2, 4], "x.f")!([1, 3])).toBe(7);
  // A merged axis: features are the low digits of tokens × features.
  const merged = [
    {
      size: 24,
      terms: [
        { label: "x.t", size: 3, role: "input" as const },
        { label: "x.f", size: 8, role: "input" as const },
      ],
      note: null,
    },
  ];
  expect(sourcePosition(merged, [24], "x.f")!([13])).toBe(5);
  // Missing pieces or sliced axes cannot tell a position.
  expect(sourcePosition([stories[0]], [2], "x.f")).toBeNull();
});

test("without the source axis, a tensor shades along its own traced axis", () => {
  // proj = x @ w: features are gone; tokens (amber) shade the cells.
  expect(at("proj", [0, 1, 0])).toMatch(/^hsl\(27 /);
  expect(INK_PALETTE[1]).toBe("#f2b27c");
  // Tensors from another trace are never painted.
  expect(paint({ ...byName("x"), storage_id: "elsewhere" }, 0)).toBeNull();
  expect(paint({ ...byName("x"), axes: ["b", "t", "f"] }, 5)).toBe(
    at("x", [0, 0, 5]),
  );
  expect(cellPaintOfAll([null, trace])(byName("x"), 0)).not.toBeNull();
});
