import { expect, test } from "vitest";
import { asNumber, binCells, planeMargins } from "./margins";

const cube = {
  shape: [2, 2, 3],
  numel: 12,
  values: [1, 2, 3, 4, 5, 6, 0.5, "nan", 1.5, -1, -2, -3] as (
    number | string
  )[],
};
const plane = { row: 1, column: 2 };

test("margins reduce each row and column of the plane through the cursor", () => {
  expect(planeMargins(cube, plane, [0, 0, 0], "sum")).toEqual({
    rows: [6, 15],
    columns: [5, 7, 9],
    all: 21,
  });
  expect(planeMargins(cube, plane, [0, 1, 2], "mean")?.rows).toEqual([2, 5]);
  expect(planeMargins(cube, plane, [0, 0, 0], "max")?.columns).toEqual([
    4, 5, 6,
  ]);
  // NaN spreads through its row and column, as in torch.
  const second = planeMargins(cube, plane, [1, 0, 0], "min")!;
  expect(second.rows).toEqual([NaN, -3]);
  expect(second.columns).toEqual([-1, NaN, -3]);
  expect(
    planeMargins({ ...cube, values: [] }, plane, [0, 0, 0], "sum"),
  ).toBeNull();
});

test("Python's words for infinity read as numbers", () => {
  expect([
    asNumber("inf"),
    asNumber("-inf"),
    asNumber("nan"),
    asNumber(true),
  ]).toEqual([Infinity, -Infinity, NaN, 1]);
});

test("a histogram bar finds the cells in its half-open bin", () => {
  const values = { numel: 6, values: [0, 0.25, 0.5, 0.75, 1, "nan"] };
  expect(binCells(values, 0, 1, 4, 1)).toEqual({
    from: 0.25,
    to: 0.5,
    cells: [1],
  });
  // The last bin keeps the maximum.
  expect(binCells(values, 0, 1, 4, 3)?.cells).toEqual([3, 4]);
  expect(binCells({ numel: 3, values: [1] }, 0, 1, 4, 0)).toBeNull();
});
