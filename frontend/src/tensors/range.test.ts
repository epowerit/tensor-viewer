import { expect, test } from "vitest";
import { ravel } from "./coordinates";
import { rangeCells, rangeList, rangeStats, rangeTsv } from "./range";

const shape = [2, 3, 4];
const plane = { row: 1, column: 2 };

test("a selection is the rectangle between two cells of one slice", () => {
  const range = rangeCells(
    shape,
    plane,
    ravel([1, 2, 3], shape),
    ravel([1, 0, 2], shape),
  );
  expect(range).toEqual({
    cells: [14, 15, 18, 19, 22, 23],
    rows: 3,
    columns: 2,
    rowSpan: [0, 2],
    columnSpan: [2, 3],
  });
  // Different slices, or one cell, make no selection.
  expect(
    rangeCells(shape, plane, ravel([0, 0, 0], shape), ravel([1, 1, 1], shape)),
  ).toBeNull();
  expect(rangeCells(shape, plane, 5, 5)).toBeNull();
});

test("selection totals leave NaN and infinity out, and count them", () => {
  expect(rangeStats([1, "nan", -3, 4, "-inf", undefined])).toEqual({
    count: 3,
    sum: 2,
    mean: 2 / 3,
    std: Math.sqrt((1 + 9 + 16) / 3 - (2 / 3) ** 2),
    min: -3,
    max: 4,
    broken: 2,
  });
  expect(rangeStats([undefined])).toBeNull();
});

test("a selection copies as spreadsheet rows or a nested Python list", () => {
  const values = [1, 2.5, "nan", 4];
  expect(rangeTsv(values, { columns: 2 })).toBe("1\t2.5\nnan\t4");
  expect(rangeList(values, { rows: 2, columns: 2 }, "float32")).toBe(
    '[[1, 2.5], [float("nan"), 4]]',
  );
  expect(rangeList([1, 2], { rows: 1, columns: 2 }, "int64")).toBe("[1, 2]");
});
