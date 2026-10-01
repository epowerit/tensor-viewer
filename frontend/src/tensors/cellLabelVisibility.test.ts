import { describe, expect, it } from "vitest";
import { visibleCellLabels } from "./cellLabelVisibility";

type Point = readonly [number, number];
const square = (x: number, y: number, width = 1): Point[] => [
  [x, y],
  [x + width, y],
  [x + width, y + width],
  [x, y + width],
];
const cell = (
  index: number,
  x = 0,
  y = 0,
  labelPoint: Point = [x + 0.5, y + 0.5],
) => ({
  index,
  faces: [square(x, y)],
  labelPoint,
});

describe("glass cell label visibility", () => {
  it("keeps only the nearer value when front and back cells coincide", () => {
    expect(visibleCellLabels([cell(8), cell(19)])).toEqual(new Set([19]));
    expect(visibleCellLabels([cell(19), cell(8)])).toEqual(new Set([8]));
  });

  it("keeps separated cells and values visible through real gutters", () => {
    expect(visibleCellLabels([cell(1, 0), cell(2, 1.04)])).toEqual(
      new Set([1, 2]),
    );
    expect(
      visibleCellLabels([
        cell(0, 0, 0, [1.02, 0.5]),
        cell(1, 0),
        cell(2, 1.04),
      ]),
    ).toEqual(new Set([0, 1, 2]));
  });

  it("tests the value center rather than rejecting every partially overlapped cell", () => {
    expect(visibleCellLabels([cell(1), cell(2, 0.6)])).toEqual(new Set([1, 2]));
    expect(visibleCellLabels([cell(1), cell(2, 0.4)])).toEqual(new Set([2]));
  });

  it("never lets a cell's own faces cover its label", () => {
    expect(
      visibleCellLabels([
        {
          ...cell(4),
          faces: [square(0, 0), square(0.2, 0.2)],
        },
      ]),
    ).toEqual(new Set([4]));
  });

  it.each<Point>([
    [0, 0.5],
    [1, 0.5],
    [0.5, 0],
    [0.5, 1],
    [1, 1],
  ])("treats a center exactly on a nearer boundary as covered: %j", (x, y) => {
    expect(visibleCellLabels([cell(1, 0, 0, [x, y]), cell(2)])).toEqual(
      new Set([2]),
    );
  });

  it("checks projected faces rather than just their bounding boxes in either winding", () => {
    const diamond: Point[] = [
      [0, 1],
      [1, 0],
      [2, 1],
      [1, 2],
    ];
    for (const points of [diamond, [...diamond].reverse()]) {
      const front = { index: 4, faces: [points], labelPoint: [1, 1] as Point };
      expect(visibleCellLabels([cell(1, 0, 0, [0.1, 0.1]), front])).toEqual(
        new Set([1, 4]),
      );
      expect(visibleCellLabels([cell(1, 0, 0, [0.8, 0.8]), front])).toEqual(
        new Set([4]),
      );
    }
  });

  it("can cover a label with any face of a nearer cell", () => {
    const front = {
      index: 2,
      faces: [square(8, 8), square(0, 0)],
      labelPoint: [8.5, 8.5] as Point,
    };
    expect(visibleCellLabels([cell(1), front])).toEqual(new Set([2]));
  });

  it("handles an empty scene and the 1024-cell view budget without changing input", () => {
    expect(visibleCellLabels([])).toEqual(new Set());
    const cells = Array.from({ length: 1024 }, (_, i) =>
      cell(i, (i % 32) * 1.04, Math.floor(i / 32) * 1.04),
    );
    const before = structuredClone(cells);
    expect(visibleCellLabels(cells)).toEqual(
      new Set(cells.map((cell) => cell.index)),
    );
    expect(cells).toEqual(before);
  });
});
