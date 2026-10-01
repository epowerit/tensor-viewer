import { describe, expect, it } from "vitest";
import { roundedCellPath } from "./cellOutline";

type Point = readonly [number, number];
function coordinates(path: string): number[] {
  return [...path.matchAll(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi)].map((match) =>
    Number(match[0]),
  );
}
function withinBounds(path: string, points: readonly Point[]) {
  const values = coordinates(path);
  expect(path).not.toMatch(/NaN|Infinity/);
  expect(values.every(Number.isFinite)).toBe(true);
  for (let axis = 0; axis < 2; axis++) {
    const minimum = Math.min(...points.map((point) => point[axis]));
    const maximum = Math.max(...points.map((point) => point[axis]));
    values
      .filter((_, index) => index % 2 === axis)
      .forEach((value) => {
        expect(value).toBeGreaterThanOrEqual(minimum);
        expect(value).toBeLessThanOrEqual(maximum);
      });
  }
}

describe("rounded projected cell outlines", () => {
  it("rounds a square with small quadratic corners and leaves its coordinates unchanged", () => {
    const square: readonly Point[] = Object.freeze([
      Object.freeze([0, 0] as const),
      Object.freeze([28, 0] as const),
      Object.freeze([28, 28] as const),
      Object.freeze([0, 28] as const),
    ]);
    const path = roundedCellPath(square);
    const values = coordinates(path);
    expect(values.slice(0, 8)).toEqual([1.6, 0, 26.4, 0, 28, 0, 28, 1.6]);
    expect(path.match(/ Q /g)).toHaveLength(4);
    expect(path.endsWith(" Z")).toBe(true);
    expect(square).toEqual([
      [0, 0],
      [28, 0],
      [28, 28],
      [0, 28],
    ]);
    withinBounds(path, square);
  });

  it("keeps skewed face curves inside the original convex polygon in either winding", () => {
    const quad: Point[] = [
      [2, 4],
      [26, 10],
      [21, 29],
      [-3, 23],
    ];
    for (const points of [quad, [...quad].reverse()]) {
      const path = roundedCellPath(points, 100);
      withinBounds(path, points);
      const values = coordinates(path);
      // After M, each corner contributes L incoming, Q vertex + outgoing.
      for (let index = 0; index < points.length; index++) {
        const vertexIndex = (index + 1) % points.length;
        const vertex = points[vertexIndex];
        const previous = points[index];
        const next = points[(vertexIndex + 1) % points.length];
        const [ix, iy, vx, vy, ox, oy] = values.slice(
          2 + index * 6,
          8 + index * 6,
        );
        expect([vx, vy]).toEqual(vertex);
        const previousLength = Math.hypot(previous[0] - vx, previous[1] - vy);
        const nextLength = Math.hypot(next[0] - vx, next[1] - vy);
        const limit = Math.min(previousLength, nextLength) * 0.12;
        expect(Math.hypot(ix - vx, iy - vy)).toBeLessThanOrEqual(limit + 1e-12);
        expect(Math.hypot(ox - vx, oy - vy)).toBeLessThanOrEqual(limit + 1e-12);
        for (const t of [0, 0.25, 0.5, 0.75, 1]) {
          const x = (1 - t) ** 2 * ix + 2 * t * (1 - t) * vx + t ** 2 * ox;
          const y = (1 - t) ** 2 * iy + 2 * t * (1 - t) * vy + t ** 2 * oy;
          const sides = points.map((a, edge) => {
            const b = points[(edge + 1) % points.length];
            return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
          });
          expect(
            sides.every((side) => side >= -1e-10) ||
              sides.every((side) => side <= 1e-10),
          ).toBe(true);
        }
      }
    }
  });

  it.each<readonly Point[]>([
    [
      [0, 0],
      [0, 0],
      [28, 28],
      [0, 28],
    ],
    [
      [4, 4],
      [4, 4],
      [4, 4],
      [4, 4],
    ],
    [
      [0, 0],
      [1e-12, 0],
      [1e-12, 28],
      [0, 28],
    ],
    [
      [0, 0],
      [28, 0],
      [56, 0],
    ],
  ])("keeps degenerate and nearly edge-on outlines finite: %j", (...points) => {
    const path = roundedCellPath(points);
    withinBounds(path, points);
    expect(path.match(/ Q /g)).toHaveLength(points.length);
  });

  it("handles large finite coordinate spans without arithmetic overflow", () => {
    const points: Point[] = [
      [-1e308, -1e308],
      [1e308, -1e308],
      [1e308, 1e308],
      [-1e308, 1e308],
    ];
    withinBounds(roundedCellPath(points), points);
  });

  it("supports triangles and zero-radius straight outlines", () => {
    const points: Point[] = [
      [0, 0],
      [28, 0],
      [14, 28],
    ];
    const path = roundedCellPath(points, 0);
    expect(path.match(/ Q /g)).toHaveLength(3);
    expect(
      coordinates(path).every((value) => [0, 14, 28].includes(value)),
    ).toBe(true);
    withinBounds(roundedCellPath(points), points);
  });

  it("returns safe paths for missing, incomplete, or invalid geometry", () => {
    expect(roundedCellPath([])).toBe("");
    expect(roundedCellPath([[4, 8]])).toBe("M 4 8 Z");
    expect(
      roundedCellPath([
        [4, 8],
        [7, 12],
      ]),
    ).toBe("M 4 8 L 7 12 Z");
    expect(
      roundedCellPath([
        [0, 0],
        [Infinity, 4],
        [4, 4],
      ]),
    ).toBe("");
    expect(
      roundedCellPath([
        [0, 0],
        [4, NaN],
        [4, 4],
      ]),
    ).toBe("");
  });
});
