import { ravel } from "./coordinates";
import { isBroken } from "./find";
import { coordinatesFor, type Plane } from "./plane";
import { pythonValue } from "./coordinates";

export type CellRange = {
  /** Flat indices, row by row across the rectangle. */
  cells: number[];
  rows: number;
  columns: number;
  /** First and last row, and first and last column, in the plane. */
  rowSpan: [number, number];
  columnSpan: [number, number];
};

/**
 * The rectangle of the plane on screen between two cells, like a
 * spreadsheet selection. Null when they sit on different slices or are the
 * same cell.
 */
export function rangeCells(
  shape: number[],
  plane: Plane,
  anchor: number,
  cursor: number,
): CellRange | null {
  const a = coordinatesFor(anchor, shape);
  const b = coordinatesFor(cursor, shape);
  const sameSlice = a.every(
    (coordinate, axis) =>
      axis === plane.row || axis === plane.column || coordinate === b[axis],
  );
  if (!sameSlice || anchor === cursor) return null;
  const span = (axis: number | null) =>
    axis === null
      ? [0, 0]
      : [Math.min(a[axis], b[axis]), Math.max(a[axis], b[axis])];
  const [r0, r1] = span(plane.row);
  const [c0, c1] = span(plane.column);
  const cells: number[] = [];
  for (let row = r0; row <= r1; row++)
    for (let column = c0; column <= c1; column++)
      cells.push(
        ravel(
          b.map((coordinate, axis) =>
            axis === plane.row
              ? row
              : axis === plane.column
                ? column
                : coordinate,
          ),
          shape,
        ),
      );
  return {
    cells,
    rows: r1 - r0 + 1,
    columns: c1 - c0 + 1,
    rowSpan: [r0, r1],
    columnSpan: [c0, c1],
  };
}

export type RangeStats = {
  count: number;
  sum: number;
  mean: number;
  /** Population standard deviation, as numpy's std. */
  std: number;
  min: number;
  max: number;
  /** NaN or infinite values, left out of the other figures. */
  broken: number;
};

/** Sum, mean and extremes of the finite values; null with none. */
export function rangeStats(
  values: (number | string | boolean | undefined)[],
): RangeStats | null {
  let count = 0,
    sum = 0,
    squares = 0,
    min = Infinity,
    max = -Infinity,
    broken = 0;
  for (const value of values) {
    if (value === undefined) continue;
    if (isBroken(value)) {
      broken += 1;
      continue;
    }
    const x = Number(value);
    if (!Number.isFinite(x)) continue;
    count += 1;
    sum += x;
    squares += x * x;
    min = Math.min(min, x);
    max = Math.max(max, x);
  }
  return count || broken
    ? {
        count,
        sum,
        mean: count ? sum / count : NaN,
        std: count
          ? Math.sqrt(Math.max(0, squares / count - (sum / count) ** 2))
          : NaN,
        min: count ? min : NaN,
        max: count ? max : NaN,
        broken,
      }
    : null;
}

/** The rectangle as tab-separated rows, which spreadsheets paste as cells. */
export function rangeTsv(
  values: (number | string | boolean | undefined)[],
  range: Pick<CellRange, "columns">,
) {
  const rows: string[] = [];
  for (let start = 0; start < values.length; start += range.columns)
    rows.push(
      values
        .slice(start, start + range.columns)
        .map((value) => (value === undefined ? "" : String(value)))
        .join("\t"),
    );
  return rows.join("\n");
}

/** The rectangle as a nested Python list, rows of columns. */
export function rangeList(
  values: (number | string | boolean | undefined)[],
  range: Pick<CellRange, "columns" | "rows">,
  dtype: string,
) {
  const rows: string[] = [];
  for (let start = 0; start < values.length; start += range.columns)
    rows.push(
      `[${values
        .slice(start, start + range.columns)
        .map((value) => pythonValue(value, dtype))
        .join(", ")}]`,
    );
  return range.rows === 1 ? rows[0] : `[${rows.join(", ")}]`;
}
