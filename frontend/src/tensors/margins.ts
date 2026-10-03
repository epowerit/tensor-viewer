import type { Tensor } from "../api/client";
import { ravel } from "./coordinates";
import type { Plane } from "./plane";

export type Reduce = "sum" | "mean" | "max" | "min";

/** A recorded value as a number: Python's nan, inf and -inf included. */
export function asNumber(value: number | string | boolean | undefined) {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return Number(value);
  if (value === undefined) return NaN;
  const word = value.trim().toLowerCase();
  if (word === "inf" || word === "infinity") return Infinity;
  if (word === "-inf" || word === "-infinity") return -Infinity;
  return Number(value);
}

function reduce(values: number[], how: Reduce) {
  if (!values.length) return NaN;
  // NaN spreads, as it does in torch.sum, mean, max and min.
  if (values.some(Number.isNaN)) return NaN;
  if (how === "max") return Math.max(...values);
  if (how === "min") return Math.min(...values);
  const sum = values.reduce((total, value) => total + value, 0);
  return how === "sum" ? sum : sum / values.length;
}

export type Margins = {
  /** One per row of the plane: the reduction across its columns. */
  rows: number[];
  /** One per column of the plane: the reduction down its rows. */
  columns: number[];
  /** The reduction over the whole plane. */
  all: number;
};

/** Larger planes are not reduced in the browser. */
export const MARGIN_LIMIT = 1_000_000;

/**
 * Row and column reductions of the plane through `coords`, over the plane's
 * full axes, not only the window on screen. Null unless every value was
 * recorded inline.
 */
export function planeMargins(
  tensor: Pick<Tensor, "shape" | "values" | "numel">,
  plane: Plane,
  coords: number[],
  how: Reduce,
): Margins | null {
  if (tensor.values.length !== tensor.numel || !tensor.numel) return null;
  const rows = plane.row === null ? 1 : tensor.shape[plane.row];
  const columns = plane.column === null ? 1 : tensor.shape[plane.column];
  if (rows * columns > MARGIN_LIMIT) return null;
  const grid: number[][] = [];
  for (let row = 0; row < rows; row++) {
    const line: number[] = [];
    for (let column = 0; column < columns; column++)
      line.push(
        asNumber(
          tensor.values[
            ravel(
              coords.map((coordinate, axis) =>
                axis === plane.row
                  ? row
                  : axis === plane.column
                    ? column
                    : coordinate,
              ),
              tensor.shape,
            )
          ],
        ),
      );
    grid.push(line);
  }
  return {
    rows: grid.map((line) => reduce(line, how)),
    columns: Array.from({ length: columns }, (_, column) =>
      reduce(
        grid.map((line) => line[column]),
        how,
      ),
    ),
    all: reduce(grid.flat(), how),
  };
}

/** The cells of `counts` bin `bin` of a histogram from `low` to `high`. */
export function binCells(
  tensor: Pick<Tensor, "values" | "numel">,
  low: number,
  high: number,
  bins: number,
  bin: number,
) {
  if (tensor.values.length !== tensor.numel) return null;
  const width = (high - low) / bins;
  const from = low + bin * width;
  const to = bin === bins - 1 ? high : low + (bin + 1) * width;
  const found: number[] = [];
  tensor.values.forEach((raw, index) => {
    const value = asNumber(raw);
    if (!Number.isFinite(value)) return;
    // Bins are half-open, except the last, which keeps the maximum.
    if (value >= from && (value < to || (bin === bins - 1 && value <= to)))
      found.push(index);
  });
  return { from, to, cells: found };
}
