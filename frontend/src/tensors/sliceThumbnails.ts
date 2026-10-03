import type { Tensor } from "../api/client";
import { planeValues } from "./find";
import { asNumber } from "./margins";
import type { Plane } from "./plane";

export type Thumbnails = {
  rows: number;
  columns: number;
  /** How many slices the axis has; at most `SLICES` are pictured. */
  total: number;
  /** One picture per slice, row by row. */
  thumbs: number[][];
};

/** Slices beyond this many are not pictured; the backend uses the same. */
export const SLICES = 256;

function downsample(
  values: number[],
  rows: number,
  columns: number,
  size: number,
) {
  const split = (length: number) => {
    const parts = Math.min(size, length);
    // Like numpy.array_split: the first `length % parts` parts take one more.
    const base = Math.floor(length / parts),
      extra = length % parts;
    const bounds: [number, number][] = [];
    let start = 0;
    for (let part = 0; part < parts; part++) {
      const end = start + base + (part < extra ? 1 : 0);
      bounds.push([start, end]);
      start = end;
    }
    return bounds;
  };
  const rowParts = split(rows),
    columnParts = split(columns);
  const picture: number[] = [];
  for (const [r0, r1] of rowParts)
    for (const [c0, c1] of columnParts) {
      let sum = 0;
      for (let row = r0; row < r1; row++)
        for (let column = c0; column < c1; column++)
          sum += values[row * columns + column];
      picture.push(sum / ((r1 - r0) * (c1 - c0)));
    }
  return { picture, rows: rowParts.length, columns: columnParts.length };
}

/**
 * A small averaged picture of the plane at each index of a hidden `axis`,
 * the other hidden axes held at `coords`. Null unless every value is inline.
 */
export function sliceThumbnails(
  tensor: Pick<Tensor, "shape" | "values" | "numel" | "dtype">,
  plane: Plane,
  coords: number[],
  axis: number,
  size = 8,
): Thumbnails | null {
  if (tensor.values.length !== tensor.numel || !tensor.numel) return null;
  const count = Math.min(tensor.shape[axis], SLICES);
  let rows = 0,
    columns = 0;
  const thumbs: number[][] = [];
  for (let index = 0; index < count; index++) {
    const at = coords.map((coordinate, i) => (i === axis ? index : coordinate));
    const slice = planeValues(tensor, plane, at);
    if (!slice) return null;
    const [height, width] =
      slice.shape.length === 2 ? slice.shape : [1, slice.shape[0]];
    const small = downsample(slice.values.map(asNumber), height, width, size);
    rows = small.rows;
    columns = small.columns;
    thumbs.push(small.picture);
  }
  return { rows, columns, total: tensor.shape[axis], thumbs };
}

/**
 * A small averaged picture of a tensor's first plane: its last two axes, with
 * any leading axes at 0. A vector is one row. Null unless every value is
 * inline. A cell averaging a NaN or infinity is NaN.
 */
export function planeThumbnail(
  tensor: Pick<Tensor, "shape" | "values" | "numel" | "dtype">,
  size = 6,
): { rows: number; columns: number; picture: number[] } | null {
  if (tensor.values.length !== tensor.numel || !tensor.numel) return null;
  const rank = tensor.shape.length;
  const plane = {
    row: rank > 1 ? rank - 2 : null,
    column: rank ? rank - 1 : null,
  };
  const slice = planeValues(
    tensor,
    plane,
    tensor.shape.map(() => 0),
  );
  if (!slice) return null;
  const [height, width] =
    slice.shape.length === 2 ? slice.shape : [1, slice.shape[0] ?? 1];
  const values = slice.values.map((value) => {
    const number = asNumber(value);
    return Number.isFinite(number) ? number : NaN;
  });
  return downsample(values, height, width, size);
}
