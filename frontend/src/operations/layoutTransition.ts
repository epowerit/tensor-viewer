import type { Operation, Tensor } from "../api/client";
import { product, ravel, unravel } from "../tensors/coordinates";
import { sourceIndex } from "../tensors/relationships";
import { storagePosition } from "../tensors/plane";

export type LayoutTransition = {
  input: Tensor;
  output: Tensor;
  source: number;
  target: number;
  before: number[];
  after: number[];
  rule: "identity" | "permutation";
  order: number[] | null;
  sharedStorage: boolean;
  sourceStorage: number;
  targetStorage: number;
};

const regroup = new Set([
  "reshape",
  "view",
  "flatten",
  "squeeze",
  "unsqueeze",
  "contiguous",
  "clone",
]);
const reorder = new Set(["permute", "transpose", "t"]);

/** Animate only verified, value-preserving bijections. Unfold needs a one-to-many lesson. */
export function layoutTransition(
  op: Operation,
  input: Tensor | undefined,
  output: Tensor | undefined,
  target: number,
): LayoutTransition | null {
  if (
    !input ||
    !output ||
    op.status !== "ok" ||
    op.lesson.interaction !== "mapping" ||
    input.dtype !== output.dtype ||
    !input.numel ||
    input.numel !== output.numel ||
    !Number.isSafeInteger(target) ||
    target < 0 ||
    target >= output.numel
  )
    return null;
  const order = op.lesson.axis_order;
  const rule =
    op.lesson.mapping_rule ??
    (order
      ? "permutation"
      : op.lesson.mapping && regroup.has(op.kind)
        ? "identity"
        : null);
  if (rule === "identity") {
    if (!regroup.has(op.kind)) return null;
  } else if (rule === "permutation") {
    if (
      !reorder.has(op.kind) ||
      !order ||
      order.length !== input.shape.length ||
      output.shape.length !== order.length ||
      new Set(order).size !== order.length ||
      order.some(
        (axis, i) =>
          !Number.isInteger(axis) ||
          axis < 0 ||
          axis >= order.length ||
          output.shape[i] !== input.shape[axis],
      )
    )
      return null;
  } else return null;
  if (
    product(input.shape) !== input.numel ||
    product(output.shape) !== output.numel
  )
    return null;
  const source = sourceIndex(op, input, output, target);
  if (
    source === undefined ||
    !Number.isSafeInteger(source) ||
    source < 0 ||
    source >= input.numel
  )
    return null;
  const before = unravel(source, input.shape),
    after = unravel(target, output.shape);
  if (
    rule === "identity"
      ? source !== target
      : order!.some((axis, i) => before[axis] !== after[i])
  )
    return null;
  return {
    input,
    output,
    source,
    target,
    before,
    after,
    rule,
    order: order ?? null,
    sharedStorage: input.storage_id === output.storage_id,
    sourceStorage: storagePosition(before, input.strides, input.storage_offset),
    targetStorage: storagePosition(
      after,
      output.strides,
      output.storage_offset,
    ),
  };
}

/** A focused last-two-axis slice: at most 32 cells, independent of tensor size. */
export function transitionWindow(tensor: Tensor, selected: number) {
  const coordinates = unravel(selected, tensor.shape);
  const rows = tensor.shape.at(-2) ?? 1,
    columns = tensor.shape.at(-1) ?? 1;
  const row = coordinates.at(-2) ?? 0,
    column = coordinates.at(-1) ?? 0;
  const startRow = Math.floor(row / 4) * 4,
    startColumn = Math.floor(column / 8) * 8;
  const visibleRows = Math.min(4, rows - startRow),
    visibleColumns = Math.min(8, columns - startColumn);
  const prefix = coordinates.slice(0, -2);
  const cells = Array.from({ length: visibleRows * visibleColumns }, (_, i) => {
    const r = startRow + Math.floor(i / visibleColumns),
      c = startColumn + (i % visibleColumns);
    const coords =
      tensor.shape.length > 1
        ? [...prefix, r, c]
        : tensor.shape.length
          ? [c]
          : [];
    return {
      index: ravel(coords, tensor.shape),
      coordinates: coords,
      row: r - startRow,
      column: c - startColumn,
    };
  });
  return {
    cells,
    prefix,
    startRow,
    startColumn,
    rows: visibleRows,
    columns: visibleColumns,
    selectedRow: row - startRow,
    selectedColumn: column - startColumn,
  };
}

export type Point = { x: number; y: number };
export const CELL_SIZE = 28;
export function transitionScene(mapping: LayoutTransition, vertical: boolean) {
  const before = transitionWindow(mapping.input, mapping.source),
    after = transitionWindow(mapping.output, mapping.target);
  const width = vertical ? 304 : 728,
    height = vertical ? 560 : 260;
  const origin = (window: typeof before, side: number) => ({
    x: (vertical ? 152 : side ? 576 : 152) - (window.columns * CELL_SIZE) / 2,
    y: (vertical ? (side ? 422 : 108) : 130) - (window.rows * CELL_SIZE) / 2,
  });
  const inputOrigin = origin(before, 0),
    outputOrigin = origin(after, 1);
  const center = (window: typeof before, origin: Point) => ({
    x: origin.x + (window.selectedColumn + 0.5) * CELL_SIZE,
    y: origin.y + (window.selectedRow + 0.5) * CELL_SIZE,
  });
  return {
    width,
    height,
    before,
    after,
    inputOrigin,
    outputOrigin,
    start: center(before, inputOrigin),
    end: center(after, outputOrigin),
    middle: { x: width / 2, y: vertical ? 276 : 130 },
  };
}

/** Two smooth legs with exact source, rule, and destination stops. */
export function transitionPoint(
  start: Point,
  middle: Point,
  end: Point,
  progress: number,
): Point {
  const p = Math.max(0, Math.min(1, progress));
  const a = p < 0.5 ? start : middle,
    b = p < 0.5 ? middle : end;
  const t = p < 0.5 ? p * 2 : (p - 0.5) * 2;
  const eased = t * t * (3 - 2 * t);
  return { x: a.x + (b.x - a.x) * eased, y: a.y + (b.y - a.y) * eased };
}

export function transitionStep(progress: number, direction: -1 | 1) {
  const stops = [0, 0.5, 1];
  return direction === 1
    ? (stops.find((stop) => stop > progress + 0.0001) ?? 1)
    : ([...stops].reverse().find((stop) => stop < progress - 0.0001) ?? 0);
}
