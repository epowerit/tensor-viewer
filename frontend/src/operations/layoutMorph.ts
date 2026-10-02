import type { Operation, Tensor } from "../api/client";
import { product, unravel } from "../tensors/coordinates";
import { sourceIndex } from "../tensors/relationships";
import { reductionTarget, relationSource, type Relation } from "./relations";
import { axisPhrase } from "../tensors/axisPhrase";

/** Every cell is drawn, so whole-tensor motion is limited to small tensors. */
export const MORPH_LIMIT = 256;
const MAX_SPAN = 40;
export const MORPH_CELL = 26;

export type Point = { x: number; y: number };
export type CellLayout = {
  points: Point[];
  width: number;
  height: number;
  /** Cells across and down, ignoring the gaps between blocks. */
  columns: number;
  rows: number;
};

/**
 * Row-major cells on a plane. The last axis runs across, the one before it
 * runs down, and earlier axes alternate as separated blocks.
 */
export function cellLayout(shape: number[], cell = MORPH_CELL): CellLayout {
  const rank = shape.length;
  const step: number[] = Array(rank).fill(0);
  let width = cell,
    height = cell,
    columns = 1,
    rows = 1;
  for (let axis = rank - 1; axis >= 0; axis--) {
    const level = rank - 1 - axis;
    const across = level % 2 === 0;
    const gap = level < 2 ? 0 : cell * 0.45 * Math.floor(level / 2);
    const size = shape[axis];
    if (across) {
      step[axis] = width + gap;
      width = size * width + (size - 1) * gap;
      columns *= size;
    } else {
      step[axis] = height + gap;
      height = size * height + (size - 1) * gap;
      rows *= size;
    }
  }
  const count = shape.reduce((a, b) => a * b, 1);
  const points = Array.from({ length: count }, (_, index) => {
    const coordinates = unravel(index, shape);
    let x = 0,
      y = 0;
    coordinates.forEach((coordinate, axis) => {
      if ((rank - 1 - axis) % 2 === 0) x += coordinate * step[axis];
      else y += coordinate * step[axis];
    });
    return { x, y };
  });
  return { points, width, height, columns, rows };
}

export type Mover = { source: number; target: number };
export type LayoutMorph = {
  input: Tensor;
  output: Tensor;
  /**
   * rearrange: every cell moves once. select: output cells are copies and the
   * input stays in place. collapse: groups of input cells merge into one.
   */
  kind: "rearrange" | "select" | "collapse";
  movers: Mover[];
  /** Input flat index for every output flat index (not for collapse). */
  sources: number[];
  caption: string;
  before: CellLayout;
  after: CellLayout;
};

const moving = new Set([
  "reshape",
  "view",
  "flatten",
  "squeeze",
  "unsqueeze",
  "contiguous",
  "clone",
  "permute",
  "transpose",
  "t",
  "roll",
  "ravel",
  "view_as",
  "reshape_as",
  "unflatten",
  "swapaxes",
  "swapdims",
  "movedim",
  "moveaxis",
  "T",
  "mT",
]);

function drawable(input: Tensor, output: Tensor) {
  if (
    !input.numel ||
    !output.numel ||
    input.numel > MORPH_LIMIT ||
    output.numel > MORPH_LIMIT ||
    product(input.shape) !== input.numel ||
    product(output.shape) !== output.numel
  )
    return null;
  const before = cellLayout(input.shape),
    after = cellLayout(output.shape);
  return Math.max(before.columns, after.columns, before.rows, after.rows) >
    MAX_SPAN
    ? null
    : { before, after };
}

/** Animate a whole tensor only when every cell has exactly one recorded origin. */
export function layoutMorph(
  op: Operation,
  input: Tensor | undefined,
  output: Tensor | undefined,
): LayoutMorph | null {
  if (
    !input ||
    !output ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.lesson.interaction !== "mapping" ||
    !moving.has(op.kind) ||
    input.dtype !== output.dtype ||
    input.numel !== output.numel
  )
    return null;
  const layouts = drawable(input, output);
  if (!layouts) return null;
  const rule =
    op.lesson.mapping_rule ?? (op.lesson.axis_order ? "permutation" : null);
  if (rule !== "identity" && rule !== "permutation" && rule !== "roll")
    return null;
  const sources: number[] = [];
  const seen = new Set<number>();
  for (let target = 0; target < output.numel; target++) {
    const source = sourceIndex(op, input, output, target);
    if (
      source === undefined ||
      !Number.isInteger(source) ||
      source < 0 ||
      source >= input.numel ||
      seen.has(source)
    )
      return null;
    seen.add(source);
    sources.push(source);
  }
  return {
    input,
    output,
    kind: "rearrange",
    movers: sources.map((source, target) => ({ source, target })),
    sources,
    caption:
      rule === "permutation"
        ? `Output axes read input axes in order [${(op.lesson.axis_order ?? []).join(", ")}].`
        : rule === "roll"
          ? "Cells shift along the rolled axes and wrap around the edge."
          : "Reading order is unchanged; only the dimension boundaries move.",
    ...layouts,
  };
}

/** Motion for selection, repetition, lookup, and reduction rules. */
export function relationMorph(
  op: Operation,
  relation: Relation | null,
  input: Tensor | undefined,
  output: Tensor | undefined,
): LayoutMorph | null {
  if (!relation || !input || !output || op.mutations?.length) return null;
  if (relation.rule === "elementwise") return null;
  const layouts = drawable(input, output);
  if (!layouts) return null;
  if (relation.rule === "reduce") {
    const axes = axisPhrase(relation.axes, input.axes);
    return {
      input,
      output,
      kind: "collapse",
      movers: Array.from({ length: input.numel }, (_, source) => ({
        source,
        target: reductionTarget(relation, input, output, source),
      })),
      sources: [],
      caption: `Cells that differ only along ${axes} meet in one output cell.`,
      ...layouts,
    };
  }
  const sources: number[] = [];
  for (let target = 0; target < output.numel; target++) {
    const source = relationSource(relation, input, output, target);
    if (
      source === undefined ||
      !Number.isInteger(source) ||
      source < 0 ||
      source >= input.numel
    )
      return null;
    sources.push(source);
  }
  return {
    input,
    output,
    kind: "select",
    movers: sources.map((source, target) => ({ source, target })),
    sources,
    caption:
      relation.rule === "index"
        ? "Selected cells keep their order; everything else is left behind."
        : relation.rule === "tile"
          ? "Each input cell is read again for every repetition."
          : "Each output cell copies the input cell its index names.",
    ...layouts,
  };
}

export type MorphScene = {
  width: number;
  height: number;
  vertical: boolean;
  inputOrigin: Point;
  outputOrigin: Point;
};

const HEADER = 34;
const MARGIN = 14;
/** Place both arrangements side by side, or stacked when they are wide. */
export function morphScene(morph: LayoutMorph, narrow: boolean): MorphScene {
  const { before, after } = morph;
  const gap = MORPH_CELL * 3;
  const vertical = narrow || before.width + after.width + gap > 900;
  if (vertical) {
    const width = Math.max(before.width, after.width) + MARGIN * 2;
    return {
      vertical,
      width,
      height: before.height + after.height + HEADER * 2 + gap * 0.6 + MARGIN,
      inputOrigin: { x: (width - before.width) / 2, y: HEADER },
      outputOrigin: {
        x: (width - after.width) / 2,
        y: HEADER * 2 + before.height + gap * 0.6,
      },
    };
  }
  const height = Math.max(before.height, after.height);
  return {
    vertical,
    width: before.width + after.width + gap + MARGIN * 2,
    height: height + HEADER + MARGIN,
    inputOrigin: { x: MARGIN, y: HEADER + (height - before.height) / 2 },
    outputOrigin: {
      x: MARGIN + before.width + gap,
      y: HEADER + (height - after.height) / 2,
    },
  };
}

/** Cells leave one after another, so the reading order stays visible. */
const STAGGER = 0.4;
export function morphPosition(
  morph: LayoutMorph,
  scene: MorphScene,
  mover: number,
  progress: number,
): Point {
  const count = morph.movers.length;
  const delay = count > 1 ? (mover / (count - 1)) * STAGGER : 0;
  const t = Math.max(0, Math.min(1, (progress - delay) / (1 - STAGGER)));
  const eased = t * t * (3 - 2 * t);
  const { source, target } = morph.movers[mover];
  const from = morph.before.points[source],
    to = morph.after.points[target];
  const a = {
      x: scene.inputOrigin.x + from.x,
      y: scene.inputOrigin.y + from.y,
    },
    b = { x: scene.outputOrigin.x + to.x, y: scene.outputOrigin.y + to.y };
  return { x: a.x + (b.x - a.x) * eased, y: a.y + (b.y - a.y) * eased };
}

/** A stable color per element, ordered by its position in the input. */
export function morphColor(source: number, count: number) {
  const t = count > 1 ? source / (count - 1) : 0;
  return `hsl(${Math.round(265 - 110 * t)} 62% ${Math.round(46 + 14 * t)}%)`;
}
