import type { Run, Tensor } from "../api/client";
import { NEUTRAL_INK, inkOf, type Ink } from "./axisInk";
import { inTrace, lineageOf, rootLabel, type AxisStory } from "./axisLineage";
import { unravel } from "./coordinates";
import { inkShade } from "./inkShade";

/**
 * Cell paint: every cell is tinted by where its value sat along one input
 * axis (by default the first input's last axis), in that axis's ink. Lineage
 * finds that axis in every later tensor, even split into pieces or merged
 * with others, so a value keeps its color through transposes and reshapes.
 * A tensor that no longer has the axis is shaded along its own last traced
 * axis instead; one with no traced axis is not painted.
 */
export type CellPaint = (tensor: Tensor, index: number) => string | null;

const lastIndex = (
  sizes: number[],
  keep: (size: number, at: number) => boolean,
) => {
  for (let at = sizes.length - 1; at >= 0; at--)
    if (keep(sizes[at], at)) return at;
  return -1;
};

/** Digit `k` of `value` written in the mixed radix `sizes` (row-major). */
function digit(value: number, sizes: number[], k: number) {
  for (let j = sizes.length - 1; j > k; j--)
    value = Math.floor(value / sizes[j]);
  return value % sizes[k];
}

/** Where along `label` each cell of a tensor sits, or null if it cannot tell. */
export function sourcePosition(
  stories: AxisStory[],
  shape: number[],
  label: string,
): ((coords: number[]) => number) | null {
  type Found = { axis: number; sizes: number[]; k: number };
  let whole: Found | null = null;
  const pieces = new Map<number, Found & { of: number; parts: number[] }>();
  stories.forEach((story, axis) => {
    const sizes = story.terms.map((term) => term.size);
    // A sliced or padded axis no longer holds whole terms: positions are unknown.
    if (!sizes.length || sizes.reduce((a, b) => a * b, 1) !== shape[axis])
      return;
    story.terms.forEach((term, k) => {
      if (term.label !== label || term.role !== "input") return;
      if (!term.part) whole ??= { axis, sizes, k };
      else
        pieces.set(term.part.index, {
          axis,
          sizes,
          k,
          of: term.part.of,
          parts: term.part.sizes,
        });
    });
  });
  if (whole) {
    const { axis, sizes, k } = whole;
    return (coords) => digit(coords[axis], sizes, k);
  }
  const first = pieces.get(0);
  if (!first || pieces.size !== first.of) return null;
  const order = [...Array(first.of).keys()].map((i) => pieces.get(i)!);
  return (coords) =>
    order.reduce(
      (position, found, i) =>
        position * first.parts[i] +
        digit(coords[found.axis], found.sizes, found.k),
      0,
    );
}

/** The paint for the tensors of one trace; other tensors get none. */
export function cellPaintOf(trace: Run["trace"]): CellPaint {
  const lineage = lineageOf(trace);
  const ink = inkOf(trace);
  const input = trace.tensors[trace.input_ids?.[0] ?? ""];
  const axis = input ? lastIndex(input.shape, (size) => size > 1) : -1;
  const sourceInk: Ink | null =
    input && axis >= 0 ? (ink(input)?.[axis] ?? null) : null;
  const label = input && axis >= 0 ? rootLabel(input, axis) : null;
  const size = input && axis >= 0 ? input.shape[axis] : 0;
  type Plan = ((coords: number[]) => string) | null;
  const plans = new Map<string, Plan>();
  const plan = (tensor: Tensor): Plan => {
    const position =
      sourceInk && label
        ? sourcePosition(lineage(tensor.id), tensor.shape, label)
        : null;
    if (position)
      return (coords) => inkShade(sourceInk!, position(coords), size).fill;
    // Fall back to the tensor's own last single-source axis.
    const own = ink(tensor);
    const fallback = lastIndex(
      tensor.shape,
      (extent, at) =>
        extent > 1 &&
        own?.[at]?.colors.length === 1 &&
        own[at]!.colors[0] !== NEUTRAL_INK,
    );
    if (fallback < 0) return null;
    const axisInk = own![fallback]!;
    return (coords) =>
      inkShade(axisInk, coords[fallback], tensor.shape[fallback]).fill;
  };
  return (tensor, index) => {
    if (!inTrace(trace, tensor)) return null;
    if (!plans.has(tensor.id)) plans.set(tensor.id, plan(tensor));
    const paint = plans.get(tensor.id);
    return paint ? paint(unravel(index, tensor.shape)) : null;
  };
}

/** Paint for tensors from any of several traces (a run and a shape check). */
export function cellPaintOfAll(
  traces: (Run["trace"] | null | undefined)[],
): CellPaint {
  const painters = traces.flatMap((trace) =>
    trace ? [cellPaintOf(trace)] : [],
  );
  return (tensor, index) => {
    for (const paint of painters) {
      const found = paint(tensor, index);
      if (found) return found;
    }
    return null;
  };
}
