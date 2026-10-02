import type { Run, Tensor } from "../api/client";
import {
  describeAxis,
  inTrace,
  lineageOf,
  rootLabel,
  type AxisStory,
} from "./axisLineage";

/**
 * Axis ink: every axis of a model input gets one color, and the color follows
 * that axis through permutes, reshapes, products, and reductions, so a learner
 * can watch where `batch`, `tokens`, or `features` went without reading numbers.
 * Pieces of a split axis keep its color; a merged axis carries all of its
 * sources' colors. Axes of weights or tensors made inside the code share one
 * neutral ink, and an axis without a traced story has none.
 */
export type Ink = {
  /** One color per source, in order: two or more for a merged axis. */
  colors: string[];
  /** For a piece of a split axis: "1/2". */
  piece: string | null;
  /** Readable origin, for titles and screen readers. */
  text: string;
};

// Distinct on the dark studio background, and distinct from each other.
export const INK_PALETTE = [
  "#6fd3bf", // teal
  "#f2b27c", // amber
  "#b69cf8", // violet
  "#85b6f6", // blue
  "#f28fb1", // pink
  "#c3dd80", // lime
  "#e9d37b", // yellow
  "#80d6ee", // cyan
];
export const NEUTRAL_INK = "#9a90ad";

/** Ink lookups for the tensors of one trace; other tensors get none. */
export function inkOf(
  trace: Run["trace"],
): (tensor: Tensor) => (Ink | null)[] | null {
  const lineage = lineageOf(trace);
  const colors = new Map<string, string>();
  for (const id of trace.input_ids ?? []) {
    const tensor = trace.tensors[id];
    tensor?.shape.forEach((_, axis) => {
      const label = rootLabel(tensor, axis);
      if (!colors.has(label))
        colors.set(label, INK_PALETTE[colors.size % INK_PALETTE.length]);
    });
  }
  const color = (label: string, role: string | undefined) =>
    (role === "input" && colors.get(label)) || NEUTRAL_INK;
  const ink = (story: AxisStory): Ink | null =>
    story.terms.length
      ? {
          colors: story.terms.map((term) => color(term.label, term.role)),
          piece:
            story.terms.length === 1 && story.terms[0].part
              ? `${story.terms[0].part.index + 1}/${story.terms[0].part.of}`
              : null,
          text: describeAxis(story),
        }
      : null;
  const memo = new Map<string, (Ink | null)[]>();
  return (tensor) => {
    // Ids repeat across traces: only this trace's own tensor object counts.
    if (!inTrace(trace, tensor)) return null;
    let found = memo.get(tensor.id);
    if (!found) {
      found = lineage(tensor.id).map(ink);
      memo.set(tensor.id, found);
    }
    return found;
  };
}

/** Ink for tensors from any of several traces (a run and a shape check). */
export function inkOfAll(
  traces: (Run["trace"] | null | undefined)[],
): (tensor: Tensor) => (Ink | null)[] | null {
  const lookups = traces.flatMap((trace) => (trace ? [inkOf(trace)] : []));
  return (tensor) => {
    for (const lookup of lookups) {
      const found = lookup(tensor);
      if (found) return found;
    }
    return null;
  };
}
