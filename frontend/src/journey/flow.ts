import type { Tensor } from "../api/client";
import type { StepDiff } from "../workspace/compare";
import type { JourneyEdge, JourneyNode } from "./graph";
import { NODE_HEIGHT, NODE_WIDTH } from "./graph";

type FlowGraph = { nodes: JourneyNode[]; edges: JourneyEdge[] };

const carried = (graph: FlowGraph, edge: JourneyEdge) =>
  graph.nodes
    .find((node) => node.id === edge.source)
    ?.tensors.find((tensor) => tensor.id === edge.tensorId);

/**
 * Line widths that show how much data each connection carries: half a pixel
 * more for every tenfold increase in elements, so a 64-value activation draws
 * at about 2px and a 65,536-value attention map at about 3.6px. The scale is
 * absolute, so tensors of similar size look alike in every model.
 */
export function edgeWidths(graph: FlowGraph): Map<string, number> {
  const widths = new Map<string, number>();
  for (const edge of graph.edges) {
    const tensor = carried(graph, edge);
    if (tensor && edge.kind !== "storage")
      widths.set(
        edge.id,
        Math.min(4.2, 1.2 + 0.5 * Math.log10(Math.max(1, tensor.numel))),
      );
  }
  return widths;
}

/**
 * The main line of flow into a node: back through each step's first operand
 * to where it started, returned from the start. It is the path a reader
 * follows to see how the shape evolved.
 */
export function mainPath(graph: FlowGraph, id: string): JourneyNode[] {
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const path: JourneyNode[] = [];
  const seen = new Set<string>();
  let current = byId.get(id);
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    path.unshift(current);
    const into = graph.edges
      .filter((edge) => edge.target === current!.id && edge.kind !== "storage")
      .sort((a, b) => a.inputIndex - b.inputIndex)[0];
    current = into ? byId.get(into.source) : undefined;
  }
  return path;
}

/**
 * Where the spread of values (σ) jumps along a path: more than tenfold up or
 * down from one step to the next, the signature of exploding or vanishing
 * activations. Returns the factor by node id; steps without statistics are
 * skipped, and the comparison resumes at the next step that has them.
 */
export function spreadJumps(path: JourneyNode[]): Map<string, number> {
  const jumps = new Map<string, number>();
  let previous: number | undefined;
  for (const node of path) {
    const std = node.tensors[0]?.histogram?.std;
    if (typeof std !== "number" || !Number.isFinite(std)) continue;
    if (previous !== undefined && previous > 0 && std > 0) {
      const factor = std / previous;
      if (factor >= 10 || factor <= 0.1) jumps.set(node.id, factor);
    }
    previous = std;
  }
  return jumps;
}

/**
 * The next step along the flow from a node: forward to the earliest step that
 * reads its tensor, or back to the producer of its first operand.
 */
export function flowNeighbor(
  graph: FlowGraph,
  id: string,
  direction: "forward" | "back",
): string | undefined {
  const order = (nodeId: string) =>
    graph.nodes.find((node) => node.id === nodeId)?.operation?.index ??
    Infinity;
  if (direction === "forward")
    return graph.edges
      .filter((edge) => edge.source === id && edge.kind !== "storage")
      .map((edge) => edge.target)
      .sort((a, b) => order(a) - order(b))[0];
  return graph.edges
    .filter((edge) => edge.target === id && edge.kind !== "storage")
    .sort((a, b) => a.inputIndex - b.inputIndex)[0]?.source;
}

type Box = { id: string; x: number; y: number };

/**
 * The path of a connection from (x, y) into `to` at endY. A straight curve
 * that would pass through nodes standing between its ends, such as a
 * residual connection reaching back over a block, instead rises just above
 * the highest of them, runs level, and drops into its target, so a skip
 * reads as a skip. Returns the path and whether it was rerouted.
 */
export function routeEdge(
  from: Box,
  to: Box,
  x: number,
  y: number,
  endY: number,
  nodes: Box[],
): { path: string; skip: boolean; mid: { x: number; y: number } } {
  const bend = Math.max(38, (to.x - x) * 0.5);
  const straight = `M${x},${y} C${x + bend},${y} ${to.x - bend},${endY} ${to.x - 3},${endY}`;
  const middle = { x: (x + to.x) / 2, y: (y + endY) / 2 };
  if (to.x - x <= NODE_WIDTH)
    return { path: straight, skip: false, mid: middle };
  const blockers = nodes.filter((node) => {
    if (node.id === from.id || node.id === to.id) return false;
    if (node.x <= x || node.x + NODE_WIDTH >= to.x) return false;
    const t = (node.x + NODE_WIDTH / 2 - x) / (to.x - x);
    const lineY = y + (endY - y) * t;
    return lineY > node.y - 8 && lineY < node.y + NODE_HEIGHT + 8;
  });
  if (!blockers.length) return { path: straight, skip: false, mid: middle };
  const top = Math.min(...blockers.map((node) => node.y)) - 26;
  const rise = Math.min(70, (to.x - x) / 4);
  return {
    path: `M${x},${y} C${x + rise},${y} ${x + rise},${top} ${x + rise * 2},${top} L${to.x - rise * 2},${top} C${to.x - rise},${top} ${to.x - rise},${endY} ${to.x - 3},${endY}`,
    skip: true,
    mid: { x: middle.x, y: top },
  };
}

export type FlowLens = "spread" | "zeros" | "magnitude" | "change";

/** A tensor's value for a lens, from its recorded histogram. */
export function lensValue(tensor: Tensor | undefined, lens: FlowLens) {
  const histogram = tensor?.histogram;
  if (!histogram || !histogram.counts.length) return undefined;
  if (lens === "spread")
    return typeof histogram.std === "number" ? histogram.std : undefined;
  if (lens === "magnitude")
    return Math.max(Math.abs(histogram.low), Math.abs(histogram.high));
  const total =
    histogram.counts.reduce((sum, count) => sum + count, 0) +
    histogram.non_finite;
  return total ? histogram.zeros / total : undefined;
}

/**
 * Place each value on a 0–1 scale for colouring: zeros as a plain fraction,
 * spread and magnitude on a log scale between the graph's own extremes, so
 * a tenfold jump stands out whatever the model's typical size.
 */
export function lensScale(
  values: number[],
  lens: FlowLens,
): (value: number) => number {
  if (lens === "zeros")
    return (value: number) => Math.min(1, Math.max(0, value));
  if (lens === "change") {
    // Unchanged is cold, a structural change hottest; differences between.
    const finite = values.filter((v) => v > 0 && Number.isFinite(v));
    const inner = lensScale(finite, "spread");
    return (value: number) =>
      value === 0 ? 0 : value === Infinity ? 1 : 0.25 + 0.65 * inner(value);
  }
  const logs = values.filter((v) => v > 0).map(Math.log10);
  const low = Math.min(...logs),
    high = Math.max(...logs);
  return (value: number) =>
    value <= 0 || !logs.length
      ? 0
      : high > low
        ? (Math.log10(value) - low) / (high - low)
        : 0.5;
}

/** Cool (low) through neutral to hot (high). */
export function lensColor(t: number): string {
  const stops = [
    [110, 168, 217],
    [185, 166, 211],
    [242, 166, 108],
  ];
  const at = Math.min(1, Math.max(0, t)) * 2;
  const [a, b] = at <= 1 ? [stops[0], stops[1]] : [stops[1], stops[2]];
  const f = at <= 1 ? at : at - 1;
  const mix = a.map((c, i) => Math.round(c + (b[i] - c) * f));
  return `rgb(${mix.join(", ")})`;
}

/**
 * How much each step changed since another run, by operation id: 0 when its
 * values are the same, the largest difference when they differ (or the mean
 * shift of a large tensor), and Infinity when its shape or the step changed.
 */
export function changeValues(diffs: StepDiff[]): Map<string, number> {
  const values = new Map<string, number>();
  for (const diff of diffs) {
    if (!diff.left) continue;
    values.set(
      diff.left.id,
      diff.change === "same"
        ? 0
        : diff.change === "values"
          ? (diff.delta ??
            (diff.shift
              ? Math.abs(diff.shift.after - diff.shift.before)
              : Number.MIN_VALUE))
          : Infinity,
    );
  }
  return values;
}

const formatLens = (value: number) =>
  value >= 1000 || (value !== 0 && value < 0.01)
    ? value.toExponential(1)
    : String(Number(value.toPrecision(2)));

/** A lens value as the chip shows it. */
export function lensText(value: number, lens: FlowLens): string {
  if (lens === "change")
    return value === 0
      ? "unchanged"
      : value === Infinity
        ? "shape changed"
        : `Δ ${formatLens(value)}`;
  if (lens === "zeros") return `${Math.round(value * 100)}% zeros`;
  const text = formatLens(value);
  return lens === "spread" ? `σ ${text}` : `|x| ≤ ${text}`;
}

/**
 * The previous or next step that reads the same tensor as this one's first
 * operand, in step order: the other branches a fanned-out tensor feeds.
 */
export function flowSibling(
  graph: FlowGraph,
  id: string,
  direction: "next" | "previous",
): string | undefined {
  const input = graph.edges
    .filter((edge) => edge.target === id && edge.kind !== "storage")
    .sort((a, b) => a.inputIndex - b.inputIndex)[0];
  if (!input) return undefined;
  const order = (nodeId: string) =>
    graph.nodes.find((node) => node.id === nodeId)?.operation?.index ??
    Infinity;
  const readers = [
    ...new Set(
      graph.edges
        .filter(
          (edge) =>
            edge.source === input.source &&
            edge.tensorId === input.tensorId &&
            edge.kind !== "storage",
        )
        .map((edge) => edge.target),
    ),
  ].sort((a, b) => order(a) - order(b));
  if (readers.length < 2) return undefined;
  const at = readers.indexOf(id);
  return readers[
    (at + (direction === "next" ? 1 : readers.length - 1)) % readers.length
  ];
}
