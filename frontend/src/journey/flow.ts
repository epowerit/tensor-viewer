import type { JourneyEdge, JourneyNode } from "./graph";

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
