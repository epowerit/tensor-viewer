import { NODE_HEIGHT, type JourneyEdge, type JourneyGraph } from "./graph";

export type EdgePorts = { sourceY: number; targetY: number };

/** Ports stay within the tensor drawing, even for a join with hundreds of operands. */
export function portPosition(index: number, count: number): number {
  const inset = 48;
  return count <= 1
    ? NODE_HEIGHT / 2
    : inset +
        (Math.max(0, Math.min(index, count - 1)) / (count - 1)) *
          (NODE_HEIGHT - 2 * inset);
}

export function journeyPorts(graph: JourneyGraph): Map<string, EdgePorts> {
  const junctions = new Set(
    graph.nodes.filter((node) => node.junction).map((node) => node.id),
  );
  const incoming = new Map<string, JourneyEdge[]>();
  const outputs = new Map(
    graph.nodes.map((node) => [
      node.id,
      [...new Set(node.tensors.map((tensor) => tensor.id))],
    ]),
  );
  for (const edge of graph.edges) {
    const list = incoming.get(edge.target) ?? [];
    list.push(edge);
    incoming.set(edge.target, list);
    const carried = outputs.get(edge.source) ?? [];
    if (!carried.includes(edge.tensorId)) carried.push(edge.tensorId);
    outputs.set(edge.source, carried);
  }
  const targets = new Map<string, number>();
  for (const list of incoming.values())
    [...list]
      .sort((a, b) => a.inputIndex - b.inputIndex)
      .forEach((edge, index) =>
        targets.set(edge.id, portPosition(index, list.length)),
      );
  return new Map(
    graph.edges.map((edge) => {
      const carried = outputs.get(edge.source)!;
      return [
        edge.id,
        {
          sourceY: junctions.has(edge.source)
            ? NODE_HEIGHT / 2
            : portPosition(carried.indexOf(edge.tensorId), carried.length),
          targetY: junctions.has(edge.target)
            ? NODE_HEIGHT / 2
            : targets.get(edge.id)!,
        },
      ];
    }),
  );
}

export function edgeDescription(
  graph: JourneyGraph,
  edge: JourneyEdge,
): string {
  const source = graph.nodes.find((node) => node.id === edge.source);
  const target = graph.nodes.find((node) => node.id === edge.target);
  const tensor = source?.tensors.find((item) => item.id === edge.tensorId);
  const role =
    edge.kind === "storage"
      ? "shared storage"
      : target?.stage
        ? "used within stage"
        : `input ${edge.inputIndex + 1}`;
  return `${tensor?.name ?? edge.tensorId}${tensor ? ` [${tensor.shape.join(", ")}]` : ""} → ${target?.stage?.title ?? target?.operation?.kind ?? "tensor"}, ${role}`;
}
