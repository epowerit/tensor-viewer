import type { Run } from "../api/client";
import { tensorProducer } from "../tensors/provenance";
import type { JourneyGraph, JourneyNode } from "./graph";

export type JourneyConnection = {
  key: string;
  /** The neighboring node in the supplied (possibly collapsed) graph. */
  nodeId: string;
  tensorId: string;
  kind: "operand" | "storage";
  /** Zero-based graph edge slots, not module arguments for collapsed stages. */
  inputIndices: number[];
  operandRole: string;
};

/** Identify final returned states, rather than every node that once used their IDs. */
export function returnedTensorIds(
  graph: JourneyGraph,
  nodeId: string,
  trace: Run["trace"],
): string[] {
  const node = graph.nodes.find((entry) => entry.id === nodeId);
  if (!node) return [];
  return [...new Set(trace.output_ids)].filter((id) => {
    const producer = tensorProducer(trace.operations, id);
    if (producer)
      return (
        node.operation?.id === producer.id ||
        !!node.stage?.operationIds.includes(producer.id)
      );
    return (
      !node.operation &&
      !node.stage &&
      node.tensors.some((tensor) => tensor.id === id)
    );
  });
}

export function journeyConnections(
  graph: JourneyGraph,
  nodeId: string,
  trace: Run["trace"],
): { incoming: JourneyConnection[]; outgoing: JourneyConnection[] } {
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const selected = nodes.get(nodeId);
  if (!selected) return { incoming: [], outgoing: [] };
  const executionOrder = new Map(
    trace.operations.map((operation, index) => [operation.id, index]),
  );
  const nodeOrder = new Map(graph.nodes.map((node, index) => [node.id, index]));
  const tensorOrder = new Map(
    Object.keys(trace.tensors).map((id, index) => [id, index]),
  );
  const position = (node: JourneyNode) =>
    node.operation
      ? (executionOrder.get(node.operation.id) ?? node.operation.index)
      : (node.stage?.start_index ?? -1);

  function connections(direction: "incoming" | "outgoing") {
    const grouped = new Map<string, JourneyConnection>();
    for (const edge of graph.edges) {
      if (
        (direction === "incoming" ? edge.target : edge.source) !== nodeId ||
        edge.source === edge.target
      )
        continue;
      const neighborId = direction === "incoming" ? edge.source : edge.target;
      const neighbor = nodes.get(neighborId);
      if (!neighbor) continue;
      const kind = edge.kind ?? "operand";
      const key = JSON.stringify([direction, neighborId, edge.tensorId, kind]);
      const existing = grouped.get(key);
      if (existing) {
        if (!existing.inputIndices.includes(edge.inputIndex))
          existing.inputIndices.push(edge.inputIndex);
      } else {
        grouped.set(key, {
          key,
          nodeId: neighborId,
          tensorId: edge.tensorId,
          kind,
          inputIndices: [edge.inputIndex],
          operandRole: "",
        });
      }
    }
    const entries = [...grouped.values()];
    for (const entry of entries) {
      entry.inputIndices.sort((a, b) => a - b);
      const target =
        direction === "incoming" ? selected! : nodes.get(entry.nodeId)!;
      entry.operandRole =
        entry.kind === "storage"
          ? "Shared storage"
          : target.stage
            ? "Used within stage"
            : `${entry.inputIndices.length === 1 ? "Input" : "Inputs"} ${entry.inputIndices.map((index) => index + 1).join(", ")}`;
    }
    return entries.sort((a, b) => {
      const left = nodes.get(a.nodeId)!;
      const right = nodes.get(b.nodeId)!;
      return (
        position(left) - position(right) ||
        nodeOrder.get(left.id)! - nodeOrder.get(right.id)! ||
        a.inputIndices[0] - b.inputIndices[0] ||
        (tensorOrder.get(a.tensorId) ?? Number.MAX_SAFE_INTEGER) -
          (tensorOrder.get(b.tensorId) ?? Number.MAX_SAFE_INTEGER) ||
        a.tensorId.localeCompare(b.tensorId) ||
        a.kind.localeCompare(b.kind)
      );
    });
  }

  return {
    incoming: connections("incoming"),
    outgoing: connections("outgoing"),
  };
}
