import type { Operation, Run, Tensor } from "../api/client";
import type { JourneyStage } from "./stages";
import { mutationInputs, producedTensorIds } from "../tensors/provenance";

export const NODE_WIDTH = 184;
export const NODE_HEIGHT = 192;
const COLUMN_GAP = 98;
const ROW_GAP = 96;

export type JourneyNode = {
  id: string;
  operation?: Operation;
  stage?: JourneyStage;
  tensors: Tensor[];
  parameterCount: number;
  terminal: boolean;
  depth: number;
  x: number;
  y: number;
  /** A shared operation connector used only to bound dense display scenes. */
  junction?: boolean;
  /** A stage showing another loop iteration keeps its first-iteration operations. */
  repOperationIds?: string[];
};
export type JourneyEdge = {
  id: string;
  source: string;
  target: string;
  tensorId: string;
  inputIndex: number;
  kind?: "operand" | "storage";
};
export type JourneyGraph = {
  nodes: JourneyNode[];
  edges: JourneyEdge[];
  width: number;
  height: number;
  /** Display-only tensor actors retain their canonical operation identity. */
  actorOrigins?: Record<string, string>;
  actorRoles?: Record<string, { side: "input" | "output"; index: number }>;
  sceneNodeIds?: Set<string>;
  sceneOperationId?: string;
};

/** Build a dataflow graph from execution order, never from source-line order. */
export function buildJourney(trace: Run["trace"]): JourneyGraph {
  const nodes: JourneyNode[] = [];
  const edges: JourneyEdge[] = [];
  const producers = new Map<string, JourneyNode>();
  function addRoot(id: string) {
    if (producers.has(id)) return producers.get(id)!;
    const tensor = trace.tensors[id];
    const node: JourneyNode = {
      id: `input-${id}`,
      tensors: tensor ? [tensor] : [],
      parameterCount: 0,
      terminal: false,
      depth: 0,
      x: 0,
      y: 0,
    };
    nodes.push(node);
    producers.set(id, node);
    return node;
  }
  trace.input_ids.forEach(addRoot);
  for (const operation of trace.operations) {
    const ids = [...operation.inputs, ...mutationInputs(operation)];
    const parents = ids.flatMap((id, inputIndex) => {
      // Weights are still available in the inspector; omit only unproduced
      // parameter roots so the main graph follows the input's transformations.
      if (!producers.has(id) && trace.tensors[id]?.role === "parameter")
        return [];
      return [
        { node: producers.get(id) ?? addRoot(id), tensorId: id, inputIndex },
      ];
    });
    const node: JourneyNode = {
      id: operation.id,
      operation,
      tensors: producedTensorIds(operation)
        .map((id) => trace.tensors[id])
        .filter(Boolean),
      parameterCount: operation.inputs.filter(
        (id) => trace.tensors[id]?.role === "parameter",
      ).length,
      terminal: false,
      depth: parents.length
        ? Math.max(...parents.map((p) => p.node.depth)) + 1
        : 0,
      x: 0,
      y: 0,
    };
    parents.forEach((parent) =>
      edges.push({
        id: `${node.id}-input-${parent.inputIndex}`,
        source: parent.node.id,
        target: node.id,
        tensorId: parent.tensorId,
        inputIndex: parent.inputIndex,
        kind:
          parent.inputIndex >= operation.inputs.length ? "storage" : "operand",
      }),
    );
    nodes.push(node);
    // Update only after inputs have been connected. This also supports an
    // operation that reuses an earlier tensor ID without introducing cycles.
    producedTensorIds(operation).forEach((id) => producers.set(id, node));
  }
  trace.output_ids.forEach((id) => {
    const node = producers.get(id) ?? addRoot(id);
    node.terminal = true;
  });
  return layoutJourney(nodes, edges);
}

export function layoutJourney(
  nodes: JourneyNode[],
  edges: JourneyEdge[],
): JourneyGraph {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  // Work that does not start from an input (arange, ones, a mask read from a
  // buffer) sits just before its first consumer instead of at the far left,
  // beside the code that uses it. Consumers are placed first.
  const fed = new Set(
    nodes.filter((node) => !node.operation).map((node) => node.id),
  );
  for (const edge of [...edges].sort(
    (a, b) =>
      (byId.get(a.source)?.operation?.index ?? -1) -
      (byId.get(b.source)?.operation?.index ?? -1),
  ))
    if (fed.has(edge.source)) fed.add(edge.target);
  for (const node of [...nodes].reverse()) {
    if (!node.operation || fed.has(node.id)) continue;
    const consumers = edges
      .filter((edge) => edge.source === node.id)
      .map((edge) => byId.get(edge.target)?.depth)
      .filter((depth): depth is number => depth !== undefined);
    if (consumers.length) node.depth = Math.max(0, Math.min(...consumers) - 1);
  }
  const depths = [...new Set(nodes.map((node) => node.depth))].sort(
    (a, b) => a - b,
  );
  for (const depth of depths) {
    const column = nodes
      .filter((node) => node.depth === depth)
      .map((node) => {
        const parentIds = [
          ...new Set(
            edges
              .filter((edge) => edge.target === node.id)
              .map((edge) => edge.source),
          ),
        ];
        const desired = parentIds.length
          ? parentIds.reduce((sum, id) => sum + byId.get(id)!.y, 0) /
            parentIds.length
          : 0;
        return { node, desired };
      })
      .sort((a, b) => a.desired - b.desired);
    let previous = -Infinity;
    for (const item of column) {
      item.node.y = Math.max(item.desired, previous + NODE_HEIGHT + ROW_GAP);
      item.node.x = depth * (NODE_WIDTH + COLUMN_GAP);
      previous = item.node.y;
    }
    const shift =
      column.reduce((sum, item) => sum + item.node.y - item.desired, 0) /
      column.length;
    column.forEach(({ node }) => {
      node.y -= shift;
    });
  }
  const top = Math.min(0, ...nodes.map((node) => node.y));
  nodes.forEach((node) => {
    node.x += 48;
    node.y += 48 - top;
  });
  return {
    nodes,
    edges,
    width: Math.max(0, ...nodes.map((node) => node.x + NODE_WIDTH)) + 48,
    height: Math.max(0, ...nodes.map((node) => node.y + NODE_HEIGHT)) + 48,
  };
}

/** Everything a node's tensors flow into, directly or through later steps. */
export function descendants(graph: JourneyGraph, id: string): Set<string> {
  const found = new Set([id]);
  const queue = [id];
  for (let i = 0; i < queue.length; i++) {
    graph.edges
      .filter((edge) => edge.source === queue[i])
      .forEach((edge) => {
        if (!found.has(edge.target)) {
          found.add(edge.target);
          queue.push(edge.target);
        }
      });
  }
  return found;
}

/** Transitive ancestors identify the real dependencies of a selected operation. */
export function ancestors(graph: JourneyGraph, id: string): Set<string> {
  const found = new Set([id]);
  const queue = [id];
  for (let i = 0; i < queue.length; i++) {
    graph.edges
      .filter((edge) => edge.target === queue[i])
      .forEach((edge) => {
        if (!found.has(edge.source)) {
          found.add(edge.source);
          queue.push(edge.source);
        }
      });
  }
  return found;
}
