import type { Tensor } from "../api/client";
import { producedTensorIds } from "../tensors/provenance";
import {
  layoutJourney,
  type JourneyEdge,
  type JourneyGraph,
  type JourneyNode,
} from "./graph";

export type ProjectedSceneGraph = JourneyGraph & {
  actorOrigins: Record<string, string>;
  actorRoles: Record<string, { side: "input" | "output"; index: number }>;
  sceneNodeIds: Set<string>;
  sceneOperationId: string;
};

const DIRECT_EDGE_LIMIT = 256;

/**
 * A display projection: canonical operations and the saved trace are untouched.
 * Each active result and each tensor used by it has its own visual actor.
 * Edges remain operation dependencies, not claims of cell-level contribution.
 */
export function projectOperationScene(
  graph: JourneyGraph,
  operationId: string,
  tensors: Record<string, Tensor>,
): ProjectedSceneGraph {
  const active = graph.nodes.find((node) => node.id === operationId);
  const operation = active?.operation;
  const actorOrigins: Record<string, string> = {};
  const actorRoles: ProjectedSceneGraph["actorRoles"] = {};
  const sceneNodeIds = new Set<string>();
  if (!active || !operation)
    return {
      ...graph,
      actorOrigins,
      actorRoles,
      sceneNodeIds,
      sceneOperationId: operationId,
    };

  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  const edgeIds = new Set(graph.edges.map((edge) => edge.id));
  function uniqueId(prefix: string, used: Set<string>) {
    let id = prefix;
    for (let suffix = 1; used.has(id); suffix++) id = `${prefix}:${suffix}`;
    used.add(id);
    return id;
  }
  const incoming = graph.edges.filter((edge) => edge.target === operationId);
  const expanded = new Set([
    operationId,
    ...incoming.map((edge) => edge.source),
  ]);
  const actors = new Map<string, JourneyNode[]>();
  const sourceActors = new Map<string, Map<string, string>>();
  const nodes: JourneyNode[] = [];
  for (const original of graph.nodes) {
    const values =
      original.id === operationId
        ? producedTensorIds(operation)
            .map((id) => tensors[id])
            .filter(Boolean)
        : original.tensors;
    // Preserve sibling outputs as surrounding model context. An outgoing edge
    // must never attach to a different tensor merely because it is not in focus.
    const displayed = expanded.has(original.id)
      ? values.map((tensor) => [tensor])
      : [values];
    if (!displayed.length) displayed.push([]);
    const siblings = displayed.map((items, index) => {
      const id = index
        ? uniqueId(`scene:${encodeURIComponent(original.id)}:${index}`, nodeIds)
        : original.id;
      const node = { ...original, id, tensors: [...items] };
      actorOrigins[id] = original.id;
      nodes.push(node);
      return node;
    });
    actors.set(original.id, siblings);
    sourceActors.set(
      original.id,
      new Map(
        siblings.flatMap((node) =>
          node.tensors.map((tensor) => [tensor.id, node.id]),
        ),
      ),
    );
  }

  const dependencies: JourneyEdge[] = [...graph.edges];
  const absentRoots = new Map<string, JourneyNode>();
  operation.inputs.forEach((tensorId, inputIndex) => {
    if (
      incoming.some(
        (edge) =>
          edge.kind !== "storage" &&
          edge.inputIndex === inputIndex &&
          edge.tensorId === tensorId,
      )
    )
      return;
    const tensor = tensors[tensorId];
    if (!tensor) return;
    let root = absentRoots.get(tensorId);
    if (!root) {
      const canonicalId = `input-${tensorId}`;
      const id = uniqueId(canonicalId, nodeIds);
      root = {
        id,
        tensors: [tensor],
        parameterCount: 0,
        terminal: false,
        depth: 0,
        x: 0,
        y: 0,
      };
      nodes.push(root);
      absentRoots.set(tensorId, root);
      actors.set(id, [root]);
      sourceActors.set(id, new Map([[tensorId, id]]));
      actorOrigins[id] = canonicalId;
    }
    dependencies.push({
      id: uniqueId(
        `scene:${encodeURIComponent(operationId)}:input:${inputIndex}`,
        edgeIds,
      ),
      source: root.id,
      target: operationId,
      tensorId,
      inputIndex,
      kind: "operand",
    });
  });

  const dependencyCounts = new Map<string, number>();
  for (const edge of dependencies)
    dependencyCounts.set(
      edge.target,
      (dependencyCounts.get(edge.target) ?? 0) + 1,
    );
  const junctions = new Map<string, JourneyNode>();
  for (const original of graph.nodes) {
    if (!expanded.has(original.id)) continue;
    const outputs = actors.get(original.id)!;
    if (
      !outputs.every((node) => node.tensors.length === 1) ||
      (dependencyCounts.get(original.id) ?? 0) * outputs.length <=
        DIRECT_EDGE_LIMIT
    )
      continue;
    const id = uniqueId(
      `scene:${encodeURIComponent(original.id)}:junction`,
      nodeIds,
    );
    const junction: JourneyNode = {
      ...original,
      id,
      tensors: [],
      terminal: false,
      junction: true,
    };
    junctions.set(original.id, junction);
    actorOrigins[id] = original.id;
    nodes.push(junction);
  }
  // Decide on bundling before allocating any per-output edges. The junction
  // represents the recorded operation, not an invented tensor or extra step.
  const edges = dependencies.flatMap((edge) => {
    const source =
      sourceActors.get(edge.source)?.get(edge.tensorId) ?? edge.source;
    const junction = junctions.get(edge.target);
    if (junction) return [{ ...edge, source, target: junction.id }];
    return (actors.get(edge.target) ?? []).map((target, index) => ({
      ...edge,
      id: index
        ? uniqueId(`scene:${encodeURIComponent(edge.id)}:${index}`, edgeIds)
        : edge.id,
      source,
      target: target.id,
    }));
  });
  for (const [ownerId, junction] of junctions)
    actors.get(ownerId)!.forEach((output, inputIndex) => {
      edges.push({
        id: uniqueId(
          `scene:${encodeURIComponent(ownerId)}:result:${inputIndex}`,
          edgeIds,
        ),
        source: junction.id,
        target: output.id,
        tensorId: output.tensors[0].id,
        inputIndex,
        kind: "operand",
      });
    });
  const results = actors.get(operationId)!;
  results.forEach((node, index) => {
    sceneNodeIds.add(node.id);
    actorRoles[node.id] = { side: "output", index };
  });
  const activeJunction = junctions.get(operationId);
  if (activeJunction) sceneNodeIds.add(activeJunction.id);
  const destinations = new Set(
    activeJunction ? [activeJunction.id] : results.map((node) => node.id),
  );
  for (const edge of edges) {
    if (!destinations.has(edge.target)) continue;
    sceneNodeIds.add(edge.source);
    const role = actorRoles[edge.source];
    if (!role || role.index > edge.inputIndex)
      actorRoles[edge.source] = { side: "input", index: edge.inputIndex };
  }

  // Recompute depth for parameter-only operations, which previously had no
  // visible parents. Counting unique parents preserves repeated operand edges.
  const parents = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  const children = new Map(nodes.map((node) => [node.id, new Set<string>()]));
  const byId = new Map(nodes.map((node) => [node.id, node]));
  for (const edge of edges) {
    parents.get(edge.target)?.add(edge.source);
    children.get(edge.source)?.add(edge.target);
  }
  const remaining = new Map([...parents].map(([id, set]) => [id, set.size]));
  const queue = nodes.filter((node) => !remaining.get(node.id));
  queue.forEach((node) => {
    node.depth = 0;
  });
  for (let index = 0; index < queue.length; index++) {
    const node = queue[index];
    for (const id of children.get(node.id) ?? []) {
      const count = remaining.get(id)! - 1;
      remaining.set(id, count);
      if (!count) {
        const child = byId.get(id)!;
        child.depth =
          Math.max(
            ...[...parents.get(id)!].map((parent) => byId.get(parent)!.depth),
          ) + 1;
        queue.push(child);
      }
    }
  }
  // Scene-only parameter roots belong beside this operation's other operands,
  // rather than at the beginning of an arbitrarily deep recorded model.
  // Canonical roots keep their original place in the dataflow.
  const operandDepth = Math.max(0, (activeJunction ?? results[0]).depth - 1);
  for (const root of absentRoots.values()) root.depth = operandDepth;
  return {
    ...layoutJourney(nodes, edges),
    actorOrigins,
    actorRoles,
    sceneNodeIds,
    sceneOperationId: operationId,
  };
}
