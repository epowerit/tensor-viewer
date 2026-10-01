import { NODE_HEIGHT, NODE_WIDTH, type JourneyGraph } from "./graph";
import type { CanvasSize, Viewport } from "./viewport";

/** A scene is one recorded operation and its actual incoming dependencies. */
export function operationScene(graph: JourneyGraph, selectedId: string | null) {
  const nodes = new Set<string>();
  const edges = new Set<string>();
  if (!selectedId || !graph.nodes.some((node) => node.id === selectedId))
    return { nodes, edges };
  if (graph.sceneOperationId === selectedId && graph.sceneNodeIds) {
    for (const id of graph.sceneNodeIds) nodes.add(id);
    const junctions = new Set(
      graph.nodes.filter((node) => node.junction).map((node) => node.id),
    );
    for (const edge of graph.edges)
      if (
        nodes.has(edge.source) &&
        nodes.has(edge.target) &&
        (graph.actorRoles?.[edge.target]?.side === "output" ||
          junctions.has(edge.target))
      )
        edges.add(edge.id);
    return { nodes, edges };
  }
  nodes.add(selectedId);
  for (const edge of graph.edges) {
    if (edge.target !== selectedId) continue;
    nodes.add(edge.source);
    edges.add(edge.id);
  }
  return { nodes, edges };
}

/** Frame operands together with their result without shrinking cells to dots.
 * Large fan-ins remain in the graph; the camera centers on the result when
 * their full extent cannot be shown at a readable scale.
 */
export function sceneView(
  graph: JourneyGraph,
  selectedId: string,
  size: CanvasSize,
  previous: Viewport,
): Viewport {
  const active = graph.nodes.find((node) => node.id === selectedId);
  if (!active || !size.width || !size.height) return previous;
  const scene = operationScene(graph, selectedId);
  const actors = graph.nodes.filter((node) => scene.nodes.has(node.id));
  const left = Math.min(...actors.map((node) => node.x));
  const top = Math.min(...actors.map((node) => node.y));
  const right = Math.max(...actors.map((node) => node.x + NODE_WIDTH));
  const bottom = Math.max(...actors.map((node) => node.y + NODE_HEIGHT));
  // Leave room for a two-line cell trace without moving the camera on selection.
  const captionInset = 148;
  const available = {
    width: Math.max(
      NODE_WIDTH * 0.55,
      size.width - (size.width < 600 ? 40 : 112),
    ),
    height: Math.max(
      NODE_HEIGHT * 0.55,
      size.height - captionInset - (size.width < 600 ? 164 : 96),
    ),
  };
  const fit = Math.min(
    1.5,
    available.width / (right - left),
    available.height / (bottom - top),
  );
  const scale = Math.max(0.55, fit);
  const center =
    fit >= 0.55
      ? { x: (left + right) / 2, y: (top + bottom) / 2 }
      : { x: active.x + NODE_WIDTH / 2, y: active.y + NODE_HEIGHT / 2 };
  return {
    scale,
    x: size.width / 2 - center.x * scale,
    y: captionInset + available.height / 2 - center.y * scale,
  };
}
