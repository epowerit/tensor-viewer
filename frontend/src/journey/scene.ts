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
/** The room a scene keeps above it for the step's caption. */
export const CAPTION_ROOM = 148;
/** The smallest scale at which tensor cards still draw cells. */
const SMALLEST = 0.45;

export function sceneView(
  graph: JourneyGraph,
  selectedId: string,
  size: CanvasSize,
  previous: Viewport,
  /** Extra room above the actors, such as for a loop frame's header. */
  headroom = 0,
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
  const captionInset = CAPTION_ROOM + headroom;
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
  // Down to the smallest scale that still draws cells; past it, the active
  // step keeps the center and its operands go partly out of view.
  const scale = Math.max(SMALLEST, fit);
  const center =
    fit >= SMALLEST
      ? { x: (left + right) / 2, y: (top + bottom) / 2 }
      : { x: active.x + NODE_WIDTH / 2, y: active.y + NODE_HEIGHT / 2 };
  const centered = captionInset + available.height / 2 - center.y * scale;
  // Too tall to fit, the scene hangs from the caption rather than slipping
  // under it, while the active step stays whole: the overflow goes below.
  const hung = captionInset - top * scale;
  const activeBottom = (active.y + NODE_HEIGHT) * scale + hung;
  const y =
    fit < SMALLEST &&
    top * scale + centered < captionInset &&
    activeBottom <= captionInset + available.height
      ? hung
      : centered;
  return { scale, x: size.width / 2 - center.x * scale, y };
}
