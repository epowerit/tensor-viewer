import { NODE_HEIGHT, NODE_WIDTH, type JourneyNode } from "./graph";
import type { JourneyStage } from "./stages";

/** An open call or capsule drawn around the cards it holds. */
export type StageFrame = {
  stage: JourneyStage;
  left: number;
  top: number;
  right: number;
  bottom: number;
};

/**
 * A frame around each open stage's cards, so it can be folded again where it
 * is drawn. A card belongs to a stage when its step is one of the stage's,
 * or when it is a folded stage inside it. A stage drawn as fewer than two
 * cards gets no frame. Frames holding other frames leave room around them,
 * so nested calls read as nested. A step drawn as its scene brings its
 * operands into the frames around it, so the call holds the whole scene.
 */
export function frameStages(
  open: JourneyStage[],
  nodes: JourneyNode[],
  scene?: { operationId: string; nodeIds: ReadonlySet<string> } | null,
): StageFrame[] {
  const members = new Map(
    open.map((stage) => {
      const ids = new Set(stage.operationIds);
      const holdsScene = !!scene && ids.has(scene.operationId);
      return [
        stage.id,
        nodes.filter((node) =>
          holdsScene && scene.nodeIds.has(node.id)
            ? true
            : node.stage
              ? node.stage.id !== stage.id &&
                (node.repOperationIds ?? node.stage.operationIds).every((id) =>
                  ids.has(id),
                )
              : ids.has(node.id),
        ),
      ] as const;
    }),
  );
  const framed = open.filter((stage) => members.get(stage.id)!.length >= 2);
  const framedIds = new Set(framed.map((stage) => stage.id));
  // How many levels of framed stages sit inside each one.
  const inner = new Map<string, number>();
  const nesting = (stage: JourneyStage): number => {
    if (inner.has(stage.id)) return inner.get(stage.id)!;
    inner.set(stage.id, 0);
    const levels = framed
      .filter(
        (other) => other.parentStageId === stage.id && framedIds.has(other.id),
      )
      .map((child) => nesting(child) + 1);
    const value = Math.max(0, ...levels);
    inner.set(stage.id, value);
    return value;
  };
  return framed.map((stage) => {
    const cards = members.get(stage.id)!;
    const pad = 14 + 12 * nesting(stage);
    return {
      stage,
      left: Math.min(...cards.map((node) => node.x)) - pad,
      top: Math.min(...cards.map((node) => node.y)) - pad,
      right: Math.max(...cards.map((node) => node.x + NODE_WIDTH)) + pad,
      bottom: Math.max(...cards.map((node) => node.y + NODE_HEIGHT)) + pad,
    };
  });
}
