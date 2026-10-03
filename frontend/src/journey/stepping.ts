import type { PlaybackStep } from "./loops";
import { stageAncestors, type JourneyStage } from "./stages";

/**
 * A playback step: one operation, a folded loop's repeats, or a folded call or
 * capsule, which plays as one step running all of its operations together.
 */
export type PlayStep =
  | (PlaybackStep & { stage?: undefined })
  | {
      id: string;
      stage: JourneyStage;
      operation?: undefined;
      fold?: undefined;
    };

/**
 * Playback over what the canvas draws: the steps inside each outermost folded
 * stage become that stage, one step, so the count and the stepping keys match
 * the cards on screen. A loop whose body lies inside a folded stage repeats
 * inside it too.
 */
export function foldedPlayback(
  steps: PlaybackStep[],
  stages: JourneyStage[],
  collapsed: ReadonlySet<string>,
): PlayStep[] {
  // Each operation's outermost folded stage: larger stages written last win.
  const owner = new Map<string, JourneyStage>();
  for (const stage of stages
    .filter((item) => collapsed.has(item.id))
    .sort((a, b) => a.operationIds.length - b.operationIds.length))
    for (const id of stage.operationIds) owner.set(id, stage);
  if (!owner.size) return steps;
  const played: PlayStep[] = [];
  for (const step of steps) {
    const stage = owner.get(
      step.operation ? step.operation.id : step.fold.iterations[0][0],
    );
    if (!stage) played.push(step);
    else if (played.at(-1)?.id !== stage.id && step.operation)
      played.push({ id: stage.id, stage });
  }
  return played;
}

/**
 * How deep each playback step sits in recorded module calls: the number of
 * stages (calls of two or more operations) around it. A loop's repeats sit
 * at the depth of its body's shallowest step.
 */
export function stepDepths(
  steps: PlayStep[],
  stages: JourneyStage[],
): number[] {
  const depth = (id: string) => stageAncestors(stages, id).length;
  return steps.map((step) =>
    step.operation
      ? depth(step.operation.id)
      : step.stage
        ? // A folded card sits where its call is made, beside the steps
          // around it: the stages that hold it, not those inside it.
          stageAncestors(stages, step.stage.operationIds[0]).filter(
            (around) =>
              around.operationIds.length > step.stage.operationIds.length,
          ).length
        : Math.min(...step.fold.iterations[0].map(depth)),
  );
}

/** Step over: the next step no deeper than this one, past the calls it makes. */
export function stepOverTarget(depths: number[], from: number): number | null {
  if (from < 0) return depths.length ? 0 : null;
  for (let next = from + 1; next < depths.length; next++)
    if (depths[next] <= depths[from]) return next;
  return null;
}

/** Step out: the next step outside the innermost call around this one. */
export function stepOutTarget(depths: number[], from: number): number | null {
  if (from < 0) return null;
  for (let next = from + 1; next < depths.length; next++)
    if (depths[next] < depths[from]) return next;
  return null;
}

/**
 * Whether playback arriving at a step should stop for one of `ids` (break-
 * point or warning steps). A loop's repeats stop when any later pass holds one,
 * and a folded card when any of its operations does.
 */
export function stopsAt(step: PlayStep, ids?: ReadonlySet<string>) {
  return !!stopInside(step, ids);
}

/**
 * The operation of `ids` a step would stop for, if any: itself, one in a
 * loop's later passes, or one inside a folded card, where playback stops on
 * the card.
 */
export function stopInside(
  step: PlayStep,
  ids?: ReadonlySet<string>,
): string | undefined {
  if (!ids?.size) return undefined;
  if (step.operation)
    return ids.has(step.operation.id) ? step.operation.id : undefined;
  if (step.stage) return step.stage.operationIds.find((id) => ids.has(id));
  return step.fold.iterations
    .slice(1)
    .flat()
    .find((id) => ids.has(id));
}
