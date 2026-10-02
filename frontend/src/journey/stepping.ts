import type { PlaybackStep } from "./loops";
import { stageAncestors, type JourneyStage } from "./stages";

/**
 * How deep each playback step sits in recorded module calls: the number of
 * stages (calls of two or more operations) around it. A loop's repeats sit
 * at the depth of its body's shallowest step.
 */
export function stepDepths(
  steps: PlaybackStep[],
  stages: JourneyStage[],
): number[] {
  const depth = (id: string) => stageAncestors(stages, id).length;
  return steps.map((step) =>
    step.operation
      ? depth(step.operation.id)
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
 * point or warning steps). A loop's repeats stop when any later pass holds one.
 */
export function stopsAt(step: PlaybackStep, ids?: ReadonlySet<string>) {
  if (!ids?.size) return false;
  return step.operation
    ? ids.has(step.operation.id)
    : step.fold.iterations
        .slice(1)
        .some((ops) => ops.some((id) => ids.has(id)));
}
