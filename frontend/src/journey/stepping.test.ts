import { describe, expect, it } from "vitest";
import type { Operation } from "../api/client";
import type { LoopFold, PlaybackStep } from "./loops";
import type { JourneyStage } from "./stages";
import {
  foldedPlayback,
  stepDepths,
  stopInside,
  stepOutTarget,
  stepOverTarget,
  stopsAt,
} from "./stepping";

const op = (index: number) => ({ id: `op${index}`, index }) as Operation;
const stage = (id: string, operationIds: string[]) =>
  ({ id, operationIds }) as JourneyStage;
// model: op0 · block(op1 · attention(op2 op3) · op4) · op5
const stages = [
  stage("model", ["op0", "op1", "op2", "op3", "op4", "op5"]),
  stage("block", ["op1", "op2", "op3", "op4"]),
  stage("attention", ["op2", "op3"]),
];
const steps: PlaybackStep[] = [0, 1, 2, 3, 4, 5].map((i) => ({
  id: `op${i}`,
  operation: op(i),
}));

describe("debugger stepping", () => {
  const depths = stepDepths(steps, stages);

  it("measures depth in recorded calls", () => {
    expect(depths).toEqual([1, 2, 3, 3, 2, 1]);
  });

  it("steps over a call to the next step at the same depth", () => {
    expect(stepOverTarget(depths, 1)).toBe(4); // past the attention call
    expect(stepOverTarget(depths, 0)).toBe(5); // past the whole block
    expect(stepOverTarget(depths, 2)).toBe(3);
    expect(stepOverTarget(depths, 5)).toBeNull();
    expect(stepOverTarget(depths, -1)).toBe(0);
  });

  it("steps out to the first step after the call around this one", () => {
    expect(stepOutTarget(depths, 2)).toBe(4);
    expect(stepOutTarget(depths, 4)).toBe(5);
    expect(stepOutTarget(depths, 5)).toBeNull();
  });

  it("puts a loop's repeats at its body's shallowest depth", () => {
    const fold = {
      id: "/0:9",
      iterations: [
        ["op1", "op2"],
        ["op3", "op4"],
      ],
    } as LoopFold;
    expect(stepDepths([{ id: "loop:/0:9", fold }], stages)).toEqual([2]);
  });

  it("stops at breakpoint steps, and at repeats that run one", () => {
    expect(stopsAt(steps[2], new Set(["op2"]))).toBe(true);
    expect(stopsAt(steps[2], new Set())).toBe(false);
    expect(stopsAt(steps[2], undefined)).toBe(false);
    const fold = {
      id: "/0:9",
      iterations: [["op1"], ["op3"]],
    } as LoopFold;
    const repeats: PlaybackStep = { id: "loop:/0:9", fold };
    expect(stopsAt(repeats, new Set(["op3"]))).toBe(true);
    // The first pass already played step by step.
    expect(stopsAt(repeats, new Set(["op1"]))).toBe(false);
  });
});

describe("folded playback", () => {
  const folded = foldedPlayback(steps, stages, new Set(["attention"]));
  it("plays a folded call as one step, where its operations were", () => {
    expect(folded.map((step) => step.id)).toEqual([
      "op0",
      "op1",
      "attention",
      "op4",
      "op5",
    ]);
  });
  it("plays the outermost fold when folds nest", () => {
    expect(
      foldedPlayback(steps, stages, new Set(["attention", "block"])).map(
        (step) => step.id,
      ),
    ).toEqual(["op0", "block", "op5"]);
  });
  it("steps over and out of a folded card at its call's level", () => {
    const depths = stepDepths(folded, stages);
    expect(depths).toEqual([1, 2, 2, 2, 1]);
    // From op1, step over lands on the card, then past it.
    expect(stepOverTarget(depths, 1)).toBe(2);
    expect(stepOverTarget(depths, 2)).toBe(3);
    // From the card, step out leaves the block.
    expect(stepOutTarget(depths, 2)).toBe(4);
  });
  it("stops on a folded card for a breakpoint inside it", () => {
    const card = folded[2];
    expect(stopsAt(card, new Set(["op3"]))).toBe(true);
    expect(stopInside(card, new Set(["op3"]))).toBe("op3");
    expect(stopsAt(card, new Set(["op4"]))).toBe(false);
  });
  it("leaves playback as it was with nothing folded", () => {
    expect(foldedPlayback(steps, stages, new Set())).toBe(steps);
  });
});
