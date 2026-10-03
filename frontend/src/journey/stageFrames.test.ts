import { expect, test } from "vitest";
import type { JourneyNode } from "./graph";
import { NODE_HEIGHT, NODE_WIDTH } from "./graph";
import { frameStages } from "./stageFrames";
import type { JourneyStage } from "./stages";

const node = (id: string, x: number, y = 0, stage?: JourneyStage) =>
  ({ id, x, y, stage, tensors: [] }) as unknown as JourneyNode;
const stage = (id: string, ops: string[], parent: string | null = null) =>
  ({ id, operationIds: ops, parentStageId: parent }) as JourneyStage;

test("an open call is framed around its cards, wider around nested frames", () => {
  const block = stage("block", ["a", "b", "c", "d"]);
  const attention = stage("attention", ["b", "c"], "block");
  const nodes = [node("a", 0), node("b", 300), node("c", 600), node("d", 900)];
  const frames = frameStages([block, attention], nodes);
  const outer = frames.find((frame) => frame.stage.id === "block")!;
  const inner = frames.find((frame) => frame.stage.id === "attention")!;
  expect(inner).toMatchObject({ left: 300 - 14, right: 600 + NODE_WIDTH + 14 });
  expect(outer).toMatchObject({
    left: -26,
    top: -26,
    bottom: NODE_HEIGHT + 26,
  });
});

test("a folded stage inside counts as one card; one card gets no frame", () => {
  const capsule = stage("capsule", ["b", "c"], "block");
  const block = stage("block", ["a", "b", "c"]);
  const nodes = [node("a", 0), node("capsule", 300, 0, capsule)];
  expect(frameStages([block], nodes).map((f) => f.stage.id)).toEqual(["block"]);
  expect(frameStages([stage("lone", ["a"])], nodes)).toEqual([]);
});

test("a step drawn as its scene brings its operands into the call's frame", () => {
  const call = stage("call", ["a", "b"]);
  // The scene draws b's operands apart, below the chain, as their own cards.
  const nodes = [node("a", 0), node("b", 300), node("weight", 0, 400)];
  const plain = frameStages([call], nodes)[0];
  const scene = frameStages([call], nodes, {
    operationId: "b",
    nodeIds: new Set(["b", "weight"]),
  })[0];
  expect(plain.bottom).toBe(NODE_HEIGHT + 14);
  expect(scene.bottom).toBe(400 + NODE_HEIGHT + 14);
});
