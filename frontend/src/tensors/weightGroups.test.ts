import { expect, test } from "vitest";
import { groupWeights } from "./weightGroups";

test("weights follow the run, grouped by the layer that holds them", () => {
  const read: Record<string, number> = {
    "token.weight": 3,
    "position.weight": 4,
    "blocks.0.norm1.weight": 6,
    "blocks.0.norm1.bias": 6,
    "blocks.0.attention.qkv.weight": 7,
    "blocks.0.attention.qkv.bias": 7,
    "blocks.0.attention.output.weight": 25,
    "blocks.0.norm2.weight": 27,
    "blocks.0.feed_forward.0.weight": 28,
    "norm.weight": 58,
  };
  const weights = [...Object.keys(read), "unused.weight"]
    .reverse()
    .map((name) => ({ name }));
  const groups = groupWeights(weights, (weight) => read[weight.name] ?? null);
  expect(groups.map((group) => group.path)).toEqual([
    "",
    "blocks.0",
    "blocks.0.attention",
    "blocks.0",
    "blocks.0.feed_forward",
    "",
  ]);
  expect(groups[0].weights.map((each) => each.label)).toEqual([
    "token.weight",
    "position.weight",
  ]);
  // A weight before its bias, then the next weight the run reads.
  expect(groups[2].weights.map((each) => each.label)).toEqual([
    "qkv.weight",
    "qkv.bias",
    "output.weight",
  ]);
  // The final norm comes last, with any weight the run never read.
  expect(groups.at(-1)!.weights.map((each) => each.label)).toEqual([
    "norm.weight",
    "unused.weight",
  ]);
});
