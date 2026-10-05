import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { attentionMaps, headWeights } from "./attentionMaps";

const tensor = (id: string, shape: number[], axes: string[]) => ({
  id,
  name: id,
  shape,
  axes,
  numel: shape.reduce((a, b) => a * b, 1),
});

const trace = {
  operations: [
    {
      id: "op0",
      index: 0,
      kind: "softmax",
      outputs: ["w"],
      arguments: { dim: -1 },
      module: "GPT / blocks.0 / blocks.0.attention",
      status: "ok",
    },
    {
      id: "op1",
      index: 1,
      kind: "softmax",
      outputs: ["p"],
      arguments: { dim: -1 },
      module: "GPT",
      status: "ok",
    },
  ],
  tensors: {
    w: tensor("w", [1, 2, 3, 3], ["batch", "heads", "queries", "keys"]),
    // Next-word probabilities are a softmax too, but not attention.
    p: tensor("p", [1, 3, 64], ["batch", "tokens", "vocabulary"]),
  },
} as unknown as Run["trace"];

test("attention weights are found and the vocabulary softmax is not", () => {
  const maps = attentionMaps(trace);
  expect(
    maps.map((map) => [map.label, map.heads, map.queries, map.keys]),
  ).toEqual([["blocks.0.attention", 2, 3, 3]]);
});

test("one head's rows come out of the flat values", () => {
  const map = attentionMaps(trace)[0];
  const values = Array.from({ length: 18 }, (_, i) => i);
  expect(headWeights(values, map, 1)).toEqual([
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
  ]);
});
