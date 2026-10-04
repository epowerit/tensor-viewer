import { expect, test } from "vitest";
import { attentionChange, attentionRows } from "./AttentionLines";

test("weights are drawn as they are; scores as softmax would weigh them", () => {
  const weights = [
    [1, 0],
    [0.25, 0.75],
  ];
  expect(attentionRows(weights)).toEqual({ rows: weights, normalized: false });
  // Causal scores: a masked −∞ draws nothing, equal scores share evenly.
  const scores = attentionRows([
    [2, -Infinity],
    [0, 0],
  ]);
  expect(scores.normalized).toBe(true);
  expect(scores.rows).toEqual([
    [1, 0],
    [0.5, 0.5],
  ]);
  // A row masked entirely has no weights at all.
  expect(attentionRows([[-Infinity, -Infinity]]).rows).toEqual([[0, 0]]);
});

test("the change between two states of attention, and its largest gain", () => {
  const found = attentionChange(
    [
      [1, 0],
      [0.2, 0.8],
    ],
    [
      [1, 0],
      [0.6, 0.4],
    ],
  );
  expect(found.deltas[1][1]).toBeCloseTo(0.4);
  expect(found.deltas[1][0]).toBeCloseTo(-0.4);
  expect(found.largest).toBeCloseTo(0.4);
  expect(found.gain).toMatchObject({ q: 1, k: 1 });
});
