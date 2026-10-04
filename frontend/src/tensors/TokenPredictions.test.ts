import { expect, test } from "vitest";
import type { Tensor } from "../api/client";
import { predictions } from "./TokenPredictions";

const tensor = (shape: number[], values: number[]) =>
  ({ shape, numel: values.length, values }) as unknown as Tensor;
const axes = {
  positions: 1,
  vocabulary: 2,
  words: ["the", "cat"],
  ids: ["cat", "sat", "the"],
};

test("logits go through a softmax; each word gets its most likely next entry", () => {
  const logits = tensor([1, 2, 3], [0, 0, 5, 3, 1, 0]);
  const found = predictions(logits, [0, 0, 0], axes)!;
  expect(found.map((item) => [item.word, item.top[0].word])).toEqual([
    ["the", "the"],
    ["cat", "cat"],
  ]);
  expect(found[0].top[0].probability).toBeGreaterThan(0.98);
  // The cell to select for "cat → cat": position 1, id 0.
  expect(found[1].index).toBe(3);
});

test("probabilities are read as they are", () => {
  const probabilities = tensor([1, 2, 3], [0.2, 0.5, 0.3, 0.6, 0.3, 0.1]);
  const found = predictions(probabilities, [0, 0, 0], axes)!;
  expect(found[0].top.map((item) => [item.word, item.probability])).toEqual([
    ["sat", 0.5],
    ["the", 0.3],
    ["cat", 0.2],
  ]);
  // Values that stay in the backend are not read here.
  const paged = { shape: [1, 2, 3], numel: 6, values: [] } as unknown as Tensor;
  expect(predictions(paged, [0, 0, 0], axes)).toBeNull();
});
