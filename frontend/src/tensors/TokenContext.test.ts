import { expect, test } from "vitest";
import type { Run, Tensor } from "../api/client";
import type { AxisStory } from "./axisLineage";
import { shortWord, tokenAxes } from "./TokenContext";

const tensor = (id: string, name: string, shape: number[], axes: string[]) =>
  ({ id, name, shape, axes }) as unknown as Tensor;

const story = (label: string, size: number): AxisStory => ({
  size,
  terms: [{ label, size }],
  note: null,
});

test("axes from a sentence's positions read as its words; vocabulary axes as its vocabulary", () => {
  const tokens = tensor("t0", "tokens", [1, 4], ["batch", "tokens"]);
  const weights = tensor(
    "t1",
    "weights",
    [1, 4, 4],
    ["batch", "queries", "keys"],
  );
  const logits = tensor(
    "t2",
    "logits",
    [1, 4, 6],
    ["batch", "tokens", "vocabulary"],
  );
  const run = {
    project: {
      input_name: "tokens",
      input: { text: "The cat saw the" },
      additional_inputs: [],
    },
    trace: {
      input_ids: ["t0"],
      tensors: { t0: tokens, t1: weights, t2: logits },
    },
  } as unknown as Run;
  const lineage = (id: string) =>
    ({
      t0: [story("tokens.batch", 1), story("tokens.tokens", 4)],
      t1: [
        story("tokens.batch", 1),
        story("tokens.tokens", 4),
        story("tokens.tokens", 4),
      ],
      t2: [
        story("tokens.batch", 1),
        story("tokens.tokens", 4),
        story("w axis 1", 6),
      ],
    })[id] ?? [];
  const words = tokenAxes(run, lineage)!;
  expect(words(tokens, 1)?.words).toEqual(["the", "cat", "saw", "the"]);
  // A what-if run that set the second token to "saw" (id 1) reads so.
  const edited = {
    ...run,
    project: {
      ...run.project,
      input: { ...run.project.input, edits: [{ index: 1, value: 1 }] },
    },
  } as Run;
  expect(tokenAxes(edited, lineage)!(tokens, 1)?.words).toEqual([
    "the",
    "saw",
    "saw",
    "the",
  ]);
  expect(words(weights, 2)?.words).toEqual(["the", "cat", "saw", "the"]);
  expect(words(weights, 0)).toBeNull();
  // The sentence's vocabulary is sorted; ids past it have no word.
  expect(words(logits, 2)).toEqual({
    words: ["cat", "saw", "the", "#3", "#4", "#5"],
    vocabulary: true,
  });
  expect(shortWord("attention")).toBe("atten…");
});
