import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { runWords, wordChanges } from "./wordChanges";

const run = (text: string, edits: { index: number; value: number }[] = []) =>
  ({
    project: { input: { text, edits }, additional_inputs: [] },
  }) as unknown as Run;

test("a run's words carry its what-if edits", () => {
  // Vocabulary ids: cat 0, mat 1, on 2, sat 3, the 4.
  expect(
    runWords(run("the cat sat on the mat", [{ index: 1, value: 1 }])),
  ).toEqual(["the", "mat", "sat", "on", "the", "mat"]);
  expect(runWords(run(""))).toBeNull();
});

test("two runs' words are compared position by position", () => {
  const before = run("the cat sat on the mat");
  expect(
    wordChanges(
      run("the cat sat on the mat", [{ index: 1, value: 1 }]),
      before,
    ),
  ).toEqual([{ at: 1, from: "cat", to: "mat" }]);
  expect(wordChanges(run("the dog sat on the mat"), before)).toEqual([
    { at: 1, from: "cat", to: "dog" },
  ]);
  expect(wordChanges(before, before)).toEqual([]);
  expect(
    wordChanges(run("a longer sentence than the other one"), before),
  ).toBeNull();
});
