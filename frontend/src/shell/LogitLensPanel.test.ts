import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { lensWords } from "./LogitLensPanel";

const run = (input: object, additional: object[] = []) =>
  ({
    project: { input_name: "tokens", input, additional_inputs: additional },
  }) as unknown as Run;

test("columns read the sentence whose length matches the positions", () => {
  const found = lensWords(run({ text: "the cat sat" }), 3);
  expect(found?.tokens).toEqual(["the", "cat", "sat"]);
  expect(found?.vocabulary).toEqual(["cat", "sat", "the"]);
  // A caption fed beside an image is found among the other inputs.
  const caption = lensWords(
    run({ text: null }, [{ name: "caption", input: { text: "a red bird" } }]),
    3,
  );
  expect(caption?.tokens).toEqual(["a", "red", "bird"]);
  expect(lensWords(run({ text: "too short" }), 5)).toBeNull();
});

test("a what-if's changed words head their columns", () => {
  const found = lensWords(
    run({ text: "the cat sat", edits: [{ index: 1, value: 1 }] }),
    3,
  );
  expect(found?.tokens).toEqual(["the", "sat", "sat"]);
});
