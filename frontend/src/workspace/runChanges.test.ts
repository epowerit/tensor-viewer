import { expect, test } from "vitest";
import type { Draft } from "../api/client";
import { changedLines, draftChanges } from "./runChanges";

const input = (shape: number[]) =>
  ({
    shape,
    generator: "random",
    dtype: "float32",
    axis_names: [],
  }) as unknown as Draft["input"];
const draft = (patch: Record<string, unknown> = {}): Draft =>
  ({
    name: "p",
    code: "a\nb\nc\nd",
    class_name: "Model",
    constructor: {},
    input: input([2, 3]),
    input_name: "x",
    additional_inputs: [
      { name: "mask", binding: "keyword", input: input([2]) },
    ],
    capture_mode: "values",
    files: {},
    ...patch,
  }) as unknown as Draft;

test("changed lines count the differing middle", () => {
  expect(changedLines("a\nb\nc", "a\nb\nc")).toBe(0);
  expect(changedLines("a\nb\nc", "a\nX\nc")).toBe(1);
  expect(changedLines("a\nb\nc", "a\nX\nY\nc")).toBe(2);
  expect(changedLines("a\nb\nc", "a\nc")).toBe(1);
});

test("a run says what changed since the one before", () => {
  expect(draftChanges(draft(), draft())).toEqual([]);
  expect(
    draftChanges(
      draft(),
      draft({
        code: "a\nB\nc\nd",
        input: input([4, 3]),
        capture_mode: "shapes",
        files: { "util.py": "x" },
      }),
    ),
  ).toEqual([
    "model.py · 1 line",
    "added util.py",
    "input x",
    "values → shapes",
  ]);
  expect(draftChanges(draft(), draft({ additional_inputs: [] }))).toEqual([
    "removed input mask",
  ]);
});
