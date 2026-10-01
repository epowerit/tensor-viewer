import { describe, expect, it } from "vitest";
import { tensorSelection } from "./selection";

const tensors = [
  { id: "values", numel: 4 },
  { id: "state", numel: 32 },
  { id: "empty", numel: 0 },
];
describe("requested operation tensor", () => {
  it("validates a linked cell against the selected output, not output zero", () => {
    expect(tensorSelection(tensors, "state", 31)).toEqual({
      choice: 1,
      index: 31,
    });
    expect(tensorSelection(tensors, "values", 31)).toEqual({
      choice: 0,
      index: 0,
    });
  });
  it("keeps default and obsolete links on a valid first output cell", () => {
    expect(tensorSelection(tensors, undefined, 3)).toEqual({
      choice: 0,
      index: 3,
    });
    expect(tensorSelection(tensors, "old", 3)).toEqual({ choice: 0, index: 3 });
    for (const cell of [-1, 4, 1.5, Infinity, NaN])
      expect(tensorSelection(tensors, "values", cell).index).toBe(0);
  });
  it("retains a requested empty tensor without inventing a cell", () => {
    expect(tensorSelection(tensors, "empty", 0)).toEqual({
      choice: 2,
      index: 0,
    });
    expect(tensorSelection([], "missing", 5)).toEqual({ choice: 0, index: 0 });
  });
});
