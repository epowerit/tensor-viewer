import { describe, expect, it } from "vitest";
import type { Run } from "../api/client";
import { knockedTensor, knockoutText } from "./knockoutText";

const trace = {
  operations: [
    { id: "op0", index: 0, kind: "mul", inputs: ["t0"], outputs: ["t1"] },
    { id: "op1", index: 1, kind: "softmax", inputs: ["t1"], outputs: ["t2"] },
  ],
  tensors: {
    t1: {
      id: "t1",
      name: "doubled",
      shape: [1, 3, 4],
      axes: ["axis 0", "axis 1", "axis 2"],
    },
    t2: {
      id: "t2",
      name: "weights",
      shape: [1, 2, 3, 3],
      axes: ["batch", "heads", "queries", "keys"],
    },
  },
} as unknown as Run["trace"];

describe("knockoutText", () => {
  it("reads a whole result set to zero as an assignment", () => {
    expect(knockoutText(trace, { step: 0, mode: "zero" })).toBe("doubled = 0");
  });

  it("indexes one slice the way Python would", () => {
    expect(
      knockoutText(trace, { step: 1, mode: "zero", axis: 3, index: 2 }),
    ).toBe("weights[:, :, :, 2] = 0");
    expect(
      knockoutText(trace, { step: 1, mode: "mean", axis: 1, index: 0 }),
    ).toBe("weights[:, 0] = mean over heads");
  });

  it("names a numbered axis by its number, and a whole mean plainly", () => {
    expect(
      knockoutText(trace, { step: 0, mode: "mean", axis: 1, index: 1 }),
    ).toBe("doubled[:, 1] = mean over axis 1");
    expect(knockoutText(trace, { step: 0, mode: "mean" })).toBe(
      "doubled = its mean",
    );
  });

  it("says where a patch comes from", () => {
    expect(
      knockoutText(trace, {
        step: 1,
        mode: "patch",
        axis: 2,
        index: 1,
        patch_from: "run",
      }),
    ).toBe("weights[:, :, 1] from the run before");
  });

  it("falls back to the step number past the recorded steps", () => {
    expect(knockedTensor(trace, { step: 7, mode: "zero" })).toBeNull();
    expect(knockoutText(trace, { step: 7, mode: "zero" })).toBe("step 8 = 0");
  });
});
