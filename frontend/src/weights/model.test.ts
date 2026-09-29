import { describe, expect, it } from "vitest";
import { draftSignature, toDraft, type SavedWeights } from "../api/client";
import { blankProject } from "../builder/model";
import { checkpointFileIssue } from "./model";

const weights: SavedWeights = {
  id: "a".repeat(32),
  name: "Linear",
  file_name: "linear.pt",
  created_at: "2026-09-29",
  sha256: "b".repeat(64),
  byte_count: 1024,
  tensors: [{ name: "linear.weight", shape: [2, 3], dtype: "float32" }],
};
describe("checkpoint configuration", () => {
  it("keeps old projects unchanged and retains a checkpoint across saving", () => {
    const draft = blankProject("test");
    expect(draftSignature(draft)).toBe(
      draftSignature({ ...draft, weights: null }),
    );
    expect(toDraft({ ...draft, weights }).weights).toEqual(weights);
    expect(draftSignature({ ...draft, weights })).not.toBe(
      draftSignature(draft),
    );
    const custom = { ...draft, blueprint: null, weights };
    expect(draftSignature(custom)).not.toBe(
      draftSignature({ ...custom, weights: null }),
    );
    expect(draftSignature(custom)).not.toBe(
      draftSignature({
        ...custom,
        weights: { ...weights, sha256: "c".repeat(64) },
      }),
    );
  });
  it("rejects unsupported containers, empty files, and oversized uploads", () => {
    expect(checkpointFileIssue({ name: "model.pth", size: 1024 })).toBe("");
    expect(
      checkpointFileIssue({ name: "MODEL.PT", size: 64 * 1024 * 1024 }),
    ).toBe("");
    for (const file of [
      { name: "model.pkl", size: 12 },
      { name: "model.pt", size: 0 },
      { name: "model.pt", size: 64 * 1024 * 1024 + 1 },
      { name: "x".repeat(201) + ".pt", size: 1 },
    ])
      expect(checkpointFileIssue(file)).not.toBe("");
  });
});
