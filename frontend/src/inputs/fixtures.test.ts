import { describe, expect, it } from "vitest";
import { blankProject } from "../builder/model";
import { fixtureSettings, inputIssue, uploadIssue } from "./fixtures";
import { draftSignature } from "../api/client";

describe("saved input settings", () => {
  it("pins file shape, dtype, and identity while copying editable labels", () => {
    const uploaded = {
      id: "a".repeat(32),
      file_name: "input.npy",
      sha256: "b".repeat(64),
      shape: [2, 4, 8],
      dtype: "float32" as const,
      byte_count: 384,
    };
    const input = {
      ...blankProject("test").input,
      generator: "uploaded" as const,
      uploaded,
    };
    expect(inputIssue(input)).toBe("");
    expect(inputIssue({ ...input, shape: [8, 4, 2] })).not.toBe("");
    expect(inputIssue({ ...input, dtype: "float64" })).not.toBe("");
    expect(inputIssue({ ...input, uploaded: null })).not.toBe("");
    expect(inputIssue({ ...input, generator: "zeros" })).not.toBe("");
    const copy = fixtureSettings({ input, capture_mode: "values" });
    copy.input.uploaded!.shape[0] = 100;
    expect(uploaded.shape[0]).toBe(2);
    const draft = blankProject("test");
    expect(draftSignature(draft)).toBe(
      draftSignature({ ...draft, input: { ...draft.input, uploaded: null } }),
    );
    expect(draftSignature({ ...draft, input })).not.toBe(draftSignature(draft));
  });
  it("bounds upload sizes and rejects unsupported file containers before sending", () => {
    expect(uploadIssue({ name: "sample.npy", size: 256 })).toBe("");
    for (const file of [
      { name: "sample.npz", size: 256 },
      { name: "sample.npy", size: 0 },
      { name: "sample.npy", size: 8_388_608 * 8 + 16_385 },
    ])
      expect(uploadIssue(file)).not.toBe("");
  });
  it("copies every setting without letting project edits mutate the library", () => {
    const fixture = {
      input: {
        ...blankProject("test").input,
        seed: 32,
        dtype: "float64" as const,
      },
      capture_mode: "shapes" as const,
    };
    const applied = fixtureSettings(fixture);
    expect(applied).toEqual(fixture);
    applied.input.shape[0] = 1024;
    applied.input.axis_names[0] = "changed";
    applied.input.seed = 55;
    expect(fixture.input.shape).toEqual([2, 4, 8]);
    expect(fixture.input.axis_names[0]).toBe("batch");
    expect(fixture.input.seed).toBe(32);
  });
  it("accepts large metadata inputs while enforcing value budgets and invalid drafts", () => {
    const base = blankProject("test").input;
    expect(inputIssue(base)).toBe("");
    const huge = { ...base, shape: [1024, 1024, 1024] };
    expect(inputIssue(huge, "shapes")).toBe("");
    expect(inputIssue(huge)).not.toBe("");
    for (const patch of [
      { shape: [] },
      { shape: [0, 3, 4] },
      { shape: [2.5, 3, 4] },
      { shape: [2 ** 41], axis_names: [] },
      { axis_names: ["one"] },
      { seed: -1 },
      { seed: NaN },
      { seed: 0.5 },
      { seed: 2 ** 32 },
      { generator: "random" as const, dtype: "int64" as const },
    ])
      expect(inputIssue({ ...base, ...patch }, "shapes")).not.toBe("");
  });
});
