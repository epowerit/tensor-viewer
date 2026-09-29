import { describe, expect, it } from "vitest";
import { blankProject } from "../builder/model";
import { draftSignature, toDraft } from "../api/client";
import {
  forwardCall,
  forwardInputs,
  forwardIssue,
  withForwardInputs,
} from "./forward";

describe("forward input configuration", () => {
  const draft = { ...blankProject("test"), blueprint: null };
  const first = forwardInputs(draft)[0];
  it("keeps legacy calls and signatures stable", () => {
    expect(forwardCall([first])).toBe("model(x)");
    expect(forwardIssue([first])).toBe("");
    expect(draftSignature(draft)).toBe(
      draftSignature({
        ...draft,
        input_name: "x",
        input_binding: "positional",
        additional_inputs: [],
      }),
    );
  });
  it("keeps ordered, named inputs through persistence and detects changed settings", () => {
    const inputs = [
      { ...first, name: "query" },
      { ...first, name: "key" },
      { ...first, name: "mask", binding: "keyword" as const },
    ];
    const next = withForwardInputs(draft, inputs);
    expect(forwardInputs(toDraft(next))).toEqual(inputs);
    expect(forwardCall(inputs)).toBe("model(query, key, mask=mask)");
    expect(forwardIssue(inputs)).toBe("");
    expect(draftSignature(next)).not.toBe(draftSignature(draft));
    expect(
      draftSignature(
        withForwardInputs(
          next,
          inputs.map((item, i) =>
            i === 2 ? { ...item, input: { ...item.input, seed: 100 } } : item,
          ),
        ),
      ),
    ).not.toBe(draftSignature(next));
    expect(forwardInputs(draft)).toEqual([first]);
  });
  it("rejects ambiguous, uncallable, and excessive inputs", () => {
    for (const name of [
      "",
      "has space",
      "class",
      "True",
      "a.b",
      "3rd",
      "x".repeat(101),
    ])
      expect(forwardIssue([{ ...first, name }])).not.toBe("");
    expect(forwardIssue([first, first])).toContain("already in use");
    expect(
      forwardIssue([
        { ...first, binding: "keyword" },
        { ...first, name: "y" },
      ]),
    ).toContain("positional");
    expect(
      forwardIssue(
        Array.from({ length: 9 }, (_, i) => ({ ...first, name: `x${i}` })),
      ),
    ).not.toBe("");
    expect(forwardIssue([])).not.toBe("");
  });
  it("validates every input's axes, seeds, and shared numeric budget", () => {
    const large = { ...first.input, shape: [8_000_000], axis_names: [] };
    const inputs = Array.from({ length: 5 }, (_, i) => ({
      ...first,
      name: `x${i}`,
      input: large,
    }));
    expect(forwardIssue(inputs)).toContain("32 million");
    expect(forwardIssue(inputs, "shapes")).toBe("");
    expect(
      forwardIssue([
        first,
        { ...first, name: "y", input: { ...first.input, seed: -1 } },
      ]),
    ).toContain("y:");
    expect(
      forwardIssue([
        first,
        { ...first, name: "y", input: { ...first.input, axis_names: ["one"] } },
      ]),
    ).toContain("axis");
  });
});
