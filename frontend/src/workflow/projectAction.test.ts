import { describe, expect, it } from "vitest";
import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";
import {
  compositionSignature,
  type BuildReadiness,
} from "../builder/readiness";
import { projectAction } from "./projectAction";

function model(): Draft {
  return {
    ...blankProject("Example"),
    blueprint: {
      has_input: true,
      components: [{ id: "relu", kind: "relu", parameters: {} }],
    },
  };
}

const source = (): Draft => ({ ...model(), blueprint: null });
const ready = (draft: Draft): BuildReadiness => ({
  signature: compositionSignature(draft),
  state: "ready",
  issue: "",
});
const action = (
  draft: Draft | null,
  overrides: Partial<Parameters<typeof projectAction>[0]> = {},
) =>
  projectAction({
    draft,
    hasRun: false,
    editorValid: true,
    inputsValid: true,
    builderEditingValid: true,
    build: draft?.blueprint ? ready(draft) : null,
    ...overrides,
  });

describe("project next action", () => {
  it("guides an empty manual project through input and component setup", () => {
    expect(action(null).kind).toBe("wait");
    const draft = blankProject("New model");
    expect(action(draft)).toMatchObject({
      kind: "inputs",
      label: "Set up input",
    });
    draft.blueprint!.has_input = true;
    expect(action(draft)).toMatchObject({
      kind: "tools",
      label: "Add component",
    });
  });

  it("routes unfinished local edits even when the stored draft is valid", () => {
    expect(action(source(), { inputsValid: false })).toMatchObject({
      kind: "inputs",
      label: "Review inputs",
    });
    expect(action(source(), { editorValid: false })).toMatchObject({
      kind: "code",
      label: "Review code",
    });
    expect(action(model(), { builderEditingValid: false })).toMatchObject({
      kind: "model",
      label: "Review model",
    });
  });

  it("checks stored input settings and bindings without relying on panel state", () => {
    const draft = source();
    draft.input = { ...draft.input, shape: [0] };
    expect(action(draft).kind).toBe("inputs");
    const duplicated = source();
    duplicated.additional_inputs = [
      { name: "x", binding: "positional", input: { ...duplicated.input } },
    ];
    expect(action(duplicated)).toMatchObject({
      kind: "inputs",
      detail: "The input name “x” is already in use.",
    });
  });

  it("routes a missing name to the appropriate authoring surface", () => {
    for (const [draft, kind] of [
      [model(), "model"],
      [source(), "code"],
    ] as const) {
      draft.name = "  ";
      expect(action(draft)).toMatchObject({ kind, label: "Set project name" });
    }
  });

  it("requires source code and a usable class name without guessing class definitions", () => {
    expect(action({ ...source(), code: " \n" }).kind).toBe("code");
    for (const class_name of [
      "",
      "Model()",
      "class",
      "async",
      "1Model",
      "x".repeat(101),
    ])
      expect(action({ ...source(), class_name }).kind).toBe("code");
    expect(
      action({
        ...source(),
        code: "from my_package import DerivedModel",
        class_name: "DerivedModel",
      }).kind,
    ).toBe("run");
  });

  it("waits for current composition results instead of using an older valid plan", () => {
    const draft = model();
    const previous = ready(draft);
    draft.input = { ...draft.input, shape: [4, 4, 8] };
    expect(action(draft, { build: previous })).toMatchObject({
      kind: "wait",
      label: "Checking model…",
    });
    expect(action(draft, { build: null }).kind).toBe("wait");
    expect(
      action(draft, { build: { ...ready(draft), state: "checking" } }).kind,
    ).toBe("wait");
  });

  it("keeps generation as the next action when custom checks are needed", () => {
    const draft = model();
    const build: BuildReadiness = {
      ...ready(draft),
      state: "needs-check",
      issue: "Custom shapes need checking.",
    };
    expect(action(draft, { build })).toMatchObject({
      kind: "run",
      label: "Generate diagram",
    });
    expect(action(draft, { build }).detail).toContain(
      "Check custom component shapes",
    );
    expect(action(draft, { build, hasRun: true }).label).toBe("Run again");
  });

  it("returns the actual composition issue as a model review action", () => {
    const draft = model();
    expect(
      action(draft, {
        build: {
          ...ready(draft),
          state: "invalid",
          issue: "Attention expects three dimensions.",
          componentId: "attention",
        },
      }),
    ).toMatchObject({
      kind: "model",
      label: "Review model",
      detail: "Attention expects three dimensions.",
    });
  });

  it("offers exactly the same ready action for manual and source projects", () => {
    expect(action(model())).toEqual(action(source()));
    expect(action(model(), { hasRun: true })).toEqual(
      action(source(), { hasRun: true }),
    );
    expect(action(source(), { hasRun: true }).label).toBe("Run again");
  });
});
