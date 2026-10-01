import { describe, expect, it } from "vitest";
import type { CompositionPlan, Draft } from "../api/client";
import { blankProject } from "./model";
import {
  buildReadiness,
  canApplyCompositionPreview,
  compositionSignature,
} from "./readiness";

const draft: Draft = {
  ...blankProject("Model"),
  blueprint: {
    has_input: true,
    components: [{ id: "layer", kind: "linear", parameters: { features: 4 } }],
  },
};
const ready: CompositionPlan = {
  valid: true,
  validation_required: false,
  code: "generated",
  stages: [
    { id: "layer", title: "Linear", input_shape: [2, 4, 8], shape: [2, 4, 4] },
  ],
};
const result = { signature: compositionSignature(draft), plan: ready };

describe("builder readiness", () => {
  it("identifies missing setup without waiting for a backend preview", () => {
    expect(buildReadiness(blankProject("Model"), null)).toMatchObject({
      state: "invalid",
      componentId: "input",
    });
    expect(
      buildReadiness(
        { ...draft, blueprint: { has_input: true, components: [] } },
        null,
      ),
    ).toMatchObject({
      state: "invalid",
      issue: "Add a component from the toolbar.",
    });
    expect(buildReadiness({ ...draft, name: " " }, null)).toMatchObject({
      state: "invalid",
      issue: "Give your project a name.",
    });
  });

  it("never marks a changed model ready using its previous preview", () => {
    const edited = {
      ...draft,
      input: { ...draft.input, shape: [32, 16, 768] },
    };
    expect(buildReadiness(edited, result)).toMatchObject({
      state: "checking",
      signature: compositionSignature(edited),
    });
    expect(buildReadiness(draft, null).state).toBe("checking");
    expect(buildReadiness(draft, result, { checking: true }).state).toBe(
      "checking",
    );
    expect(buildReadiness(draft, result).state).toBe("ready");
  });

  it("allows Generate to perform the required custom shape check", () => {
    const plan: CompositionPlan = {
      ...ready,
      valid: false,
      validation_required: true,
      error: "Custom shape check needed",
      stages: [
        { ...ready.stages[0], error: "Custom shape check needed", shape: null },
      ],
    };
    expect(buildReadiness(draft, { ...result, plan })).toMatchObject({
      state: "needs-check",
      componentId: "layer",
    });
  });

  it("reports an executed check failure with the exact component to review", () => {
    const plan: CompositionPlan = {
      ...ready,
      valid: false,
      stages: [{ ...ready.stages[0], error: "Expected 8 features, got 12" }],
    };
    expect(buildReadiness(draft, { ...result, plan })).toEqual({
      signature: result.signature,
      state: "invalid",
      issue: "Expected 8 features, got 12",
      componentId: "layer",
    });
  });

  it("blocks partial constructor edits even when the applied model is valid", () => {
    expect(
      buildReadiness(draft, result, {
        editingIssue: { issue: "Fix constructor JSON", componentId: "layer" },
      }),
    ).toMatchObject({
      state: "invalid",
      componentId: "layer",
      issue: "Fix constructor JSON",
    });
  });

  it("distinguishes a current connection failure from an obsolete one", () => {
    const failure = { ...result, plan: null, error: "Backend unavailable" };
    expect(buildReadiness(draft, failure)).toMatchObject({
      state: "invalid",
      issue: "Backend unavailable",
    });
    expect(buildReadiness({ ...draft, name: "New model" }, failure).state).toBe(
      "checking",
    );
  });

  it("invalidates previews for composition changes but not regenerated source", () => {
    for (const changed of [
      { ...draft, name: "Changed" },
      { ...draft, capture_mode: "shapes" as const },
      { ...draft, input: { ...draft.input, dtype: "float64" as const } },
      {
        ...draft,
        blueprint: {
          ...draft.blueprint!,
          components: [{ id: "layer", kind: "relu", parameters: {} }],
        },
      },
    ])
      expect(compositionSignature(changed)).not.toBe(result.signature);
    expect(
      compositionSignature({ ...draft, code: "fresh generated source" }),
    ).toBe(result.signature);
  });

  it("ignores late previews after a different edit or a completed preflight check", () => {
    expect(canApplyCompositionPreview("A", "B")).toBe(false);
    expect(canApplyCompositionPreview("A", "A", "A")).toBe(false);
    expect(canApplyCompositionPreview("A", "A", "B")).toBe(true);
    expect(canApplyCompositionPreview("A", "A")).toBe(true);
  });
});
