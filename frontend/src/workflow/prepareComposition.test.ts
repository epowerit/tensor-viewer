import { describe, expect, it, vi } from "vitest";
import type { CompositionPlan } from "../api/client";
import { blankProject } from "../builder/model";
import { prepareComposition } from "./prepareComposition";

const draft = blankProject("Model");
const ready: CompositionPlan = {
  valid: true,
  validation_required: false,
  code: "checked generated source",
  stages: [],
};
const needsCheck: CompositionPlan = {
  ...ready,
  valid: false,
  validation_required: true,
  error: "Custom shapes need checking.",
};

describe("composition preparation for generation", () => {
  it("consults the backend and reuses a valid cached check without executing workers", async () => {
    const service = {
      compose: vi.fn().mockResolvedValue(ready),
      checkComposition: vi.fn(),
    };
    await expect(prepareComposition(draft, service)).resolves.toBe(ready);
    expect(service.compose).toHaveBeenCalledExactlyOnceWith(draft);
    expect(service.checkComposition).not.toHaveBeenCalled();
  });

  it("executes one shape check on a cache miss and returns its current result", async () => {
    let resolvePreview!: (plan: CompositionPlan) => void;
    const preview = new Promise<CompositionPlan>((resolve) => {
      resolvePreview = resolve;
    });
    const service = {
      compose: vi.fn().mockReturnValue(preview),
      checkComposition: vi.fn().mockResolvedValue(ready),
    };
    const prepared = prepareComposition(draft, service);
    expect(service.checkComposition).not.toHaveBeenCalled();
    resolvePreview(needsCheck);
    await expect(prepared).resolves.toBe(ready);
    expect(service.compose).toHaveBeenCalledExactlyOnceWith(draft);
    expect(service.checkComposition).toHaveBeenCalledExactlyOnceWith(draft);
  });

  it("returns structural errors without executing custom code", async () => {
    const invalid: CompositionPlan = {
      ...ready,
      valid: false,
      error: "Attention expects three dimensions.",
    };
    const service = {
      compose: vi.fn().mockResolvedValue(invalid),
      checkComposition: vi.fn(),
    };
    await expect(prepareComposition(draft, service)).resolves.toBe(invalid);
    expect(service.checkComposition).not.toHaveBeenCalled();
  });

  it("returns an executed check failure for review without retrying it", async () => {
    const failed: CompositionPlan = {
      ...ready,
      valid: false,
      error: "Custom module requires numeric values.",
    };
    const service = {
      compose: vi.fn().mockResolvedValue(needsCheck),
      checkComposition: vi.fn().mockResolvedValue(failed),
    };
    await expect(prepareComposition(draft, service)).resolves.toBe(failed);
    expect(service.checkComposition).toHaveBeenCalledTimes(1);
  });

  it("propagates a preview error without attempting an execution", async () => {
    const error = new Error("Backend unavailable");
    const service = {
      compose: vi.fn().mockRejectedValue(error),
      checkComposition: vi.fn(),
    };
    await expect(prepareComposition(draft, service)).rejects.toBe(error);
    expect(service.checkComposition).not.toHaveBeenCalled();
  });

  it("propagates a shape-check error without a silent retry", async () => {
    const error = new Error("Another run is in progress");
    const service = {
      compose: vi.fn().mockResolvedValue(needsCheck),
      checkComposition: vi.fn().mockRejectedValue(error),
    };
    await expect(prepareComposition(draft, service)).rejects.toBe(error);
    expect(service.compose).toHaveBeenCalledTimes(1);
    expect(service.checkComposition).toHaveBeenCalledTimes(1);
  });
});
