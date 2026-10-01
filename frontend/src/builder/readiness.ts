import type { CompositionPlan, Draft } from "../api/client";

export type BuildReadiness = {
  signature: string;
  state: "checking" | "needs-check" | "invalid" | "ready";
  issue: string;
  componentId?: string;
};

export type CheckedPlan = { signature: string; plan: CompositionPlan };
export type CompositionResult = {
  signature: string;
  plan: CompositionPlan | null;
  error?: string;
};

export function compositionSignature(draft: Draft): string {
  return JSON.stringify([
    draft.blueprint,
    draft.input,
    draft.capture_mode,
    draft.name,
  ]);
}

/** A lightweight preview must never replace a newer, executed shape check. */
export function canApplyCompositionPreview(
  requestSignature: string,
  currentSignature: string,
  checkedSignature?: string,
): boolean {
  return (
    requestSignature === currentSignature &&
    checkedSignature !== requestSignature
  );
}

export function buildReadiness(
  draft: Draft,
  result: CompositionResult | null,
  options: {
    checking?: boolean;
    editingIssue?: { issue: string; componentId: string };
  } = {},
): BuildReadiness {
  const signature = compositionSignature(draft);
  const invalid = (issue: string, componentId?: string): BuildReadiness => ({
    signature,
    state: "invalid",
    issue,
    ...(componentId ? { componentId } : {}),
  });
  if (options.editingIssue)
    return invalid(
      options.editingIssue.issue,
      options.editingIssue.componentId,
    );
  if (!draft.name.trim()) return invalid("Give your project a name.");
  if (!draft.blueprint?.has_input)
    return invalid("Add an input to your model.", "input");
  if (!draft.blueprint.components.length)
    return invalid("Add a component from the toolbar.");
  if (options.checking || !result || result.signature !== signature)
    return {
      signature,
      state: "checking",
      issue: "Checking model connections…",
    };
  if (result.error) return invalid(result.error);
  const plan = result.plan;
  if (!plan) return invalid("The model could not be checked. Try again.");
  const failedStage = plan.stages.find((stage) => stage.error);
  if (plan.validation_required)
    return {
      signature,
      state: "needs-check",
      issue: "Custom shapes will be checked when you generate the diagram.",
      ...(failedStage ? { componentId: failedStage.id } : {}),
    };
  if (!plan.valid || failedStage || plan.error)
    return invalid(
      failedStage?.error || plan.error || "Review your component connections.",
      failedStage?.id,
    );
  return { signature, state: "ready", issue: "" };
}
