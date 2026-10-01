import { api, type CompositionPlan, type Draft } from "../api/client";

/** Called only by an explicit generation action, never while editing. */
export async function prepareComposition(
  draft: Draft,
  service: Pick<typeof api, "compose" | "checkComposition"> = api,
): Promise<CompositionPlan> {
  const plan = await service.compose(draft);
  return plan.validation_required ? service.checkComposition(draft) : plan;
}
