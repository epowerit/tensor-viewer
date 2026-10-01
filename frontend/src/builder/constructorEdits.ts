import type { ComponentSpec } from "../api/client";
import { parseArguments } from "./custom";

export type ConstructorEdit = {
  sourceId: string;
  applied: string;
  text: string;
  error: string;
};
export type ConstructorEdits = Record<string, ConstructorEdit>;

function appliedArguments(component: ComponentSpec) {
  return component.arguments ?? component.custom?.constructor ?? {};
}

export function constructorEdit(component: ComponentSpec, text: string) {
  let value: Record<string, unknown> | undefined;
  let error = "";
  try {
    value = parseArguments(text);
  } catch (failure) {
    error = (failure as Error).message;
  }
  return {
    value,
    edit: {
      sourceId: component.custom!.id,
      applied: JSON.stringify(value ?? appliedArguments(component)),
      text,
      error,
    },
  };
}

/** An edit belongs to one node, one source version, and its latest applied arguments. */
export function currentConstructorEdit(
  edits: ConstructorEdits,
  component: ComponentSpec,
) {
  const edit = edits[component.id];
  return component.custom &&
    edit?.sourceId === component.custom.id &&
    edit.applied === JSON.stringify(appliedArguments(component))
    ? edit
    : undefined;
}

/** Removed or replaced nodes must not leave the whole project blocked. */
export function reconcileConstructorEdits(
  edits: ConstructorEdits,
  components: ComponentSpec[],
): ConstructorEdits {
  const current = Object.fromEntries(
    components.flatMap((component) => {
      const edit = currentConstructorEdit(edits, component);
      return edit ? [[component.id, edit]] : [];
    }),
  );
  return Object.keys(current).length === Object.keys(edits).length
    ? edits
    : current;
}
