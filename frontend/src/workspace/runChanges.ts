import type { ComponentSpec, Draft } from "../api/client";
import { projectFiles } from "../sources/files";

/** How many lines differ, counting the changed middle once its ends match. */
export function changedLines(before: string, after: string): number {
  if (before === after) return 0;
  const a = before.split("\n"),
    b = after.split("\n");
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start++;
  let end = 0;
  while (
    end < a.length - start &&
    end < b.length - start &&
    a[a.length - 1 - end] === b[b.length - 1 - end]
  )
    end++;
  return Math.max(a.length - start - end, b.length - start - end);
}

const plural = (count: number, word: string) =>
  `${count} ${word}${count === 1 ? "" : "s"}`;

/**
 * What a newer run changed relative to an older one, in a few words each:
 * "model.py · 2 lines", "input tokens", "values → shapes". Empty when the
 * code and inputs are the same (a re-run).
 */
export function draftChanges(before: Draft, after: Draft): string[] {
  const changes: string[] = [];
  // A canvas project's code is generated from its components, so it says
  // which components changed: "transformer · expansion".
  if (before.blueprint || after.blueprint) {
    changes.push(
      ...componentChanges(
        before.blueprint?.components ?? [],
        after.blueprint?.components ?? [],
      ),
    );
  } else if (before.script != null || after.script != null) {
    // A console project's code is generated from its script.
    const lines = changedLines(before.script ?? "", after.script ?? "");
    if (lines) changes.push(`script · ${plural(lines, "line")}`);
  } else {
    const was = projectFiles(before),
      now = projectFiles(after);
    for (const path of new Set([...Object.keys(was), ...Object.keys(now)])) {
      if (!(path in was)) changes.push(`added ${path}`);
      else if (!(path in now)) changes.push(`removed ${path}`);
      else {
        const lines = changedLines(was[path], now[path]);
        if (lines) changes.push(`${path} · ${plural(lines, "line")}`);
      }
    }
  }
  if (before.class_name !== after.class_name)
    changes.push(`class ${after.class_name}`);
  if (
    JSON.stringify(before.constructor ?? {}) !==
    JSON.stringify(after.constructor ?? {})
  )
    changes.push("constructor arguments");
  const inputs = (draft: Draft) =>
    new Map([
      [draft.input_name ?? "x", draft.input],
      ...(draft.additional_inputs ?? []).map(
        (item) => [item.name, item.input] as const,
      ),
    ]);
  const wasInputs = inputs(before),
    nowInputs = inputs(after);
  const inputChanges = [...new Set([...wasInputs.keys(), ...nowInputs.keys()])]
    .filter(
      (name) =>
        JSON.stringify(wasInputs.get(name) ?? null) !==
        JSON.stringify(nowInputs.get(name) ?? null),
    )
    .map((name) =>
      !wasInputs.has(name)
        ? `new input ${name}`
        : !nowInputs.has(name)
          ? `removed input ${name}`
          : `input ${name}`,
    );
  changes.push(...inputChanges);
  if (
    JSON.stringify(before.weights ?? null) !==
    JSON.stringify(after.weights ?? null)
  )
    changes.push(after.weights ? "weights loaded" : "weights removed");
  const mode = (draft: Draft) => draft.capture_mode ?? "values";
  if (mode(before) !== mode(after))
    changes.push(`${mode(before)} → ${mode(after)}`);
  return changes;
}

/** A component by what it is: "transformer", "layer norm", or its own name. */
const componentName = (component: ComponentSpec) =>
  component.custom?.name ?? component.kind.replace(/_/g, " ");

/** Components added, removed, reordered, or with different settings. */
export function componentChanges(
  before: ComponentSpec[],
  after: ComponentSpec[],
): string[] {
  const was = new Map(before.map((component) => [component.id, component]));
  const now = new Map(after.map((component) => [component.id, component]));
  const changes: string[] = [];
  for (const component of after)
    if (!was.has(component.id))
      changes.push(`added ${componentName(component)}`);
  for (const component of before)
    if (!now.has(component.id))
      changes.push(`removed ${componentName(component)}`);
  for (const component of after) {
    const old = was.get(component.id);
    if (!old) continue;
    const keys = new Set([
      ...Object.keys(old.parameters ?? {}),
      ...Object.keys(component.parameters ?? {}),
    ]);
    const settings = [...keys].filter(
      (key) => old.parameters?.[key] !== component.parameters?.[key],
    );
    if (
      JSON.stringify(old.arguments ?? null) !==
        JSON.stringify(component.arguments ?? null) ||
      JSON.stringify(old.custom ?? null) !==
        JSON.stringify(component.custom ?? null)
    )
      settings.push("arguments");
    if (
      JSON.stringify(old.sources ?? null) !==
      JSON.stringify(component.sources ?? null)
    )
      settings.push("connections");
    if (settings.length)
      changes.push(`${componentName(component)} · ${settings.join(", ")}`);
  }
  const order = (list: ComponentSpec[], keep: Map<string, unknown>) =>
    list
      .filter((component) => keep.has(component.id))
      .map((c) => c.id)
      .join();
  if (order(before, now) !== order(after, was))
    changes.push("components reordered");
  return changes;
}
