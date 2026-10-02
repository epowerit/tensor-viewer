import type { Draft } from "../api/client";
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
  // A console project's code is generated from its script.
  if (before.script != null || after.script != null) {
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
