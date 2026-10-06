import type { Run } from "../api/client";
import { tokenize } from "../inputs/samples";
import { lensWords } from "./LogitLensPanel";

/** The words a run read: its first text input's, with a what-if's edits. */
export function runWords(run: Run): string[] | null {
  const inputs = [
    run.project.input,
    ...(run.project.additional_inputs ?? []).map((each) => each.input),
  ];
  const text = inputs.find((input) => input.text)?.text;
  if (!text) return null;
  return lensWords(run, tokenize(text).tokens.length)?.tokens ?? null;
}

/**
 * Where two runs' words differ, position by position: what this run read
 * against what the other did. Null when either read no sentence or they read
 * sentences of different lengths, which cannot be lined up word for word.
 */
export function wordChanges(
  run: Run,
  other: Run,
): { at: number; from: string; to: string }[] | null {
  const now = runWords(run);
  const before = runWords(other);
  if (!now || !before || now.length !== before.length) return null;
  return now.flatMap((word, at) =>
    word === before[at] ? [] : [{ at, from: before[at], to: word }],
  );
}
