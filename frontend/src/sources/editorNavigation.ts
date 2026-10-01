import type { Draft, Run } from "../api/client";
import { entryPath, projectFiles } from "./files";

export type EditorTarget = { file: string; line: number | null };
export type EditorNavigation = EditorTarget & { request: number };

/** A recorded line is meaningful only while that source file is unchanged. */
export function runErrorTarget(draft: Draft, run: Run): EditorTarget | null {
  const error = run.trace.error;
  if (!error) return null;
  const file = error.file ?? entryPath(run.project);
  const current = projectFiles(draft);
  const recorded = projectFiles(run.project);
  if (!Object.hasOwn(current, file) || !Object.hasOwn(recorded, file))
    return null;
  const line =
    current[file] === recorded[file] && lineSelection(current[file], error.line)
      ? error.line!
      : null;
  return { file, line };
}

export function lineSelection(
  code: string,
  line?: number | null,
): { start: number; end: number } | null {
  if (!Number.isInteger(line) || !line || line < 1) return null;
  const lines = code.split("\n");
  if (line > lines.length) return null;
  const start = lines
    .slice(0, line - 1)
    .reduce((offset, text) => offset + text.length + 1, 0);
  return { start, end: start + lines[line - 1].replace(/\r$/, "").length };
}
