import { draftSignature, toDraft, type Draft } from "../api/client";
import { entryPath } from "../sources/files";

/** Inlays can reuse a line only while its inputs and every other file still match. */
export function executionContextSignature(
  project: Draft,
  displayedFile: string,
): string {
  const context = toDraft(project);
  context.name = "";
  context.repository = null;
  // The editor checks the unchanged prefix of the displayed source separately.
  if (context.script != null) context.script = "";
  else if (displayedFile === entryPath(context)) context.code = "";
  else context.files = { ...context.files, [displayedFile]: "" };
  return draftSignature(context);
}
