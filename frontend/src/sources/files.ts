import type { Draft, SourceImport } from "../api/client";
import { blankProject } from "../builder/model";

export const entryPath = (draft: Draft) => draft.entry_path ?? "model.py";
export const projectFiles = (draft: Draft): Record<string, string> => ({
  ...draft.files,
  [entryPath(draft)]: draft.code,
});
export const sourceCode = (draft: Draft, file?: string | null) =>
  projectFiles(draft)[file ?? entryPath(draft)] ?? "";
export function updateFile(draft: Draft, path: string, code: string): Draft {
  return path === entryPath(draft)
    ? { ...draft, code }
    : { ...draft, files: { ...draft.files, [path]: code } };
}
export function chooseEntry(draft: Draft, path: string): Draft {
  const files = projectFiles(draft);
  if (!path.endsWith(".py") || files[path] === undefined) return draft;
  const code = files[path];
  delete files[path];
  return { ...draft, code, files, entry_path: path };
}
export function importedProject(
  name: string,
  source: SourceImport,
  entry: string,
  className: string,
  root: string,
): Draft {
  const files = { ...source.files };
  const code = files[entry];
  delete files[entry];
  return {
    ...blankProject(name),
    blueprint: null,
    code,
    files,
    entry_path: entry,
    class_name: className,
    import_root: root,
    repository: source.repository,
    constructor: {},
  };
}
export function pathIssue(path: string): string | null {
  if (
    !path ||
    path.length > 240 ||
    /[\\\x00-\x1f]/.test(path) ||
    path.split("/").some((p) => ["", ".", "..", ".git"].includes(p))
  )
    return "Use a relative path, such as layers/attention.py.";
  if (!/\.(py|json|ya?ml|toml|txt|cfg|ini)$/i.test(path))
    return "Use a Python or text configuration file.";
  return null;
}
