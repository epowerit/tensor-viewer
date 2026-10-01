import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";
import { consoleProject } from "../console/script";
import { isCodeAt } from "../editor/completions";

export type CodeEntry = "statements" | "module";

/** Suggestions only: selecting a class does not import or execute its source. */
export function sourceClasses(code: string): string[] {
  const names: string[] = [];
  for (const match of code.matchAll(
    /^(?:\uFEFF)?class[ \t]+([A-Za-z_]\w*)[ \t]*(?=[:(])/gm,
  ))
    if (isCodeAt(code, match.index)) names.push(match[1]);
  return [...new Set(names)];
}

export function codeImportIssue(
  code: string,
  entry: CodeEntry,
  className: string,
): string | null {
  if (!code.trim()) return "Paste code or choose a Python file.";
  const limit = entry === "module" ? 500_000 : 20_000;
  if (code.length > limit)
    return entry === "module"
      ? "Use a Python source file of at most 500,000 characters."
      : "Use at most 20,000 characters for tensor statements, or choose Model class.";
  if (entry === "module" && !/^[A-Za-z_]\w{0,99}$/.test(className))
    return "Enter the name of the nn.Module class to run.";
  return null;
}

/** Both entry styles produce the same ordinary project/run contract. */
export function projectFromCode(
  name: string,
  code: string,
  entry: CodeEntry,
  className: string,
): Draft {
  if (entry === "statements") return { ...consoleProject(name), script: code };
  return {
    ...blankProject(name),
    blueprint: null,
    script: null,
    code,
    class_name: className,
    constructor: {},
  };
}
