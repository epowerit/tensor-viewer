import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";

// A suggestion only: aliases and inherited modules can be entered by the user.
export function suggestModuleClass(code: string): string {
  return (
    code.match(
      /^class\s+([A-Za-z_]\w*)\s*\([^)]*\b(?:nn\.Module|Module)\b[^)]*\)\s*:/m,
    )?.[1] ?? ""
  );
}

export async function readPythonFile(
  file: Pick<File, "name" | "size" | "text">,
): Promise<string> {
  if (!file.name.toLowerCase().endsWith(".py"))
    throw new Error("Choose a Python (.py) file.");
  if (file.size > 500_000)
    throw new Error("Choose a Python file no larger than 500 KB.");
  let code: string;
  try {
    code = await file.text();
  } catch {
    throw new Error("Could not read this file. Try selecting it again.");
  }
  if (!code.trim())
    throw new Error("This file is empty. Choose a file containing your model.");
  if (code.includes("\0")) throw new Error("Choose a text Python file.");
  return code;
}

export function pythonProject(
  name: string,
  code: string,
  className: string,
  constructor: Record<string, unknown> = {},
): Draft {
  return {
    ...blankProject(name.trim()),
    blueprint: null,
    code,
    class_name: className.trim(),
    constructor,
  };
}
