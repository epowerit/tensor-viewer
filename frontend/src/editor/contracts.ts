import type { LineResult } from "../console/script";

/**
 * Shape contracts: `# shape: B, T, 8` after an assignment states the shape its
 * tensor must have. Numbers are exact sizes, a letter or word binds to the size
 * it first meets and must match everywhere after, `_` accepts any one axis, and
 * `...` accepts any number of axes. The IDE checks each contract against the
 * recorded run, or a shape check of the current code.
 */
export type ContractEntry =
  | { kind: "size"; size: number }
  | { kind: "name"; name: string }
  | { kind: "any" }
  | { kind: "rest" };
export type Contract = { line: number; text: string; entries: ContractEntry[] };
export type ContractCheck = {
  line: number;
  text: string;
  ok: boolean;
  message: string;
};

const PATTERN = /#\s*shape:\s*(.+)$/;

/** Read a contract from one line, or explain why it cannot be read. */
export function parseContract(
  source: string,
  line: number,
): Contract | { line: number; error: string } | null {
  const found = PATTERN.exec(source);
  if (!found) return null;
  const text = found[1].trim();
  const body = text.replace(/^[[(]|[\])]$/g, "").trim();
  const parts = body ? body.split(/\s*,\s*|\s+/).filter(Boolean) : [];
  const entries: ContractEntry[] = [];
  for (const part of parts) {
    if (/^\d+$/.test(part)) entries.push({ kind: "size", size: Number(part) });
    else if (part === "_") entries.push({ kind: "any" });
    else if (part === "...") entries.push({ kind: "rest" });
    else if (/^[A-Za-z]\w*$/.test(part))
      entries.push({ kind: "name", name: part });
    else return { line, error: `“${part}” is not a size, a name, _, or ...` };
  }
  if (entries.filter((entry) => entry.kind === "rest").length > 1)
    return { line, error: "Use ... at most once in a contract." };
  return { line, text: `[${parts.join(", ")}]`, entries };
}

const describe = (shape: number[]) => `[${shape.join(", ")}]`;

/**
 * Check every contract in order, binding names as they first appear. A line
 * without a current recorded or predicted tensor is skipped, not failed.
 */
export function checkContracts(
  lines: string[],
  results: ReadonlyMap<number, LineResult>,
): ContractCheck[] {
  const bound = new Map<string, { size: number; line: number }>();
  const checks: ContractCheck[] = [];
  lines.forEach((source, index) => {
    const line = index + 1;
    const contract = parseContract(source, line);
    if (!contract) return;
    if ("error" in contract) {
      checks.push({ line, text: "", ok: false, message: contract.error });
      return;
    }
    const result = results.get(line);
    const shape = result?.fresh && !result.error ? result.output?.shape : null;
    if (!shape) return;
    const fail = (message: string) =>
      checks.push({ line, text: contract.text, ok: false, message });
    const rest = contract.entries.findIndex((entry) => entry.kind === "rest");
    const fixed = contract.entries.length - (rest >= 0 ? 1 : 0);
    if (rest < 0 ? shape.length !== fixed : shape.length < fixed) {
      fail(
        `Expected ${rest < 0 ? "" : "at least "}${fixed} ${fixed === 1 ? "axis" : "axes"} ${contract.text}, but ${result!.output!.name} has ${shape.length}: ${describe(shape)}.`,
      );
      return;
    }
    // Pair entries with axes; entries after ... align from the end.
    const pairs = contract.entries.flatMap((entry, i) =>
      entry.kind === "rest"
        ? []
        : [
            {
              entry,
              axis:
                rest >= 0 && i > rest
                  ? shape.length - (contract.entries.length - i)
                  : i,
            },
          ],
    );
    const pending = new Map<string, number>();
    for (const { entry, axis } of pairs) {
      const size = shape[axis];
      if (entry.kind === "size" && entry.size !== size) {
        fail(
          `Axis ${axis} of ${result!.output!.name} should be ${entry.size}, but it is ${size}: ${describe(shape)}.`,
        );
        return;
      }
      if (entry.kind !== "name") continue;
      const earlier = bound.get(entry.name);
      const here = pending.get(entry.name);
      if (earlier && earlier.size !== size) {
        fail(
          `${entry.name} was ${earlier.size} on line ${earlier.line}, but axis ${axis} of ${result!.output!.name} is ${size}: ${describe(shape)}.`,
        );
        return;
      }
      if (here !== undefined && here !== size) {
        fail(
          `${entry.name} names two axes of different sizes here (${here} and ${size}): ${describe(shape)}.`,
        );
        return;
      }
      pending.set(entry.name, size);
    }
    pending.forEach((size, name) => {
      if (!bound.has(name)) bound.set(name, { size, line });
    });
    checks.push({
      line,
      text: contract.text,
      ok: true,
      message: `${result!.output!.name} ${describe(shape)} matches ${contract.text}.`,
    });
  });
  return checks;
}
