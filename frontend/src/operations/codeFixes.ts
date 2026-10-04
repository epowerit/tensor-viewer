import type { Operation, Run, Tensor } from "../api/client";
import { entryPath, projectFiles } from "../sources/files";
import type { Diagnosis } from "./diagnosis";

/** One edit to one line that might make a failed run go through. */
export type CodeFix = {
  /** What it does, in a few words: "transpose keys". */
  label: string;
  file: string;
  /** 1-based. */
  line: number;
  before: string;
  after: string;
};

/** At most this many candidates are checked. */
const MOST = 5;

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Where `name` stands alone in a line: not part of a longer name or attribute. */
function occurrences(text: string, name: string): number[] {
  const found: number[] = [];
  const pattern = new RegExp(`(?<![\\w.])${escape(name)}(?!\\w)`, "g");
  for (const match of text.matchAll(pattern)) found.push(match.index!);
  // Only the right-hand side of an assignment is read.
  const assign = /^\s*[\w.,\s[\]]+?\s*(?<![=!<>])=(?!=)/.exec(text);
  const from = assign ? assign[0].length : 0;
  return found.filter((at) => at >= from);
}

/** The line with `.call` after each use of `name`, one candidate per use. */
function withCall(text: string, name: string, call: string): string[] {
  return occurrences(text, name)
    .slice(0, 2)
    .map(
      (at) =>
        `${text.slice(0, at + name.length)}.${call}${text.slice(at + name.length)}`,
    );
}

/** A tensor's variable in this line, when it has one there. */
const named = (tensor: Tensor | undefined, text: string) =>
  tensor &&
  /^[A-Za-z_]\w*$/.test(tensor.name) &&
  occurrences(text, tensor.name).length
    ? tensor.name
    : null;

/**
 * Candidate one-line edits for a run that failed: each is a guess until the
 * edited code is checked, which is why every plausible one is offered. They
 * come from the failure: a matrix product or a broadcast whose axes do not
 * line up (transpose an operand), a view after a permute (reshape instead),
 * an index or target of the wrong type (.long() or .float()), a misspelled
 * name or method (the "did you mean"), and a linear layer built for another
 * input size (build it for the size it gets).
 */
export function fixCandidates(
  run: Run,
  diagnosis: Diagnosis | null,
): CodeFix[] {
  const trace = run.trace;
  const error = trace.error;
  if (!error || run.project.blueprint || run.project.script != null) return [];
  const failed: Operation | undefined = trace.operations.find(
    (op) => op.status === "error",
  );
  const file = failed?.source?.file ?? error.file ?? entryPath(run.project);
  const line = failed?.source?.line ?? error.line;
  const code = projectFiles(run.project)[file];
  if (!line || code === undefined) return [];
  const lines = code.split("\n");
  const before = lines[line - 1];
  if (before === undefined) return [];
  const fixes: CodeFix[] = [];
  const add = (label: string, after: string, at = line) => {
    const was = lines[at - 1];
    if (
      after !== was &&
      !fixes.some((fix) => fix.line === at && fix.after === after)
    )
      fixes.push({ label, file, line: at, before: was, after });
  };
  const inputs = (failed?.inputs ?? []).map((id) => trace.tensors[id]);
  const kind = failed?.kind.replace(/^__|__$/g, "") ?? "";
  const message = error.message;

  // Axes that do not line up: try each operand transposed.
  if (
    failed &&
    (/^(matmul|mm|bmm|add|sub|mul|div|true_divide|where|maximum|minimum)$/.test(
      kind,
    ) ||
      /must match the size|cannot be multiplied|shapes cannot be multiplied/.test(
        message,
      ))
  )
    for (const tensor of inputs.slice(0, 2)) {
      const name = named(tensor, before);
      if (name && tensor!.shape.length >= 2)
        for (const after of withCall(before, name, "transpose(-2, -1)"))
          add(`transpose ${name}`, after);
    }

  // view() on memory a permute rearranged.
  if (/view size is not compatible|use \.reshape/i.test(message))
    add("reshape instead of view", before.replace(/\.view\(/g, ".reshape("));

  // An index, target, or condition of the wrong type.
  if (/Long|Int|index|indices|dtype|scalar type|Float|Bool/.test(message))
    for (const tensor of inputs) {
      const name = named(tensor, before);
      if (!name || !tensor) continue;
      const call = tensor.dtype.startsWith("float") ? "long()" : "float()";
      for (const after of withCall(before, name, call))
        add(`${name}.${call}`, after);
    }

  // A misspelled name or tensor method.
  const guess = diagnosis?.suggestion?.match(/^Did you mean \.?(\w+)\?$/)?.[1];
  const unknown =
    /name '(\w+)' is not defined/.exec(message)?.[1] ??
    /has no attribute '(\w+)'/.exec(message)?.[1];
  if (guess && unknown)
    add(
      `${unknown} → ${guess}`,
      before.replace(
        new RegExp(`(?<!\\w)${escape(unknown)}(?!\\w)`, "g"),
        guess,
      ),
    );

  // A linear layer made for another input size: make it for the one it gets.
  if (
    failed &&
    kind === "linear" &&
    inputs[0] &&
    inputs[1]?.shape.length === 2
  ) {
    const gets = inputs[0].shape.at(-1);
    const expects = inputs[1].shape[1];
    if (gets !== undefined && gets !== expects)
      lines.forEach((text, at) => {
        const built = new RegExp(`(Linear\\(\\s*)${expects}(?!\\w)`);
        if (built.test(text))
          add(
            `build the layer for ${gets} features`,
            text.replace(built, `$1${gets}`),
            at + 1,
          );
      });
  }
  return fixes.slice(0, MOST);
}

/** The file with one fix applied. */
export function applyFix(code: string, fix: CodeFix): string | null {
  const lines = code.split("\n");
  if (lines[fix.line - 1] !== fix.before) return null;
  lines[fix.line - 1] = fix.after;
  return lines.join("\n");
}
