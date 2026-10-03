import type { Operation } from "../api/client";
import type { PlaybackStep } from "../journey/loops";

/** Steps recorded in code outside the project's own files. */
export const OTHER_CODE = "";

export type OutlineRow =
  | {
      kind: "module";
      /** The module call's last name, such as "attention". */
      name: string;
      /** The full path, such as "GPT / blocks.0 / attention". */
      path: string;
      depth: number;
    }
  | {
      kind: "step";
      step: PlaybackStep;
      line: number | null;
      /** How deep its module call sits, to indent it under its header. */
      depth: number;
    };

/**
 * A statement's right-hand side without its comment: `q = self.query(x)  #
 * axes: …` reads `self.query(x)`.
 */
export function expression(statement: string) {
  return statement
    .replace(/^([^"'#]*?)\s*#.*$/, "$1")
    .trim()
    .replace(/^[^=()[\]{}"']*?[^=!<>]=(?!=)\s*/, "");
}

/** The module path every one of `paths` sits in, part by part. */
function commonPath(paths: string[]) {
  const [first = [], ...rest] = paths.map((path) => path.split(" / "));
  let shared = first.length;
  for (const parts of rest) {
    let i = 0;
    while (i < shared && parts[i] === first[i]) i++;
    shared = i;
  }
  return first.slice(0, shared).join(" / ");
}

/** The last part of a module path, without the parent's prefix. */
function shortName(path: string) {
  const parts = path.split(" / ");
  const last = parts.at(-1) ?? path;
  const parent = parts.at(-2);
  return parent && last.startsWith(`${parent}.`)
    ? last.slice(parent.length + 1)
    : last;
}

/**
 * The run's playback steps under the files whose lines recorded them, in
 * playback order, like an editor's outline: each file's steps, with a header
 * wherever the module call changes. Steps from code outside the project
 * gather under `OTHER_CODE`.
 */
export function stepOutline(
  steps: PlaybackStep[],
  operations: Operation[],
  files: string[],
  entry: string,
): Map<string, OutlineRow[]> {
  const byId = new Map(operations.map((op) => [op.id, op]));
  const known = new Set(files);
  const placed = steps.map((step) => {
    const op = step.fold
      ? byId.get(step.fold.iterations[0][0])
      : step.operation;
    const source = step.fold
      ? { file: step.fold.file ?? null, line: step.fold.line }
      : (op?.source ?? null);
    const file = source ? (source.file ?? entry) : null;
    // A folded loop belongs to the call that loops: the part of its passes'
    // module paths they share, such as "GPT" for blocks.0, blocks.1, ….
    const module = step.fold
      ? commonPath(
          step.fold.iterations.map((pass) => byId.get(pass[0])?.module ?? ""),
        )
      : (op?.module ?? "");
    return {
      step,
      line: source?.line ?? null,
      key: file && known.has(file) ? file : OTHER_CODE,
      module,
    };
  });
  // A module call that is a single step, such as a Linear layer's one
  // matmul, reads as part of its parent: headers mark blocks, not layers. A
  // call whose own steps are only interrupted by its children's, like a
  // block's residual add, is not a single step.
  const recorded = placed.map((item) => item.module);
  const within = (other: string | undefined, module: string) =>
    other === module || !!other?.startsWith(`${module} / `);
  placed.forEach((item, i) => {
    const module = recorded[i];
    if (
      module.includes(" / ") &&
      !within(recorded[i - 1], module) &&
      !within(recorded[i + 1], module)
    )
      item.module = module.slice(0, module.lastIndexOf(" / "));
  });
  const outline = new Map<string, OutlineRow[]>();
  const lastModule = new Map<string, string>();
  for (const { step, line, key, module } of placed) {
    const rows = outline.get(key) ?? [];
    const last = lastModule.get(key) ?? "";
    // Returning from a nested call to one already open adds no header: its
    // steps just step back out, as in an editor's outline.
    if (module && module !== last && !last.startsWith(`${module} / `))
      rows.push({
        kind: "module",
        name: shortName(module),
        path: module,
        depth: module.split(" / ").length - 1,
      });
    lastModule.set(key, module);
    rows.push({
      kind: "step",
      step,
      line,
      depth: module ? module.split(" / ").length : 0,
    });
    outline.set(key, rows);
  }
  return outline;
}

/** An outline row as shown: a module header carries its key and fold. */
export type ShownRow = {
  row: OutlineRow;
  /** A module header's key, the same across runs while its calls are. */
  key?: string;
  /** Steps a folded header hides; absent when it is open. */
  hidden?: number;
  /** The ids of the steps a folded header hides. */
  holds?: string[];
  /** A header's first recorded step: with its path, it names the call. */
  first?: string;
};

/**
 * The rows still shown once the headers in `folded` hide what they contain:
 * every row nested deeper, up to the next header at its own level or above.
 * A folded block holding the current step stays open, so stepping into it
 * shows where playback is. A header's key counts its repeats ("blocks.0",
 * then "blocks.0#2"), so folding one call leaves the others open.
 */
export function foldedOutline(
  rows: OutlineRow[],
  folded: ReadonlySet<string>,
  isCurrent: (row: OutlineRow) => boolean = () => false,
): ShownRow[] {
  const seen = new Map<string, number>();
  const shown: ShownRow[] = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (row.kind !== "module") {
      shown.push({ row });
      continue;
    }
    const count = (seen.get(row.path) ?? 0) + 1;
    seen.set(row.path, count);
    const key = count > 1 ? `${row.path}#${count}` : row.path;
    let end = i + 1;
    while (end < rows.length && rows[end].depth > row.depth) end++;
    const block = rows.slice(i + 1, end);
    const first = block.find(
      (item) => item.kind === "step" && item.step.operation,
    );
    const call = first?.kind === "step" ? { first: first.step.id } : {};
    if (!folded.has(key) || block.some(isCurrent)) {
      shown.push({ row, key, ...call });
      continue;
    }
    const steps = block.flatMap((item) =>
      item.kind === "step" ? [item.step.id] : [],
    );
    shown.push({ row, key, hidden: steps.length, holds: steps, ...call });
    i = end - 1;
  }
  return shown;
}
