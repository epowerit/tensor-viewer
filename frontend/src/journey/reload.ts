import type { Draft, Operation, Run } from "../api/client";
import { withoutComment } from "../console/script";
import { projectFiles } from "../sources/files";
import { draftChanges } from "../workspace/runChanges";
import { layoutStages } from "./relayout";
import {
  detailLevelOf,
  detailLevels,
  journeyStages,
  type DetailLevel,
  type JourneyStage,
} from "./stages";
import type { Viewport } from "./viewport";

type Trace = Run["trace"];

/** How the steps of one run line up with the steps of the next. */
export type RunMatch = {
  /** Each surviving step of the run before, by its id in the new run. */
  operations: Map<string, string>;
  /** Steps of the new run with no step before them. */
  added: string[];
  /** Steps of the run before that the new run no longer takes. */
  removed: string[];
  /** Surviving steps whose results changed shape or type. */
  reshaped: string[];
  /** Surviving steps whose line of code was edited. */
  edited: string[];
};

/**
 * A step as the code wrote it: what it does, where in the code, and inside
 * which module. Operation ids count steps, so they shift whenever a step is
 * added above; this does not.
 */
const stepKey = (op: Operation) =>
  [
    op.kind,
    op.function,
    op.module,
    op.source?.file ?? "",
    (op.source?.text ?? "").trim(),
  ].join("\u0001");

/** Above this many cells, the middle is matched by occurrence instead. */
const ALIGNED = 4_000_000;

/**
 * Lines up two runs of a project, as a diff lines up two versions of a file:
 * the steps both runs share keep their place, so an edit to the code shows as
 * the steps it added, removed, or reshaped, wherever in the model it falls.
 */
export function matchRuns(before: Trace, after: Trace): RunMatch {
  const cached = recent.get(after);
  if (cached?.before === before) return cached.match;
  const a = before.operations,
    b = after.operations;
  const keyA = a.map(stepKey),
    keyB = b.map(stepKey);
  const pairs: [number, number][] = [];
  // Edits are local: the steps before and after the edit match outright.
  let start = 0;
  while (start < a.length && start < b.length && keyA[start] === keyB[start])
    pairs.push([start, start++]);
  let endA = a.length,
    endB = b.length;
  const tail: [number, number][] = [];
  while (endA > start && endB > start && keyA[endA - 1] === keyB[endB - 1])
    tail.push([--endA, --endB]);
  const rows = endA - start,
    columns = endB - start;
  if (rows && columns) {
    if (rows * columns <= ALIGNED) {
      // Longest common subsequence of the edited middle.
      const width = columns + 1;
      const table = new Uint32Array((rows + 1) * width);
      for (let i = rows - 1; i >= 0; i--)
        for (let j = columns - 1; j >= 0; j--)
          table[i * width + j] =
            keyA[start + i] === keyB[start + j]
              ? table[(i + 1) * width + j + 1] + 1
              : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
      let i = 0,
        j = 0;
      while (i < rows && j < columns) {
        if (keyA[start + i] === keyB[start + j]) {
          pairs.push([start + i++, start + j++]);
        } else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) i++;
        else j++;
      }
    } else {
      // Too large to align: the nth step of each kind matches the nth.
      const waiting = new Map<string, number[]>();
      for (let i = endA - 1; i >= start; i--) {
        const queue = waiting.get(keyA[i]) ?? [];
        queue.push(i);
        waiting.set(keyA[i], queue);
      }
      for (let j = start; j < endB; j++) {
        const i = waiting.get(keyB[j])?.pop();
        if (i !== undefined) pairs.push([i, j]);
      }
    }
  }
  pairs.push(...tail.reverse());
  // A step whose line was edited, `return y.softmax(-1)` made `y =
  // y.softmax(-1)`, is still that step: within each gap the diff left, steps
  // of the same kind in the same module pair up in order.
  const loose = (op: Operation) =>
    `${op.kind}\u0001${op.function}\u0001${op.module}`;
  const edited: string[] = [];
  // Gaps lie between pairs in order on both sides; pairs matched by
  // occurrence can cross, so a gap only spans steps neither side has paired.
  const usedA = new Set(pairs.map(([i]) => i)),
    usedB = new Set(pairs.map(([, j]) => j));
  const bounded = [
    [-1, -1] as [number, number],
    ...[...pairs].sort((x, y) => x[0] - y[0]),
    [a.length, b.length] as [number, number],
  ];
  for (let k = 0; k + 1 < bounded.length; k++) {
    const [fromA, fromB] = bounded[k],
      [toA, toB] = bounded[k + 1];
    let j = fromB + 1;
    for (let i = fromA + 1; i < toA && j < toB; i++) {
      if (usedA.has(i)) continue;
      let found = j;
      while (
        found < toB &&
        (usedB.has(found) || loose(b[found]) !== loose(a[i]))
      )
        found++;
      if (found < toB) {
        pairs.push([i, found]);
        usedA.add(i);
        usedB.add(found);
        edited.push(b[found].id);
        j = found + 1;
      }
    }
  }
  const operations = new Map<string, string>();
  const reshaped: string[] = [];
  const shape = (trace: Trace, op: Operation) =>
    op.outputs
      .map((id) => {
        const tensor = trace.tensors[id];
        return tensor ? `${tensor.dtype}[${tensor.shape.join(",")}]` : "";
      })
      .join(";");
  for (const [i, j] of pairs) {
    operations.set(a[i].id, b[j].id);
    if (shape(before, a[i]) !== shape(after, b[j])) reshaped.push(b[j].id);
  }
  const kept = new Set(operations.values());
  const match = {
    operations,
    added: b.filter((op) => !kept.has(op.id)).map((op) => op.id),
    removed: a.filter((op) => !operations.has(op.id)).map((op) => op.id),
    reshaped,
    edited,
  };
  recent.set(after, { before, match });
  return match;
}
// The editor and the diagram both ask about the same pair of runs.
const recent = new WeakMap<Trace, { before: Trace; match: RunMatch }>();

/**
 * Each stage of the run before, by its stage in the new run: the same module
 * call (by path) or capsule holding most of the same steps, or, when every
 * step inside changed, the call at the same path and occurrence.
 */
export function matchStages(
  before: JourneyStage[],
  after: JourneyStage[],
  match: RunMatch,
): Map<string, string> {
  const holding = new Map<string, JourneyStage[]>();
  for (const stage of after)
    for (const id of stage.operationIds)
      holding.set(id, [...(holding.get(id) ?? []), stage]);
  const occurrence = (stages: JourneyStage[]) => {
    const seen = new Map<string, number>();
    return new Map(
      stages
        .filter((stage) => !stage.layout)
        .map((stage) => {
          const n = seen.get(stage.path) ?? 0;
          seen.set(stage.path, n + 1);
          return [stage.id, `${stage.path}#${n}` as string] as const;
        }),
    );
  };
  const placeAfter = new Map(
    [...occurrence(after)].map(([id, place]) => [place, id]),
  );
  const placeBefore = occurrence(before);
  const result = new Map<string, string>();
  const sizes = new Map(after.map((item) => [item.id, item]));
  for (const stage of before) {
    const moved = stage.operationIds.flatMap(
      (id) => match.operations.get(id) ?? [],
    );
    const shared = new Map<string, number>();
    for (const id of moved)
      for (const candidate of holding.get(id) ?? [])
        if (
          !!candidate.layout === !!stage.layout &&
          (stage.layout || candidate.path === stage.path)
        )
          shared.set(candidate.id, (shared.get(candidate.id) ?? 0) + 1);
    let best: string | undefined,
      score = 0;
    for (const [id, count] of shared) {
      const size = sizes.get(id)!.operationIds.length;
      const overlap = count / (stage.operationIds.length + size - count);
      if (overlap > score) [best, score] = [id, overlap];
    }
    best ??= placeAfter.get(placeBefore.get(stage.id) ?? "");
    if (best) result.set(stage.id, best);
  }
  return result;
}

/**
 * The new run's folds, kept from the run before: a detail setting stays that
 * setting, so a block added to the model folds like the others; folds made by
 * hand follow their stages.
 */
export function carryFolds(
  folded: Set<string>,
  levelsBefore: DetailLevel[],
  levelsAfter: DetailLevel[],
  stages: Map<string, string>,
): Set<string> {
  const level = detailLevelOf(levelsBefore, folded);
  const same =
    level >= 0
      ? levelsAfter.find((item) => item.label === levelsBefore[level].label)
      : undefined;
  if (same) return new Set(same.collapsed);
  return new Set(
    [...folded].flatMap((id) => {
      const next = stages.get(id);
      return next ? [next] : [];
    }),
  );
}

/**
 * Names the edit changed: a surviving step whose result is now called
 * something else, where the old name no longer appears in the run at all
 * (`scores` written as `attn`). Pins and anything else kept by name follow.
 */
export function renamedTensors(
  match: RunMatch,
  before: Trace,
  after: Trace,
): Map<string, string> {
  const opsBefore = new Map(before.operations.map((op) => [op.id, op]));
  const opsAfter = new Map(after.operations.map((op) => [op.id, op]));
  const names = new Set(Object.values(after.tensors).map((t) => t.name));
  const renames = new Map<string, string>();
  for (const [from, to] of match.operations) {
    const was = before.tensors[opsBefore.get(from)?.outputs[0] ?? ""]?.name;
    const now = after.tensors[opsAfter.get(to)?.outputs[0] ?? ""]?.name;
    if (was && now && was !== now && !names.has(was)) renames.set(was, now);
  }
  return renames;
}

/** What an edit did to the run, in a few words: "argmax added". */
export function describeChange(
  match: RunMatch,
  after: Trace,
  before?: Trace,
): string {
  const afterIds = new Map(after.operations.map((op) => [op.id, op]));
  const named = (ids: string[], from: Map<string, Operation>) => {
    const kinds = [...new Set(ids.map((id) => from.get(id)?.kind ?? "step"))];
    return kinds.length <= 2 && ids.length <= 3 && !kinds.includes("step")
      ? kinds.join(", ")
      : null;
  };
  const steps = (count: number) => `${count} ${count === 1 ? "step" : "steps"}`;
  const parts: string[] = [];
  if (before)
    for (const [was, now] of renamedTensors(match, before, after))
      parts.push(`${was} → ${now}`);
  // A new input shape explains the shapes that follow from it.
  after.input_ids.forEach((id, i) => {
    const now = after.tensors[id],
      was = before?.tensors[before.input_ids[i]];
    if (now && was && now.shape.join() !== was.shape.join())
      parts.push(`${now.name || "input"} now [${now.shape.join(", ")}]`);
  });
  if (match.added.length)
    parts.push(
      `${named(match.added, afterIds) ?? steps(match.added.length)} added`,
    );
  if (match.removed.length) {
    const was = new Map((before?.operations ?? []).map((op) => [op.id, op]));
    parts.push(
      `${named(match.removed, was) ?? steps(match.removed.length)} removed`,
    );
  }
  const reshaped = match.reshaped.length;
  const only = reshaped === 1 ? afterIds.get(match.reshaped[0]) : undefined;
  const result = only && after.tensors[only.outputs[0]];
  if (only && result)
    parts.push(`${only.kind} now [${result.shape.join(", ")}]`);
  else if (reshaped)
    parts.push(
      `${reshaped} ${reshaped === 1 ? "result" : "results"} changed shape`,
    );
  return parts.length ? parts.join(" · ") : "same steps, values updated";
}

/** Where the canvas was: its camera and where each card sat. */
export type CanvasMemory = {
  view: Viewport;
  size: { width: number; height: number };
  nodes: Map<string, { x: number; y: number }>;
};

/** What a diagram leaves behind for the next run of its project. */
export type LeftOff = {
  run: Run;
  stages: JourneyStage[];
  levels: DetailLevel[];
  collapsed: Set<string>;
  selected: string | null;
  /** The selected step was focused, drawn as its scene. */
  focused: boolean;
  inspector: boolean;
  inspectorView: "code" | "values";
  /** The selected step was enlarged for inspection. */
  expanded: boolean;
  /** The tensor and cell chosen on the selected step. */
  selection: { nodeId: string; tensorId?: string; cell?: number } | null;
  /** How playback was going: a save mid-play plays on in the new run. */
  playback: Playback;
  /** The flow lens on screen, the change lens included. */
  lens: string | null;
  canvas: { current: CanvasMemory | null };
};

export type Playback = {
  playing: boolean;
  speed: number;
  following: boolean;
  reveal: boolean;
};

/** A camera kept across runs: `from` in the run before lands where `to` is. */
export type CarriedView = {
  view: Viewport;
  from: { x: number; y: number };
  to: string;
};

export type Carried = {
  runId: string;
  collapsed: Set<string>;
  selected: string | null;
  focused: boolean;
  inspector: boolean;
  inspectorView: "code" | "values";
  expanded: boolean;
  selection: { nodeId: string; tensorId?: string; cell?: number } | null;
  playback: Playback;
  lens: string | null;
  view: CarriedView | null;
  /** Steps the edit added, edited, or reshaped, marked for a moment. */
  changed: Set<string>;
  /** The first of them to run, to show the change. */
  first: string | null;
  /** Tensors the edit renamed, old name to new. */
  renames: Map<string, string>;
  /** The edit changed which steps run or their shapes, not only values. */
  structural: boolean;
  /** The edit changed code that runs, so values may differ from before. */
  valuesMayDiffer: boolean;
  summary: string;
};

/**
 * The new run of a project opens where its last one was left, as an editor
 * keeps its place when a file reloads: the same step selected, the same
 * calls folded, and the camera on the same card, now drawn from the new run.
 * Null for a project's first run, or another project's.
 */
export function carryOver(
  left: LeftOff | null,
  run: Run | null,
): Carried | null {
  if (!left || !run || left.run.id === run.id) return null;
  // Code that does not parse records no steps: its error shows instead, and
  // the next run that draws a diagram carries on from the last one.
  if (!run.trace.operations.length) return null;
  if (left.run.project_id !== run.project_id) return null;
  const match = matchRuns(left.run.trace, run.trace);
  const calls = journeyStages(run);
  const stages = [...calls, ...layoutStages(run, calls)];
  const stageMap = matchStages(left.stages, stages, match);
  const inputs = new Map(
    left.run.trace.input_ids.map((id, i) => [
      `input-${id}`,
      run.trace.input_ids[i] ? `input-${run.trace.input_ids[i]}` : null,
    ]),
  );
  // A folded loop is named by its header's place in the code, which an edit
  // moves; it is found again through a step inside it.
  const afterOps = new Map(run.trace.operations.map((op) => [op.id, op]));
  const loop = (id: string) => {
    if (!id.startsWith("loop:")) return null;
    for (const op of left.run.trace.operations) {
      const depth = op.loops?.findIndex((step) => `loop:${step.id}` === id);
      if (depth === undefined || depth < 0) continue;
      const now = afterOps.get(match.operations.get(op.id) ?? "");
      const step = now?.loops?.[depth];
      if (step) return `loop:${step.id}`;
    }
    return null;
  };
  const node = (id: string) =>
    match.operations.get(id) ??
    stageMap.get(id) ??
    inputs.get(id) ??
    loop(id) ??
    null;
  // A selected step the edit removed gives way to the next one that is
  // still there, or else the last one before it.
  const before = left.run.trace.operations;
  const at = before.findIndex((op) => op.id === left.selected);
  const survivor = (ids: Iterable<Operation>) => {
    for (const op of ids) {
      const id = match.operations.get(op.id);
      if (id) return id;
    }
    return null;
  };
  const exact = left.selected ? node(left.selected) : null;
  const selected =
    exact ??
    (at >= 0
      ? (survivor(before.slice(at + 1)) ??
        survivor(before.slice(0, at).reverse()))
      : null);
  // The tensor chosen on it, as the same operand or result, and its cell
  // while the shape is the same.
  const selection = (() => {
    const chosen = left.selection;
    if (!exact || !chosen || chosen.nodeId !== left.selected) return null;
    const was = before[at],
      now = run.trace.operations.find((op) => op.id === exact);
    if (!was || !now) return { nodeId: exact };
    // No tensor chosen: the cell is on the step's result.
    if (!chosen.tensorId) {
      const same =
        left.run.trace.tensors[was.outputs[0]]?.shape.join() ===
        run.trace.tensors[now.outputs[0]]?.shape.join();
      return { nodeId: exact, cell: same ? chosen.cell : undefined };
    }
    const place = (op: Operation, id: string): [string, number] | null => {
      const output = op.outputs.indexOf(id);
      if (output >= 0) return ["outputs", output];
      const input = op.inputs.indexOf(id);
      return input >= 0 ? ["inputs", input] : null;
    };
    const where = place(was, chosen.tensorId);
    const tensorId = where
      ? now[where[0] as "outputs" | "inputs"][where[1]]
      : undefined;
    const same =
      tensorId &&
      left.run.trace.tensors[chosen.tensorId]?.shape.join() ===
        run.trace.tensors[tensorId]?.shape.join();
    return {
      nodeId: exact,
      tensorId: same ? tensorId : undefined,
      cell: same ? chosen.cell : undefined,
    };
  })();
  // The camera follows the selected card, or else the card nearest the
  // middle of the screen that the new run still draws.
  const memory = left.canvas.current;
  let view: CarriedView | null = null;
  if (memory) {
    const { view: camera, size, nodes } = memory;
    const middle = {
      x: (size.width / 2 - camera.x) / camera.scale,
      y: (size.height / 2 - camera.y) / camera.scale,
    };
    const candidates =
      selected && left.selected && nodes.has(left.selected)
        ? [left.selected]
        : [...nodes.keys()].sort((a, b) => {
            const pa = nodes.get(a)!,
              pb = nodes.get(b)!;
            return (
              Math.hypot(pa.x - middle.x, pa.y - middle.y) -
              Math.hypot(pb.x - middle.x, pb.y - middle.y)
            );
          });
    for (const id of candidates) {
      const to = node(id);
      if (!to) continue;
      view = { view: camera, from: nodes.get(id)!, to };
      break;
    }
  }
  const changed = new Set([...match.added, ...match.edited, ...match.reshaped]);
  // What the save edited, and whether any of it runs: a comment, or the same
  // code run again, changes no value.
  const edits = draftChanges(left.run.project, run.project);
  const renames = renamedTensors(match, left.run.trace, run.trace);
  const stepsSame =
    !match.added.length &&
    !match.removed.length &&
    !match.reshaped.length &&
    !renames.size;
  const sameCode =
    !run.project.blueprint &&
    edits.every((edit) => / · \d+ lines?$/.test(edit)) &&
    runningCode(left.run.project) === runningCode(run.project);
  return {
    runId: run.id,
    collapsed: carryFolds(
      left.collapsed,
      left.levels,
      detailLevels(stages),
      stageMap,
    ),
    selected,
    focused: !!selected && left.focused,
    inspector: !!selected && left.inspector,
    inspectorView: left.inspectorView,
    expanded: !!exact && left.expanded,
    selection,
    lens: left.lens,
    playback: {
      ...left.playback,
      playing: !!selected && left.playback.playing,
    },
    view,
    changed,
    renames,
    // Removed steps show where they were: the step that now follows them.
    first:
      run.trace.operations.find((op) => changed.has(op.id))?.id ??
      (() => {
        const gone = before.findIndex((op) => !match.operations.has(op.id));
        return gone >= 0
          ? (survivor(before.slice(gone + 1)) ??
              survivor(before.slice(0, gone).reverse()))
          : null;
      })(),
    structural: !!(
      match.added.length ||
      match.removed.length ||
      match.reshaped.length
    ),
    summary:
      stepsSame && sameCode
        ? edits.length
          ? "comments only"
          : "re-run, same code"
        : describeChange(match, run.trace, left.run.trace),
    valuesMayDiffer: !sameCode,
  };
}

/** The code that runs: every file without comments or blank lines. */
const runningCode = (draft: Draft) =>
  JSON.stringify(
    Object.entries(
      draft.script != null ? { script: draft.script } : projectFiles(draft),
    )
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([path, text]) => [
        path,
        text
          .split("\n")
          .map(withoutComment)
          .filter((line) => line.trim()),
      ]),
  );
