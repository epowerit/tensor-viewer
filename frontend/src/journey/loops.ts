import type { LoopStep, Operation, Run, Tensor } from "../api/client";
import { producedTensorIds } from "../tensors/provenance";
import {
  layoutJourney,
  type JourneyEdge,
  type JourneyGraph,
  type JourneyNode,
} from "./graph";
import type { JourneyStage } from "./stages";

/** A loop whose iterations all did the same work, so it is drawn once. */
export type LoopFold = Omit<LoopStep, "iteration"> & {
  /** Operation ids of each iteration, in execution order. */
  iterations: string[][];
  /** How many loops enclose this one. */
  depth: number;
};

/** The value range of the tensor one pass hands to the next. */
export type PassRange = { name: string; min: number; max: number } | null;

/** A folded loop as the canvas shows it. */
export type LoopView = LoopFold & {
  /** Per iteration, the range of the body's last result. */
  ranges: PassRange[];
  /** The iteration whose tensors the loop body currently shows (1-based). */
  shown: number;
  /** The loop's body, as the first iteration's operation ids. */
  members: Set<string>;
};

/** A loop whose passes did different work: drawn in full, pass by pass. */
export type PassLoop = Omit<LoopStep, "iteration"> & { iterations: string[][] };

export type LoopGraph = JourneyGraph & {
  loops?: LoopView[];
  passLoops?: PassLoop[];
};

/** One position of playback: an operation, or the repeats of a folded loop. */
export type PlaybackStep =
  | { id: string; operation: Operation; fold?: undefined }
  | { id: string; fold: LoopFold; operation?: undefined };

export const loopStepId = (fold: { id: string }) => `loop:${fold.id}`;

// Same operation, same line, same result shapes: the same visualization.
const signature = (op: Operation, tensors: Record<string, Tensor>) =>
  [
    op.kind,
    op.status,
    op.source?.file ?? "",
    op.source?.line ?? "",
    ...producedTensorIds(op).map((id) => tensors[id]?.shape.join("×") ?? "?"),
  ].join("|");

/**
 * Loops worth folding: two or more iterations, each repeating the first one's
 * operations exactly. A loop inside a later iteration of a folded loop is
 * already hidden with it.
 */
export function loopFolds(trace: Run["trace"]): LoopFold[] {
  const runs = new Map<
    string,
    { step: LoopStep; depth: number; iterations: Map<number, Operation[]> }
  >();
  for (const op of trace.operations)
    op.loops?.forEach((step, depth) => {
      const run = runs.get(step.id) ?? {
        step,
        depth,
        iterations: new Map<number, Operation[]>(),
      };
      const ops = run.iterations.get(step.iteration) ?? [];
      ops.push(op);
      run.iterations.set(step.iteration, ops);
      runs.set(step.id, run);
    });
  const candidates: LoopFold[] = [];
  for (const { step, depth, iterations } of runs.values()) {
    const numbers = [...iterations.keys()].sort((a, b) => a - b);
    if (numbers.length < 2 || numbers.some((n, i) => n !== i + 1)) continue;
    const ordered = numbers.map((n) => iterations.get(n)!);
    const first = ordered[0].map((op) => signature(op, trace.tensors)).join();
    if (
      ordered.some(
        (ops) => ops.map((op) => signature(op, trace.tensors)).join() !== first,
      )
    )
      continue;
    const { iteration: _, ...loop } = step;
    candidates.push({
      ...loop,
      depth,
      iterations: ordered.map((ops) => ops.map((op) => op.id)),
    });
  }
  // Outer loops first, so their later iterations hide the loops inside them.
  candidates.sort((a, b) => a.depth - b.depth);
  const hidden = new Set<string>();
  const folds: LoopFold[] = [];
  for (const fold of candidates) {
    if (fold.iterations[0].some((id) => hidden.has(id))) continue;
    folds.push(fold);
    fold.iterations
      .slice(1)
      .flat()
      .forEach((id) => hidden.add(id));
  }
  return folds;
}

/** Each hidden operation, by the operation of the first iteration it repeats. */
export function hiddenOperations(folds: LoopFold[]) {
  const hidden = new Map<string, string>();
  for (const fold of folds)
    fold.iterations
      .slice(1)
      .forEach((ops) =>
        ops.forEach((id, i) => hidden.set(id, fold.iterations[0][i])),
      );
  return hidden;
}

/** The first-iteration operation that stands for any operation on screen. */
export function representative(hidden: Map<string, string>, id: string) {
  const seen = new Set<string>();
  while (hidden.has(id) && !seen.has(id)) {
    seen.add(id);
    id = hidden.get(id)!;
  }
  return id;
}

/**
 * The operation a first-iteration operation shows when its loops display other
 * iterations. Inner loops apply first: the chosen inner iteration is found in
 * the first outer iteration, then carried to the chosen outer iteration.
 */
export function displayedOperation(
  folds: LoopFold[],
  shown: Record<string, number>,
  id: string,
) {
  for (const fold of [...folds].sort((a, b) => b.depth - a.depth)) {
    const position = fold.iterations[0].indexOf(id);
    const iteration = shown[fold.id] ?? 1;
    if (position >= 0 && iteration > 1)
      id = fold.iterations[iteration - 1]?.[position] ?? id;
  }
  return id;
}

/** Playback visits the first iteration, then one step for all the repeats. */
export function playbackSteps(
  operations: Operation[],
  folds: LoopFold[],
): PlaybackStep[] {
  const hidden = hiddenOperations(folds);
  const closing = new Map<string, LoopFold[]>();
  for (const fold of folds) {
    const last = fold.iterations[0].at(-1)!;
    closing.set(last, [...(closing.get(last) ?? []), fold]);
  }
  const steps: PlaybackStep[] = [];
  for (const operation of operations) {
    if (hidden.has(operation.id)) continue;
    steps.push({ id: operation.id, operation });
    // When one operation ends several loops, the inner loop repeats first.
    for (const fold of (closing.get(operation.id) ?? []).sort(
      (a, b) => b.depth - a.depth,
    ))
      steps.push({ id: loopStepId(fold), fold });
  }
  return steps;
}

/**
 * Draw each folded loop once. Later iterations disappear into the first; edges
 * that carried a value from one iteration to the next become the loop's return
 * arc instead. The body shows the operations and tensors of the chosen
 * iteration, so selecting a node there inspects that iteration.
 */
export function foldJourney(
  graph: JourneyGraph,
  folds: LoopFold[],
  shown: Record<string, number>,
  trace: Run["trace"],
): LoopGraph {
  if (!folds.length) return graph;
  const hidden = hiddenOperations(folds);
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const index = new Map(trace.operations.map((op) => [op.id, op.index]));
  // Tensors of hidden operations become the first iteration's tensors, and
  // those become the chosen iteration's, position by position.
  const zipProduced = (from: string, to: string, into: Map<string, string>) => {
    const a = byId.get(from)?.operation,
      b = byId.get(to)?.operation;
    if (!a || !b) return;
    const left = producedTensorIds(a),
      right = producedTensorIds(b);
    left.forEach((id, i) => right[i] && into.set(id, right[i]));
  };
  const toFirst = new Map<string, string>();
  for (const id of hidden.keys())
    zipProduced(id, representative(hidden, id), toFirst);
  const toShown = new Map<string, string>();
  const nodes: JourneyNode[] = [];
  for (const node of graph.nodes) {
    if (hidden.has(node.id)) continue;
    const displayed = node.operation
      ? displayedOperation(folds, shown, node.id)
      : node.id;
    const source = byId.get(displayed);
    if (displayed !== node.id && source) {
      zipProduced(node.id, displayed, toShown);
      nodes.push({
        ...node,
        operation: source.operation,
        tensors: source.tensors,
      });
    } else nodes.push({ ...node });
  }
  const tensorOf = (id: string) => {
    const first = toFirst.get(id) ?? id;
    return toShown.get(first) ?? first;
  };
  const seen = new Set<string>();
  const edges: JourneyEdge[] = [];
  for (const edge of graph.edges) {
    const source = representative(hidden, edge.source);
    const target = representative(hidden, edge.target);
    // A value carried into the next iteration runs backwards once folded.
    if (
      source === target ||
      (index.has(source) &&
        index.has(target) &&
        index.get(source)! >= index.get(target)!)
    )
      continue;
    const tensorId = tensorOf(edge.tensorId);
    const key = `${source}|${target}|${tensorId}|${edge.inputIndex}|${edge.kind ?? "operand"}`;
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push({ ...edge, source, target, tensorId });
  }
  // Depths follow the folded dependencies; the graph is acyclic again.
  const kept = new Map(nodes.map((node) => [node.id, node]));
  const depths = new Map<string, number>();
  const parents = new Map<string, string[]>();
  for (const edge of edges)
    parents.set(edge.target, [
      ...(parents.get(edge.target) ?? []),
      edge.source,
    ]);
  const depth = (id: string): number => {
    if (depths.has(id)) return depths.get(id)!;
    depths.set(id, 0);
    const value = Math.max(
      0,
      ...(parents.get(id) ?? [])
        .filter((parent) => kept.has(parent))
        .map((parent) => depth(parent) + 1),
    );
    depths.set(id, value);
    return value;
  };
  for (const node of nodes) node.depth = depth(node.id);
  return {
    ...layoutJourney(nodes, edges),
    loops: folds.map((fold) => ({
      ...fold,
      shown: shown[fold.id] ?? 1,
      members: new Set(fold.iterations[0]),
      ranges: passRanges(fold, trace),
    })),
  };
}

/**
 * A collapsed stage inside a folded loop shows the same call of the chosen
 * iteration: same module type and size, starting at the displayed operation.
 */
export function showStageIterations(
  graph: LoopGraph,
  stages: JourneyStage[],
  folds: LoopFold[],
  shown: Record<string, number>,
  trace: Run["trace"],
): LoopGraph {
  if (!folds.length || !Object.values(shown).some((n) => n > 1)) return graph;
  return {
    ...graph,
    nodes: graph.nodes.map((node) => {
      const stage = node.stage;
      if (!stage?.operationIds.length) return node;
      const start = displayedOperation(folds, shown, stage.operationIds[0]);
      if (start === stage.operationIds[0]) return node;
      const match = stages.find(
        (item) =>
          item.operationIds[0] === start &&
          item.module_type === stage.module_type &&
          item.operationIds.length === stage.operationIds.length,
      );
      if (!match) return node;
      return {
        ...node,
        repOperationIds: stage.operationIds,
        stage: {
          ...match,
          id: stage.id,
          parentStageId: stage.parentStageId,
        },
        tensors: match.outputs.map((id) => trace.tensors[id]).filter(Boolean),
      };
    }),
  };
}

/**
 * Where any operation appears on screen: the first-iteration operation that
 * stands for it, and the iteration each folded loop must show to reveal it.
 */
export function locateOperation(folds: LoopFold[], id: string) {
  const shown: Record<string, number> = {};
  for (const fold of [...folds].sort((a, b) => a.depth - b.depth)) {
    const iteration = fold.iterations.findIndex((ops) => ops.includes(id));
    if (iteration < 1) continue;
    shown[fold.id] = iteration + 1;
    id = fold.iterations[0][fold.iterations[iteration].indexOf(id)];
  }
  return { id, shown };
}

/** What each pass leaves behind: the range of its last operation's result. */
export function passRanges(fold: LoopFold, trace: Run["trace"]): PassRange[] {
  const byId = new Map(trace.operations.map((op) => [op.id, op]));
  return fold.iterations.map((ops) => {
    const last = byId.get(ops.at(-1)!);
    const tensor = last && trace.tensors[producedTensorIds(last)[0]];
    return tensor &&
      typeof tensor.minimum === "number" &&
      typeof tensor.maximum === "number" &&
      Number.isFinite(tensor.minimum) &&
      Number.isFinite(tensor.maximum)
      ? { name: tensor.name, min: tensor.minimum, max: tensor.maximum }
      : null;
  });
}

/** A loop header in the editor: how often it ran and whether it is folded. */
export type LoopLine = {
  id: string;
  text: string;
  passes: number;
  folded: boolean;
  /** The first operation the loop ran, for loops shown in full. */
  firstOperation: string;
};

/**
 * Loop headers of one source file, by 1-based line. A header that ran several
 * times (a loop inside an outer loop) describes its first run.
 */
export function loopLines(
  trace: Run["trace"],
  folds: LoopFold[],
  inFile: (file: string | null) => boolean,
): Map<number, LoopLine> {
  const lines = new Map<number, LoopLine>();
  const passes = new Map<string, number>();
  for (const op of trace.operations)
    for (const step of op.loops ?? []) {
      passes.set(step.id, Math.max(passes.get(step.id) ?? 0, step.iteration));
      if (!inFile(step.file ?? null) || lines.has(step.line)) continue;
      lines.set(step.line, {
        id: step.id,
        text: step.text,
        passes: 0,
        folded: folds.some((fold) => fold.id === step.id),
        firstOperation: op.id,
      });
    }
  for (const line of lines.values()) line.passes = passes.get(line.id) ?? 1;
  return lines;
}

/**
 * Loops with two or more passes that are not folded and not hidden inside a
 * folded loop: their passes did different work, so each is shown.
 */
export function passLoops(trace: Run["trace"], folds: LoopFold[]): PassLoop[] {
  const folded = new Set(folds.map((fold) => fold.id));
  const hidden = hiddenOperations(folds);
  const runs = new Map<
    string,
    { step: LoopStep; passes: Map<number, string[]> }
  >();
  for (const op of trace.operations)
    for (const step of op.loops ?? []) {
      if (folded.has(step.id) || hidden.has(op.id)) continue;
      const run = runs.get(step.id) ?? { step, passes: new Map() };
      run.passes.set(step.iteration, [
        ...(run.passes.get(step.iteration) ?? []),
        op.id,
      ]);
      runs.set(step.id, run);
    }
  return [...runs.values()]
    .filter(({ passes }) => passes.size > 1)
    .map(({ step, passes }) => {
      const { iteration: _, ...loop } = step;
      return {
        ...loop,
        iterations: [...passes.entries()]
          .sort(([a], [b]) => a - b)
          .map(([, ops]) => ops),
      };
    });
}
