import type { ModuleCall, Run } from "../api/client";
import type { AxisPart } from "../tensors/axisLineage";
import { layoutJourney, type JourneyGraph, type JourneyNode } from "./graph";

export type JourneyStage = ModuleCall & {
  title: string;
  parentStageId: string | null;
  operationIds: string[];
  failed: boolean;
  /**
   * A layout capsule rather than a module call: steps that only rearrange a
   * tensor, from one shape to another, with every value unchanged.
   */
  layout?: {
    from: number[];
    to: number[];
    /** When the chain ends by splitting, the tensor it splits, shaped `to`. */
    via?: string;
    /** What became of each input axis: for each result axis, its parts. */
    map?: AxisPart[][] | null;
  };
};

/**
 * A class name as words, keeping acronyms whole: "CausalSelfAttention" reads
 * "Causal Self Attention", "TinyViT" "Tiny ViT", "MLPBlock" "MLP Block".
 */
export const readable = (name: string) =>
  name
    .replace(/([a-z0-9])([A-Z][a-z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2");

/** Stages describe recorded invocations, not guesses from shape or source lines. */
export function journeyStages(run: Run): JourneyStage[] {
  const calls = run.trace.module_calls ?? [];
  const byId = new Map(calls.map((call) => [call.id, call]));
  const candidates = calls.filter(
    (call) =>
      call.end_index - call.start_index >= 2 &&
      call.start_index >= 0 &&
      call.end_index <= run.trace.operations.length &&
      !(run.project.blueprint && call.parent_id === null),
  );
  const candidateIds = new Set(candidates.map((call) => call.id));
  return candidates.map((call) => {
    let parent = call.parent_id;
    const visited = new Set<string>();
    while (parent && !candidateIds.has(parent) && !visited.has(parent)) {
      visited.add(parent);
      parent = byId.get(parent)?.parent_id ?? null;
    }
    const componentMatch = /^stage_(\d+)$/.exec(call.path);
    const component = componentMatch
      ? run.project.blueprint?.components[Number(componentMatch[1])]
      : undefined;
    const visionPart = /^stage_(\d+)\.(patches|tokens|encoder|readout)$/.exec(
      call.path,
    );
    const visionTitle =
      visionPart &&
      run.project.blueprint?.components[Number(visionPart[1])]?.kind === "vit"
        ? {
            patches: "Patch embedding",
            tokens: "Class token + positions",
            encoder: "Transformer encoder",
            readout: "Class-token classifier",
          }[visionPart[2]]
        : undefined;
    const title =
      visionTitle ??
      hierarchyStageTitle(run, call) ??
      component?.custom?.name ??
      (component
        ? ({
            attention: "Self-attention",
            transformer: "Transformer blocks",
            mlp: "Feed-forward network",
            patch_embedding: "Patch embedding",
            vit: "Vision Transformer",
            hierarchical_vit: "Hierarchical vision",
            window_attention: "Window transformer",
            shifted_window: "Shifted-window transformer",
            window_pair: "Regular + shifted windows",
            window_partition: "Partition windows",
            window_reverse: "Restore windows",
            patch_merging: "Patch merging",
            spatial_readout: "Spatial classifier",
            token_preparation: "Class token + positions",
            class_readout: "Class-token classifier",
            rnn: "RNN",
          }[component.kind] ?? readable(call.module_type))
        : readable(call.module_type));
    const operations = run.trace.operations.slice(
      call.start_index,
      call.end_index,
    );
    return {
      ...call,
      id: `stage-${call.id}`,
      parentStageId:
        parent && candidateIds.has(parent) ? `stage-${parent}` : null,
      title,
      operationIds: operations.map((op) => op.id),
      failed:
        operations.some((op) => op.status === "error") ||
        (!!run.trace.error &&
          !call.outputs.length &&
          call.end_index === run.trace.operations.length),
    };
  });
}

function hierarchyStageTitle(run: Run, call: ModuleCall) {
  const match = /^stage_(\d+)\.(.+)$/.exec(call.path);
  if (!match) return undefined;
  const kind = run.project.blueprint?.components[Number(match[1])]?.kind;
  if (kind === "hierarchical_vit") {
    const titles: Record<string, string> = {
      fine: "Fine-grid windows",
      coarse: "Coarse-grid windows",
      merge: "Patch merging",
      readout: "Spatial classifier",
      "fine.partition": "Partition fine windows",
      "coarse.partition": "Partition coarse windows",
      "fine.restore": "Restore fine grid",
      "coarse.restore": "Restore coarse grid",
      "fine.block": "Within each fine window",
      "coarse.block": "Within each coarse window",
      "merge.group": "Group 2×2 neighbors",
    };
    return titles[match[2]];
  }
  if (kind === "window_attention")
    return (
      {
        partition: "Partition windows",
        restore: "Restore grid",
        block: "Within each window",
      } as Record<string, string>
    )[match[2]];
  if (kind === "shifted_window" || kind === "window_pair") {
    const path = match[2].replace(/^(regular|shifted)\./, "");
    return (
      {
        regular: "Regular windows",
        shifted: "Shifted windows",
        partition: "Partition windows",
        restore: "Restore grid",
        "attention.adjust": "Relative bias + mask",
        attention: "Window attention",
      } as Record<string, string>
    )[path];
  }
  if (kind === "patch_merging" && match[2] === "group")
    return "Group 2×2 neighbors";
}

export function stageAncestors(
  stages: JourneyStage[],
  operationId: string,
): JourneyStage[] {
  return stages.filter((stage) => stage.operationIds.includes(operationId));
}

/** Contract contiguous call intervals while retaining every crossing tensor dependency. */
export function collapseJourney(
  graph: JourneyGraph,
  stages: JourneyStage[],
  collapsed: Set<string>,
  run: Run,
): JourneyGraph {
  if (!collapsed.size) return graph;
  const active = stages.filter(
    (stage) =>
      collapsed.has(stage.id) &&
      !stages.some(
        (parent) =>
          parent.id !== stage.id &&
          collapsed.has(parent.id) &&
          // Parentage distinguishes nested calls with exactly the same operation range.
          isAncestor(parent.id, stage, stages),
      ),
  );
  const owners = new Map(
    active.flatMap((stage) =>
      stage.operationIds.map((id) => [id, stage] as const),
    ),
  );
  const nodes: JourneyNode[] = [];
  const added = new Set<string>();
  for (const node of graph.nodes) {
    const stage = owners.get(node.id);
    if (!stage) {
      nodes.push({ ...node });
      continue;
    }
    if (added.has(stage.id)) continue;
    added.add(stage.id);
    nodes.push({
      id: stage.id,
      stage,
      tensors: stage.outputs.map((id) => run.trace.tensors[id]).filter(Boolean),
      parameterCount: 0,
      terminal: graph.nodes.some(
        (n) => stage.operationIds.includes(n.id) && n.terminal,
      ),
      depth: 0,
      x: 0,
      y: 0,
    });
  }
  const seen = new Set<string>();
  const edges = graph.edges.flatMap((edge) => {
    const source = owners.get(edge.source)?.id ?? edge.source;
    const target = owners.get(edge.target)?.id ?? edge.target;
    if (source === target) return [];
    const key = owners.has(edge.target)
      ? `${source}:${target}:${edge.tensorId}:${edge.kind ?? "operand"}`
      : edge.id;
    if (seen.has(key)) return [];
    seen.add(key);
    return [{ ...edge, source, target }];
  });
  // Include storage side effects that leave a collapsed call, even when they
  // were not explicit return values of that Python module.
  for (const node of nodes) {
    if (!node.stage) continue;
    const ids = new Set(node.tensors.map((t) => t.id));
    const effects = [
      ...edges.filter((e) => e.source === node.id).map((e) => e.tensorId),
      ...graph.nodes
        .filter((n) => node.stage!.operationIds.includes(n.id))
        .flatMap((n) => n.tensors.map((t) => t.id))
        .filter((id) => run.trace.output_ids.includes(id)),
    ];
    for (const id of effects) {
      if (!ids.has(id) && run.trace.tensors[id]) {
        node.tensors.push(run.trace.tensors[id]);
        ids.add(id);
      }
    }
  }
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const depths = new Map<string, number>();
  function depth(id: string): number {
    if (depths.has(id)) return depths.get(id)!;
    const parents = edges.filter((e) => e.target === id);
    const value = parents.length
      ? Math.max(...parents.map((e) => depth(e.source))) + 1
      : 0;
    depths.set(id, value);
    return value;
  }
  for (const node of byId.values()) node.depth = depth(node.id);
  return layoutJourney(nodes, edges);
}

function isAncestor(
  id: string,
  stage: JourneyStage,
  stages: JourneyStage[],
): boolean {
  let parent = stage.parentStageId;
  const visited = new Set<string>();
  while (parent && !visited.has(parent)) {
    if (parent === id) return true;
    visited.add(parent);
    parent = stages.find((item) => item.id === parent)?.parentStageId ?? null;
  }
  return false;
}

/**
 * A stage as a screen reader hears it: its title, the module path when it
 * adds something, and the steps it covers ("Block (blocks.1), steps 9–16").
 */
export function stageLabel(
  stage: Pick<JourneyStage, "title" | "path" | "start_index" | "end_index">,
): string {
  const path =
    stage.path && stage.path.toLowerCase() !== stage.title.toLowerCase()
      ? ` (${stage.path})`
      : "";
  return `${stage.title}${path}, steps ${stage.start_index + 1}–${stage.end_index}`;
}

/** One setting of the diagram's detail dial. */
export type DetailLevel = {
  /** What the level shows as its finest unit: "blocks", "operations". */
  label: string;
  /** A sentence for its tooltip. */
  detail: string;
  /** The stages folded at this level. */
  collapsed: Set<string>;
};

/** "blocks.3" under "blocks" reads "blocks", "encoder.attention" "attention". */
function callName(stage: JourneyStage) {
  const last = stage.path.split(".").filter((part) => !/^\d+$/.test(part));
  return last.at(-1) || stage.title;
}

/**
 * The detail dial, coarse to fine, named after the model's own calls: the
 * whole model, then each level of nested module calls, then every operation
 * with layout capsules still folded, then every step. Level k folds every
 * module call nested k deep or deeper, so the level names what is left open
 * to see inside.
 */
export function detailLevels(stages: JourneyStage[]): DetailLevel[] {
  const modules = stages.filter((stage) => !stage.layout);
  const capsules = stages.filter((stage) => stage.layout).map((s) => s.id);
  if (!modules.length && !capsules.length) return [];
  const byId = new Map(modules.map((stage) => [stage.id, stage]));
  const depthOf = (stage: JourneyStage) => {
    let depth = 0;
    let parent = stage.parentStageId;
    const seen = new Set<string>();
    while (parent && byId.has(parent) && !seen.has(parent)) {
      seen.add(parent);
      depth++;
      parent = byId.get(parent)!.parentStageId;
    }
    return depth;
  };
  const depths = new Map(modules.map((stage) => [stage.id, depthOf(stage)]));
  const deepest = Math.max(-1, ...depths.values());
  const levels: DetailLevel[] = [];
  for (let depth = 0; depth <= deepest; depth++) {
    // Named after its calls, the ones covering the most steps first, so an
    // attention block leads its small norms.
    const weight = new Map<string, number>();
    for (const stage of modules)
      if (depths.get(stage.id) === depth)
        weight.set(
          callName(stage),
          (weight.get(callName(stage)) ?? 0) + stage.operationIds.length,
        );
    const names = [...weight.keys()].sort(
      (a, b) => weight.get(b)! - weight.get(a)!,
    );
    levels.push({
      label: names.slice(0, 2).join(" · ") + (names.length > 2 ? " · …" : ""),
      detail:
        depth === 0
          ? `The model as its outermost calls: ${names.join(", ")}`
          : `Open down to the calls ${depth} deep, ${names.join(", ")}, each folded`,
      collapsed: new Set([
        ...modules
          .filter((stage) => depths.get(stage.id)! >= depth)
          .map((stage) => stage.id),
        ...capsules,
      ]),
    });
  }
  if (capsules.length)
    levels.push({
      label: "operations",
      detail:
        "Every operation, with steps that only rearrange a tensor folded into one",
      collapsed: new Set(capsules),
    });
  levels.push({
    label: "every step",
    detail: "Every recorded step, nothing folded",
    collapsed: new Set(),
  });
  return levels;
}

/** Which level of the dial a set of folded stages is, or -1 for none. */
export function detailLevelOf(levels: DetailLevel[], collapsed: Set<string>) {
  return levels.findIndex(
    (level) =>
      level.collapsed.size === collapsed.size &&
      [...level.collapsed].every((id) => collapsed.has(id)),
  );
}

/**
 * Folds as playback moves: the folded stages around the step on screen open
 * so it can be seen, and those `refold` allows (layout capsules) close again
 * once playback leaves them, as the explorer's outline does. `revealed` holds
 * the stages opened this way and still open; anything opened by hand is
 * never in it and stays open.
 */
export function followFolds(
  collapsed: ReadonlySet<string>,
  revealed: ReadonlySet<string>,
  around: ReadonlySet<string>,
  refold: (id: string) => boolean,
): { collapsed: Set<string>; revealed: Set<string> } {
  const next = new Set(collapsed);
  const still = new Set<string>();
  for (const id of revealed)
    if (around.has(id)) still.add(id);
    else next.add(id);
  for (const id of around) if (next.delete(id) && refold(id)) still.add(id);
  return { collapsed: next, revealed: still };
}

/**
 * How many cards the canvas draws at a level: one per step outside a folded
 * stage, and one per outermost folded stage. A folded loop draws its later
 * passes once, so steps in `hidden` (those passes) are not counted.
 */
export function levelCards(
  level: DetailLevel,
  stages: JourneyStage[],
  operationIds: string[],
  hidden: { has: (id: string) => boolean },
): number {
  const byId = new Map(stages.map((stage) => [stage.id, stage]));
  const inFoldedParent = (stage: JourneyStage) => {
    let parent = stage.parentStageId;
    const seen = new Set<string>();
    while (parent && !seen.has(parent)) {
      if (level.collapsed.has(parent)) return true;
      seen.add(parent);
      parent = byId.get(parent)?.parentStageId ?? null;
    }
    return false;
  };
  const folded = stages.filter(
    (stage) => level.collapsed.has(stage.id) && !inFoldedParent(stage),
  );
  const covered = new Set(folded.flatMap((stage) => stage.operationIds));
  return (
    operationIds.filter((id) => !hidden.has(id) && !covered.has(id)).length +
    folded.filter((stage) => !hidden.has(stage.operationIds[0])).length
  );
}

/**
 * The dial setting a run opens on: the finest that draws at most `most`
 * cards, so a small model shows every step and a large one its blocks, the
 * same for any model, whatever its code calls things.
 */
export function startingLevel(
  levels: DetailLevel[],
  stages: JourneyStage[],
  operationIds: string[],
  hidden: { has: (id: string) => boolean },
  most = 24,
): DetailLevel | undefined {
  for (let i = levels.length - 1; i > 0; i--)
    if (levelCards(levels[i], stages, operationIds, hidden) <= most)
      return levels[i];
  return levels[0];
}

const FOLDS = "tensorviewer.folds";

/**
 * The folds a project was left with, as stage ids that still exist. Null
 * when none were kept, or when none of them survive a change to the code, so
 * the run opens on its starting detail instead. Kept in this browser.
 */
export function rememberedFolds(
  projectId: string,
  stages: JourneyStage[],
): Set<string> | null {
  try {
    const saved: unknown = JSON.parse(localStorage.getItem(FOLDS) ?? "{}")[
      projectId
    ];
    if (!Array.isArray(saved)) return null;
    const known = new Set(stages.map((stage) => stage.id));
    const kept = saved.filter(
      (id): id is string => typeof id === "string" && known.has(id),
    );
    return saved.length && !kept.length ? null : new Set(kept);
  } catch {
    return null;
  }
}

/** Keeps a project's folds for its next run or visit. */
export function rememberFolds(projectId: string, folded: Iterable<string>) {
  try {
    const all = JSON.parse(localStorage.getItem(FOLDS) ?? "{}");
    all[projectId] = [...folded];
    localStorage.setItem(FOLDS, JSON.stringify(all));
  } catch {
    // Without storage the folds last until the page reloads.
  }
}
