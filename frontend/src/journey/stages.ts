import type { ModuleCall, Run } from "../api/client";
import { layoutJourney, type JourneyGraph, type JourneyNode } from "./graph";

export type JourneyStage = ModuleCall & {
  title: string;
  parentStageId: string | null;
  operationIds: string[];
  failed: boolean;
};

const readable = (name: string) => name.replace(/([a-z])([A-Z])/g, "$1 $2");

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
