import { einopsPattern, variableName } from "./einops";
import type { Operation, Run, Tensor } from "../api/client";
import { kindName } from "../operations/kindName";
import { axisMap, type AxisPart } from "../tensors/axisLineage";
import type { OperationSemantics } from "./sceneSemantics";
import type { JourneyStage } from "./stages";

/**
 * Operations that only rearrange a tensor's elements: every value is carried
 * through unchanged, only the shape or the order of the axes differs.
 */
const LAYOUT = new Set([
  "reshape",
  "view",
  "view_as",
  "reshape_as",
  "transpose",
  "permute",
  "flatten",
  "unflatten",
  "squeeze",
  "unsqueeze",
  "contiguous",
  "movedim",
  "moveaxis",
  "swapaxes",
  "swapdims",
  "t",
  "T",
  "mT",
]);

export const isLayout = (operation: Pick<Operation, "kind">) =>
  LAYOUT.has(operation.kind);

const same = (a: number[], b: number[]) =>
  a.length === b.length && a.every((size, i) => size === b[i]);
const product = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Steps that cut one tensor into several, each a view of its part. */
const SPLITTING = new Set(["unbind", "chunk", "split", "tensor_split"]);

/**
 * A capsule's name, its account in words, and names for the result's axes
 * where the reading gives them: a name, or the input axis an axis carries on.
 */
export type Relayout = {
  title: string;
  how: string;
  axes?: (string | number)[];
};

/**
 * A capsule read from its axis map: what became of each axis of its input.
 * The map is exact, traced through every step, so the same few readings name
 * the same move in any model, whatever its code calls things:
 *
 * - two axes each cut in two, their coarse and fine pieces gathered: a grid
 *   cut into windows (Swin), or into patches when the channels join the fine
 *   pieces (a masked autoencoder), and the reverse;
 * - two whole axes merged with the channels moved after them: an image read
 *   as tokens (ViT), and the reverse;
 * - the last axis cut into pieces that move apart: heads (attention), or
 *   q, k, and v with their heads when it is cut in three;
 * - two axes merged back across another: heads merged;
 * - axes merged in place (flatten), reordered, or size-1 axes added or dropped.
 *
 * Without a map, as for an older run, the shapes alone give a plain name.
 */
export function readRelayout(
  from: number[],
  to: number[],
  map: AxisPart[][] | null,
): Relayout {
  if (map && map.length === to.length) return readMap(from, map);
  if (same(from, to))
    return { title: "Re-layout", how: "the shape is unchanged" };
  return {
    title:
      to.length < from.length
        ? "Flatten"
        : to.length > from.length
          ? "Unflatten"
          : "Reorder axes",
    how: "the values are laid out in a new shape",
  };
}

export function relayoutTitle(
  from: number[],
  to: number[],
  map: AxisPart[][] | null = null,
): string {
  return readRelayout(from, to, map).title;
}

type Placed = AxisPart & { out: number };

function readMap(from: number[], map: AxisPart[][]): Relayout {
  const last = from.length - 1;
  const size = (out: number) => product(map[out].map((part) => part.size));
  // Each result axis carries its input axis's name when it is that axis.
  const carried: (string | number)[] = map.map((parts) =>
    parts.length === 1 && !parts[0].piece ? parts[0].axis : "",
  );
  const named = (names: Record<number, string>) =>
    carried.map((role, out) => names[out] ?? role);
  const placed: Placed[] = map.flatMap((parts, out) =>
    parts.map((part) => ({ ...part, out })),
  );
  // The pieces of each cut input axis, in piece order.
  const cut = new Map<number, Placed[]>();
  for (const part of placed)
    if (part.piece) {
      const pieces = cut.get(part.axis) ?? [];
      pieces[part.piece.index] = part;
      cut.set(part.axis, pieces);
    }
  const merged = map.flatMap((parts, out) =>
    parts.length > 1 ? [{ parts, out }] : [],
  );
  // Read along the result, the input axes come out of order when moved.
  const order = placed.map(
    (part) => part.axis + (part.piece ? part.piece.index / part.piece.of : 0),
  );
  const reordered = order.some((value, i) => i > 0 && value < order[i - 1]);

  // A grid's two axes each cut in two, coarse with coarse and fine with fine.
  const halves = [...cut.entries()].filter(
    ([, pieces]) => pieces.length === 2 && pieces.every(Boolean),
  );
  // Cutting a grid lowers the rank; the same pairing that raises it puts
  // the grid back together, read below.
  if (halves.length === 2 && map.length < from.length) {
    const [[h, [hc, hf]], [w, [wc, wf]]] = halves;
    if (hc.out === wc.out && hf.out === wf.out && hc.out !== hf.out) {
      const grid = `the ${from[h]} × ${from[w]} grid is cut into ${hc.size * wc.size}`;
      const channels = map[hf.out].filter((part) => !part.piece);
      return channels.length
        ? {
            title: "Cut into patches",
            how: `${grid} patches of ${hf.size} × ${wf.size} × ${channels.map((part) => part.size).join(" × ")} channels, each a row of ${size(hf.out)} values`,
            axes: named({ [hc.out]: "patches", [hf.out]: "patch values" }),
          }
        : {
            title: "Partition into windows",
            how: `${grid} windows of ${hf.size} × ${wf.size}, each a row of ${size(hf.out)} positions`,
            axes: named({ [hc.out]: "windows", [hf.out]: "positions" }),
          };
    }
  }
  // Two result axes each built from pieces of the same two input axes: a
  // grid put back together from its windows or patches.
  const rebuilt = merged.filter(
    ({ parts }) =>
      parts.length === 2 &&
      parts.every((part) => part.piece) &&
      parts[0].axis !== parts[1].axis,
  );
  if (
    rebuilt.length === 2 &&
    rebuilt[0].parts.map((part) => part.axis).join() ===
      rebuilt[1].parts.map((part) => part.axis).join()
  ) {
    const [height, width] = rebuilt.map(({ out }) => size(out));
    const inner = rebuilt[0].parts[1].axis;
    const fromPatches = placed.some(
      (part) => part.axis === inner && part.piece && map[part.out].length === 1,
    );
    return fromPatches
      ? {
          title: "Patches to image",
          how: `the patches go back into the ${height} × ${width} image`,
          axes: named({
            [rebuilt[0].out]: "height",
            [rebuilt[1].out]: "width",
          }),
        }
      : {
          title: "Restore window grid",
          how: `the windows go back into the ${height} × ${width} grid`,
          axes: named({
            [rebuilt[0].out]: "height",
            [rebuilt[1].out]: "width",
          }),
        };
  }
  // Two whole neighbouring axes merged, an earlier axis moved after them.
  for (const { parts, out } of merged) {
    const [a, b] = parts;
    if (parts.length !== 2 || a.piece || b.piece || b.axis !== a.axis + 1)
      continue;
    const moved = placed.find(
      (part) => part.out > out && !part.piece && part.axis < a.axis,
    );
    if (moved)
      return {
        title: "Grid to tokens",
        how: `the ${a.size} × ${b.size} grid becomes a sequence of ${size(out)} tokens, with its ${moved.size} channels last`,
        axes: named({ [out]: "tokens" }),
      };
  }
  // One axis cut into two neighbouring result axes, a later axis moved ahead.
  for (const [axis, pieces] of cut) {
    if (pieces.length !== 2 || pieces[1]?.out !== pieces[0]?.out + 1) continue;
    const moved = placed.find(
      (part) => part.out < pieces[0].out && !part.piece && part.axis > axis,
    );
    if (moved && !merged.length)
      return {
        title: "Tokens to grid",
        how: `the ${from[axis]} tokens go back into a ${pieces[0].size} × ${pieces[1].size} grid, with the ${moved.size} channels first`,
        axes: named({ [pieces[0].out]: "height", [pieces[1].out]: "width" }),
      };
  }
  // One axis cut into pieces, nothing merged: heads, or q, k, v and heads.
  if (cut.size === 1 && !merged.length) {
    const [[axis, pieces]] = [...cut.entries()];
    const sizes = pieces.map((part) => part.size);
    const whole = `${axis === last ? "the last axis" : `axis ${axis}`}, ${from[axis]}, splits into ${sizes.join(" × ")}`;
    if (!reordered)
      return {
        title: axis === last ? "Split last axis" : "Split axis",
        how: whole,
      };
    if (axis === last && pieces.length === 3 && sizes[0] <= 3) {
      const which = sizes[0] === 3 ? "q, k, and v" : "k and v";
      const label = sizes[0] === 3 ? "q, k, v" : "k, v";
      return {
        title: `Split ${label} into heads`,
        how: `${whole}: ${which}, each ${sizes[1]} heads of ${sizes[2]}; ${label} move to axis ${pieces[0].out} and the heads to axis ${pieces[1].out}`,
        axes: named({
          [pieces[0].out]: label,
          [pieces[1].out]: "heads",
          [pieces[2].out]: "head features",
        }),
      };
    }
    if (axis === last && pieces.length === 2)
      return {
        title: "Split into heads",
        how: `${whole}, and the ${sizes[0]} moves to axis ${pieces[0].out}`,
        axes: named({
          [pieces[0].out]: "heads",
          [pieces[1].out]: "head features",
        }),
      };
    return { title: "Split and reorder", how: `${whole}, and the pieces move` };
  }
  // Whole axes merged into one, nothing cut: flattened, or heads merged.
  if (!cut.size && merged.length === 1) {
    const { parts, out } = merged[0];
    const axes = parts.map((part) => part.axis);
    const inPlace =
      !reordered &&
      axes.every((axis, i) => i === 0 || axis === axes[i - 1] + 1);
    if (inPlace)
      return parts.length === 2 && axes[1] === last
        ? {
            title: "Merge last axes",
            how: `the last two axes, ${parts[0].size} × ${parts[1].size}, merge into one of ${size(out)}`,
          }
        : {
            title: "Flatten",
            how: `${parts.length} axes merge into one of ${size(out)}`,
          };
    if (parts.length === 2 && out === map.length - 1)
      return {
        title: "Merge heads",
        how: `the ${parts[0].size} on axis ${axes[0]} moves back beside the ${parts[1].size}, and the two merge into a last axis of ${size(out)}`,
        axes: named({ [out]: "features" }),
      };
    return {
      title: "Reorder and flatten",
      how: `the axes are reordered, and ${parts.length} merge into one of ${size(out)}`,
    };
  }
  // Every axis whole: reordered, or only size-1 axes added or dropped.
  if (!cut.size && !merged.length) {
    if (reordered)
      return {
        title: "Reorder axes",
        how: `the axes come in the order ${placed.map((part) => part.axis).join(", ")}`,
      };
    if (map.length > from.length)
      return {
        title: "Add size-1 axes",
        how: `${map.length - from.length} axes of size 1 are added`,
      };
    if (map.length < from.length)
      return {
        title: "Drop size-1 axes",
        how: `${from.length - map.length} axes of size 1 are dropped`,
      };
    return { title: "Re-layout", how: "the shape is unchanged" };
  }
  return {
    title: reordered ? "Regroup and reorder" : "Regroup",
    how: `the axes regroup as ${map
      .map(
        (parts) =>
          parts
            .map((part) =>
              part.piece
                ? `${part.size} of axis ${part.axis}`
                : `axis ${part.axis}`,
            )
            .join(" × ") || "1",
      )
      .join(", ")}`,
  };
}

/**
 * Runs of layout operations that hand a tensor straight from one to the next
 * (reshape, then transpose) as capsules the diagram can fold into one card.
 * A run stays within one module call, and each tensor inside it is read only
 * by the next step and is not a model output, so folding it hides nothing
 * another step uses. A run is at least two steps; the capsule sits in the
 * innermost recorded stage around it; a named call made of exactly that run,
 * such as a window partition, already folds it and gets no capsule.
 */
export function layoutStages(run: Run, stages: JourneyStage[]): JourneyStage[] {
  const { operations, tensors } = run.trace;
  const outputs = new Set(run.trace.output_ids ?? []);
  const readers = new Map<string, number>();
  for (const operation of operations)
    for (const id of operation.inputs)
      readers.set(id, (readers.get(id) ?? 0) + 1);
  const chained = (operation: Operation) =>
    isLayout(operation) &&
    operation.status !== "error" &&
    operation.outputs.length === 1 &&
    !operation.mutations?.length;
  const calls = stages.map((stage) => ({
    stage,
    ids: new Set(stage.operationIds),
  }));
  // The outermost call of code is the whole model, or a console's wrapper:
  // a run that is all of it still gets a capsule, which says what it does.
  // A diagram's outermost calls are its components, named by the builder.
  const named = calls.filter(
    (call) => call.stage.parentStageId !== null || !!run.project?.blueprint,
  );
  const capsules: JourneyStage[] = [];
  for (let i = 0; i < operations.length; i++) {
    const first = operations[i];
    if (!chained(first)) continue;
    let last = i;
    while (last + 1 < operations.length) {
      const here = operations[last];
      const next = operations[last + 1];
      const passed = here.outputs[0];
      if (
        !chained(next) ||
        next.module !== first.module ||
        next.inputs[0] !== passed ||
        readers.get(passed) !== 1 ||
        outputs.has(passed)
      )
        break;
      last++;
    }
    // A chain may end by splitting its tensor into parts, as timm's
    // reshape → permute → unbind hands on q, k, and v: the parts are the
    // capsule's results, and the shape before the split names it.
    const shaped = operations[last].outputs[0];
    const after = operations[last + 1];
    const parts =
      after &&
      SPLITTING.has(after.kind) &&
      after.status !== "error" &&
      after.outputs.length > 1 &&
      after.module === first.module &&
      after.inputs[0] === shaped &&
      readers.get(shaped) === 1 &&
      !outputs.has(shaped)
        ? after
        : undefined;
    if (parts) last++;
    if (last === i) continue;
    const steps = operations.slice(i, last + 1);
    const from = tensors[first.inputs[0]]?.shape;
    const to = tensors[shaped]?.shape;
    if (!from || !to) {
      i = last;
      continue;
    }
    const ids = steps.map((operation) => operation.id);
    // A recorded call that is nothing but this run already folds it.
    if (
      named.some(
        (call) =>
          call.ids.size === ids.length && ids.every((id) => call.ids.has(id)),
      )
    ) {
      i = last;
      continue;
    }
    // What became of each input axis, traced through the steps.
    const map = axisMap(run.trace, first.inputs[0], shaped);
    const parent = calls
      .filter((call) => ids.every((id) => call.ids.has(id)))
      .sort((a, b) => a.ids.size - b.ids.size)[0]?.stage;
    capsules.push({
      id: `stage-layout-${first.id}`,
      path: "",
      module_type: "layout",
      parent_id: null,
      start_index: first.index,
      end_index: steps.at(-1)!.index + 1,
      inputs: [first.inputs[0]],
      outputs: parts ? parts.outputs : [shaped],
      title: readRelayout(from, to, map).title,
      parentStageId: parent?.id ?? null,
      operationIds: ids,
      failed: false,
      layout: { from, to, map, ...(parts ? { via: shaped } : {}) },
    });
    i = last;
  }
  return capsules;
}

/** An axis name the code gave, not the "axis 2" a tensor gets without one. */
const given = (name: string | undefined) =>
  name && !/^axis \d+$/.test(name) ? name : undefined;

/**
 * Each axis's name: the one the code gave, or for an unnamed result axis the
 * one its reading gives (`heads`, `windows`) or its input axis carries.
 */
export function axisNames(
  tensor: Tensor,
  roles?: (string | number)[],
  input?: Tensor,
): (string | null)[] {
  return tensor.shape.map((_, axis) => {
    const role = roles?.[axis];
    return (
      given(tensor.axes?.[axis]) ??
      (typeof role === "number" ? given(input?.axes?.[role]) : role) ??
      null
    );
  });
}

/** A shape with its axis names: "batch 2 · tokens 5 · 8". */
function namedShape(sizes: number[], names: (string | null)[]) {
  return sizes
    .map((size, axis) => (names[axis] ? `${names[axis]} ${size}` : `${size}`))
    .join(" · ");
}

/**
 * A capsule's caption: its name, and what its steps do to the tensor, in the
 * axis names the code declared, with the values unchanged.
 */
export function layoutSemantics(
  run: Run,
  stage: JourneyStage,
): OperationSemantics | undefined {
  if (!stage.layout) return undefined;
  const { tensors } = run.trace;
  const input = tensors[stage.inputs?.[0] ?? ""];
  // A chain ending in a split is described up to the tensor it splits.
  const shaped = tensors[stage.layout.via ?? stage.outputs?.[0] ?? ""];
  const parts = (stage.outputs ?? [])
    .map((id) => tensors[id])
    .filter((tensor): tensor is Tensor => !!tensor);
  if (!input || !shaped || !parts.length) return undefined;
  const byId = new Map(run.trace.operations.map((op) => [op.id, op]));
  const steps = stage.operationIds
    .map((id) => byId.get(id))
    .filter((op): op is Operation => !!op);
  const { how, axes } = readRelayout(
    input.shape,
    shaped.shape,
    stage.layout.map ?? null,
  );
  const handed =
    stage.layout.via && parts.length > 1
      ? `, which ${kindName(steps.at(-1)!.kind)} hands on as ${parts.length} tensors, ${parts.map((part) => part.name).join(", ")}`
      : "";
  const fromNames = axisNames(input);
  const toNames = axisNames(shaped, axes, input);
  const map = stage.layout.map;
  return {
    title: stage.title,
    summary: `${input.name} [${namedShape(input.shape, fromNames)}] → [${namedShape(shaped.shape, toNames)}]: ${how}${handed}. Every value is carried over unchanged; ${steps.map((op) => kindName(op.kind)).join(" then ")} only change the order they are read in.`,
    ...(map
      ? {
          wiring: { input, fromNames, to: shaped.shape, toNames, map },
          einops: einopsPattern(
            input.shape,
            fromNames,
            shaped.shape,
            toNames,
            map,
          )?.call(variableName(input.name)),
        }
      : {}),
    inputs: [{ tensorId: input.id, label: input.name }],
    outputs: parts.map((part) => ({ tensorId: part.id, label: part.name })),
  };
}
