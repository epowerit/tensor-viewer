import type { Operation, Run, Tensor } from "../api/client";
import { tensorRelation } from "../operations/relations";
import { producedTensorIds } from "./provenance";

/**
 * Where each axis of a tensor comes from, traced back through the recorded
 * operations to the axes of inputs and parameters.
 *
 * A term names one source axis. An axis built by merging is a product of
 * terms; an axis cut out of a larger one is a part of a term. When an
 * operation computes an axis in a way that has no axis-level story (a
 * convolution's spatial output, a value-dependent selection), the lineage
 * stops there and says so rather than guessing.
 */
export type AxisTerm = {
  /** `x.features`, or `x axis 2` when the axis has no meaningful name. */
  label: string;
  size: number;
  /** For a piece of a split axis: which piece, out of how many. */
  part?: { index: number; of: number; sizes: number[] };
  /** Role of the tensor the axis starts at: data, weights, or created inside. */
  role?: Tensor["role"];
};
export type AxisStory = {
  size: number;
  terms: AxisTerm[];
  /** Why the trace stops, or what happened to an axis without a source. */
  note: string | null;
};

const named = (name: string) => !/^axis \d+$/.test(name);

/** The name an axis goes by wherever it travels: `x.features`, or `x axis 2`. */
export function rootLabel(tensor: Tensor, axis: number): string {
  return named(tensor.axes[axis] ?? "")
    ? `${tensor.name}.${tensor.axes[axis]}`
    : `${tensor.name} axis ${axis}`;
}

function root(tensor: Tensor): AxisStory[] {
  return tensor.shape.map((size, axis) => ({
    size,
    terms: [
      {
        label: rootLabel(tensor, axis),
        size,
        role: tensor.role,
      },
    ],
    note: null,
  }));
}
const fresh = (size: number, note: string): AxisStory => ({
  size,
  terms: [],
  note,
});
const carry = (story: AxisStory, size = story.size): AxisStory =>
  size === story.size ? story : { ...story, size };

/** Consecutive pieces that rebuild a whole split axis become that axis again. */
function rejoin(terms: AxisTerm[]): AxisTerm[] {
  const first = terms[0]?.part;
  if (
    first &&
    terms.length === first.of &&
    terms.every(
      (term, index) =>
        term.label === terms[0].label &&
        term.part?.index === index &&
        term.part.of === first.of &&
        term.part.sizes.join() === first.sizes.join(),
    )
  ) {
    const { part: _, ...whole } = terms[0];
    return [{ ...whole, size: first.sizes.reduce((a, b) => a * b, 1) }];
  }
  return terms;
}

/**
 * The runs of non-size-one axes that a reshape regroups together: within a
 * run, input and output element counts agree. Null when the shapes disagree.
 */
export function regroupRuns(
  input: number[],
  output: number[],
): { from: number[]; to: number[] }[] | null {
  const inAxes = input.flatMap((size, axis) => (size === 1 ? [] : [axis]));
  const outAxes = output.flatMap((size, axis) => (size === 1 ? [] : [axis]));
  const runs: { from: number[]; to: number[] }[] = [];
  let i = 0,
    j = 0;
  while (i < inAxes.length && j < outAxes.length) {
    const from = [inAxes[i++]],
      to = [outAxes[j++]];
    let left = input[from[0]],
      right = output[to[0]];
    while (left !== right) {
      if (left < right) {
        if (i >= inAxes.length) return null;
        from.push(inAxes[i++]);
        left *= input[from.at(-1)!];
      } else {
        if (j >= outAxes.length) return null;
        to.push(outAxes[j++]);
        right *= output[to.at(-1)!];
      }
    }
    runs.push({ from, to });
  }
  return runs;
}

/**
 * Regroup axes the way reshape does: axes merge or split only within runs
 * whose element counts agree. Size-one axes carry no elements and are matched
 * separately.
 */
export function regroup(
  input: number[],
  output: number[],
  stories: AxisStory[],
): AxisStory[] {
  const result: AxisStory[] = output.map((size) =>
    fresh(size, "an axis of size 1"),
  );
  // Size-one axes hold no elements; carry them across only when their count
  // is unchanged, so their order identifies them.
  const inOnes = input.flatMap((size, axis) => (size === 1 ? [axis] : []));
  const outOnes = output.flatMap((size, axis) => (size === 1 ? [axis] : []));
  if (inOnes.length === outOnes.length)
    outOnes.forEach((axis, k) => (result[axis] = stories[inOnes[k]]));
  const runs = regroupRuns(input, output);
  if (!runs) return output.map((size) => fresh(size, "regrouped by reshape"));
  for (const { from, to } of runs) {
    const terms = rejoin(from.flatMap((axis) => stories[axis].terms));
    const unexplained = from.some((axis) => !stories[axis].terms.length);
    if (to.length === 1) {
      const story = from.length === 1 ? stories[from[0]] : null;
      result[to[0]] = story
        ? carry(story, output[to[0]])
        : unexplained
          ? fresh(output[to[0]], "merged by reshape")
          : { size: output[to[0]], terms, note: null };
    } else if (from.length === 1 && stories[from[0]].terms.length === 1) {
      const sizes = to.map((axis) => output[axis]);
      to.forEach((axis, index) => {
        result[axis] = {
          size: output[axis],
          terms: [
            {
              ...stories[from[0]].terms[0],
              size: output[axis],
              part: { index, of: to.length, sizes },
            },
          ],
          note: null,
        };
      });
    } else
      to.forEach(
        (axis) => (result[axis] = fresh(output[axis], "regrouped by reshape")),
      );
  }
  return result;
}

const termText = (term: AxisTerm) =>
  term.part ? `${term.label} piece ${term.part.index + 1}` : term.label;

/**
 * A reshape that puts elements from different source axes side by side, the
 * classic result of reshaping heads back without permuting them first. Only
 * runs whose lineage is fully known are judged; merging whole axes in their
 * order (batch × tokens) or merging a piece with another axis (batch × heads)
 * is left alone, because that is how attention batches its heads.
 */
export function scrambledAxes(
  input: number[],
  output: number[],
  stories: AxisStory[],
): { axes: number[]; sources: string[]; reason: "regrouped" | "reordered" }[] {
  const runs = regroupRuns(input, output);
  if (!runs || stories.length !== input.length) return [];
  const found: ReturnType<typeof scrambledAxes> = [];
  for (const { from, to } of runs) {
    if (from.length < 2 || from.some((axis) => !stories[axis].terms.length))
      continue;
    const terms = from.flatMap((axis) => stories[axis].terms);
    const sources = terms.map(termText);
    if (to.length > 1) {
      // Cut into new axes: fine only when the run is one axis, rebuilt whole.
      if (rejoin(terms).length > 1)
        found.push({ axes: to, sources, reason: "regrouped" });
      continue;
    }
    // One merged axis: every piece of a split axis present, but not in order
    // or not next to each other.
    const labels = new Set(
      terms.flatMap((term) => (term.part ? [term.label] : [])),
    );
    for (const label of labels) {
      const positions = terms.flatMap((term, at) =>
        term.label === label && term.part ? [at] : [],
      );
      const of = terms[positions[0]].part!.of;
      if (positions.length !== of) continue;
      const inOrder = positions.every(
        (at, k) =>
          terms[at].part!.index === k &&
          (k === 0 || at === positions[k - 1] + 1),
      );
      if (!inOrder) {
        found.push({ axes: to, sources, reason: "reordered" });
        break;
      }
    }
  }
  return found;
}

const REGROUP = new Set([
  "reshape",
  "view",
  "flatten",
  "unflatten",
  "squeeze",
  "unsqueeze",
  "ravel",
  "view_as",
  "reshape_as",
  "contiguous",
  "clone",
]);

/** Right-aligned operand axis that supplies an output axis of a given size. */
function aligned(
  operands: { shape: number[]; stories: AxisStory[] }[],
  rank: number,
  axis: number,
  size: number,
): AxisStory | null {
  for (const operand of operands) {
    const at = axis - (rank - operand.shape.length);
    if (at >= 0 && operand.shape[at] === size && size !== 1)
      return operand.stories[at];
  }
  return null;
}

/** Stories for the outputs of one operation, or null if it explains nothing. */
function through(
  op: Operation,
  outputIndex: number,
  output: Tensor,
  inputs: Tensor[],
  of: (tensor: Tensor) => AxisStory[],
): AxisStory[] | null {
  const lesson = op.lesson;
  const input = inputs[0];
  if (!input || !lesson) return null;
  const rank = output.shape.length;
  const order = lesson.axis_order;
  if (
    order &&
    order.length === rank &&
    order.every((from, to) => input.shape[from] === output.shape[to])
  )
    return order.map((from) => of(input)[from]);
  if (lesson.mapping_rule === "identity" && REGROUP.has(op.kind))
    return regroup(input.shape, output.shape, of(input));
  if (lesson.mapping_rule === "roll") return of(input);

  const relation = tensorRelation(op, inputs, output, { anyLesson: true });
  if (relation && outputIndex === 0) {
    const source = inputs[relation.operand];
    const stories = of(source);
    switch (relation.rule) {
      case "elementwise":
      case "prefix":
      case "channel_affine": {
        const operands = inputs.map((tensor) => ({
          shape: tensor.shape,
          stories: of(tensor),
        }));
        return output.shape.map(
          (size, axis) =>
            aligned(operands, rank, axis, size) ??
            fresh(size, size === 1 ? "an axis of size 1" : "broadcast"),
        );
      }
      case "pad":
        return stories.map((story, axis) =>
          output.shape[axis] === story.size
            ? story
            : { ...story, size: output.shape[axis], note: "padded" },
        );
      case "reduce": {
        const kept: AxisStory[] = [];
        stories.forEach((story, axis) => {
          if (!relation.axes.includes(axis)) kept.push(story);
          else if (relation.keepdim)
            kept.push(fresh(1, `reduced by ${op.kind}`));
        });
        return kept;
      }
      case "index": {
        const result = output.shape.map((size) =>
          fresh(size, "inserted by indexing"),
        );
        relation.axes.forEach((item, axis) => {
          if (item.kind === "slice")
            result[item.out] =
              output.shape[item.out] === stories[axis].size
                ? stories[axis]
                : {
                    ...stories[axis],
                    size: output.shape[item.out],
                    note: "sliced",
                  };
        });
        return result;
      }
      case "tile": {
        const offset = rank - source.shape.length;
        return output.shape.map((size, axis) => {
          if (axis < offset) return fresh(size, `new axis from ${op.kind}`);
          const story = stories[axis - offset];
          return size === story.size
            ? story
            : { ...story, size, note: `repeated ×${size / story.size}` };
        });
      }
      case "one_hot":
        return [
          ...stories,
          fresh(output.shape[rank - 1], "one position per class"),
        ];
      case "einsum": {
        const lettered = relation.inputs.map((letters, i) => ({
          letters,
          stories: of(inputs[i]),
        }));
        return [...relation.output].map((letter, axis) => {
          for (const operand of lettered) {
            const at = operand.letters.indexOf(letter);
            if (at >= 0) return operand.stories[at];
          }
          return fresh(output.shape[axis], `einsum letter ${letter}`);
        });
      }
      case "table":
        if (op.kind === "embedding" && inputs[1])
          return [...of(inputs[0]), of(inputs[1])[inputs[1].shape.length - 1]];
        return null;
      default:
        return null;
    }
  }
  if (lesson.interaction === "dot_product" && inputs[1] && rank >= 2) {
    const [left, right] = inputs;
    const operands = [left, right].map((tensor) => ({
      shape: tensor.shape.slice(0, -2),
      stories: of(tensor).slice(0, -2),
    }));
    return output.shape.map((size, axis) =>
      axis === rank - 2
        ? of(left)[left.shape.length - 2]
        : axis === rank - 1
          ? of(right)[right.shape.length - 1]
          : (aligned(operands, rank - 2, axis, size) ??
            fresh(size, "batch broadcast")),
    );
  }
  if (lesson.interaction === "linear_projection" && inputs[1]) {
    const weight = of(inputs[1]);
    return [...of(input).slice(0, -1), weight[0]];
  }
  if (
    ["normalization", "layer_normalization", "broadcast_add"].includes(
      lesson.interaction,
    )
  ) {
    const operands = inputs.map((tensor) => ({
      shape: tensor.shape,
      stories: of(tensor),
    }));
    return output.shape.map(
      (size, axis) =>
        aligned(operands, rank, axis, size) ?? fresh(size, "broadcast"),
    );
  }
  if (lesson.interaction === "tensor_assembly") {
    const raw = Number(op.arguments.dim ?? 0);
    if (op.kind === "stack") {
      const at = ((raw % rank) + rank) % rank;
      const stories = [...of(input)];
      stories.splice(at, 0, fresh(output.shape[at], "one per stacked tensor"));
      return stories;
    }
    if (["cat", "concat", "concatenate"].includes(op.kind)) {
      const at = ((raw % rank) + rank) % rank;
      return of(input).map((story, axis) =>
        axis === at
          ? {
              size: output.shape[axis],
              terms: [],
              note: `${inputs.length} parts joined by ${op.kind}`,
            }
          : story,
      );
    }
    if (input.shape.length === rank)
      return of(input).map((story, axis) =>
        output.shape[axis] === story.size
          ? story
          : {
              ...story,
              size: output.shape[axis],
              note: `part cut by ${op.kind}`,
            },
      );
  }
  return null;
}

/** Axis stories for one recorded tensor. */
export function axisLineage(
  trace: Run["trace"],
  tensorId: string,
): AxisStory[] {
  return lineageOf(trace)(tensorId);
}

/**
 * A lookup that shares its work across every tensor of one trace. `roots`
 * gives stories to start from for chosen tensors, so lineage can be traced
 * from any point, not only from inputs and parameters.
 */
export function lineageOf(
  trace: Run["trace"],
  roots?: ReadonlyMap<string, AxisStory[]>,
): (tensorId: string) => AxisStory[] {
  const producers = new Map<string, { op: Operation; index: number }>();
  for (const op of trace.operations)
    producedTensorIds(op).forEach((id) => {
      if (!producers.has(id))
        producers.set(id, { op, index: (op.outputs ?? []).indexOf(id) });
    });
  const memo = new Map<string, AxisStory[]>();
  const of = (tensor: Tensor): AxisStory[] => {
    const known = memo.get(tensor.id) ?? roots?.get(tensor.id);
    if (known) return known;
    // A cycle cannot occur in a recorded trace, but guard against bad data.
    memo.set(tensor.id, root(tensor));
    const producer = producers.get(tensor.id);
    let stories = root(tensor);
    const inputs = (producer?.op.inputs ?? [])
      .map((id) => trace.tensors[id])
      .filter(Boolean);
    // An in-place write keeps every axis where it was: the new state has the
    // axes of the state it overwrote.
    const written = producer?.op.mutations?.find(
      (mutation) => mutation.after === tensor.id && mutation.kind === "write",
    );
    const overwritten = written ? trace.tensors[written.before] : null;
    if (
      producer &&
      overwritten &&
      overwritten.id !== tensor.id &&
      overwritten.shape.join() === tensor.shape.join()
    ) {
      stories = of(overwritten);
      memo.set(tensor.id, stories);
      return stories;
    }
    // A tensor created from nothing, like torch.ones(5, 8), is its own source.
    if (
      producer &&
      inputs.length &&
      producer.op.status === "ok" &&
      producer.index >= 0
    ) {
      const traced = through(producer.op, producer.index, tensor, inputs, of);
      stories =
        traced && traced.length === tensor.shape.length
          ? traced.map((story, axis) => carry(story, tensor.shape[axis]))
          : tensor.shape.map((size) =>
              fresh(size, `computed by ${producer.op.kind}`),
            );
    }
    memo.set(tensor.id, stories);
    return stories;
  };
  return (tensorId) => {
    const tensor = trace.tensors[tensorId];
    return tensor ? of(tensor) : [];
  };
}

/** One readable line per axis, such as "x.tokens × x.features". */
export function describeAxis(story: AxisStory): string {
  if (!story.terms.length) return story.note ?? "unknown";
  const text = story.terms
    .map((term) =>
      term.part
        ? `${term.label} (piece ${term.part.index + 1} of ${term.part.sizes.join("×")})`
        : term.label,
    )
    .join(" × ");
  return story.note ? `${text}, ${story.note}` : text;
}

/** True when every axis is simply the tensor's own: nothing to explain. */
export function isOwnLineage(tensor: Tensor, stories: AxisStory[]): boolean {
  const own = root(tensor);
  return stories.every(
    (story, axis) =>
      story.terms.length === 1 &&
      !story.terms[0].part &&
      !story.note &&
      story.terms[0].label === own[axis]?.terms[0].label,
  );
}

/** Axis stories worth showing, or null when the tensor is simply its own source. */
export function explainedLineage(
  tensor: Tensor,
  stories: AxisStory[] | null | undefined,
): AxisStory[] | null {
  return stories &&
    stories.length === tensor.shape.length &&
    tensor.shape.length > 0 &&
    !isOwnLineage(tensor, stories)
    ? stories
    : null;
}

/**
 * A term's label cut to badge size: `tokens.tokens` (input `tokens`, axis
 * `tokens`) is just `tokens`, and a weight keeps the end of its module path,
 * `text_encoder.block.qkv.weight axis 0` → `qkv.weight axis 0`.
 */
function compactLabel(term: AxisTerm): string {
  const dot = term.label.lastIndexOf(".");
  const spaced = term.label.match(/^(.*) (axis \d+)$/);
  const [owner, axis, joiner] = spaced
    ? [spaced[1], spaced[2], " "]
    : dot > 0
      ? [term.label.slice(0, dot), term.label.slice(dot + 1), "."]
      : [term.label, "", ""];
  if (owner === axis) return axis;
  const path = owner.split(".");
  // Keep the last two parts, plus any parent a bare index needs to mean
  // something: `experts.0.2.weight`, not `2.weight`.
  let start = Math.max(0, path.length - 2);
  while (start > 0 && /^\d+$/.test(path[start])) start--;
  const short = term.role === "parameter" ? path.slice(start).join(".") : owner;
  return axis ? `${short}${joiner}${axis}` : short;
}

/**
 * Whether an origin only repeats the axis's own name: `batch ← x.batch`. The
 * axis already says what it is, so a badge would add nothing. A piece of an
 * axis, a note, or a differently named source still says something.
 */
export function restatesAxis(story: AxisStory, axisName: string): boolean {
  if (story.terms.length !== 1 || story.note) return false;
  const [term] = story.terms;
  return (
    !term.part &&
    !/^axis \d+$/.test(axisName) &&
    (term.label === axisName || term.label.endsWith(`.${axisName}`))
  );
}

/** A badge-sized origin, such as "x.features[1/2]" or "x.rows×x.columns". */
export function shortAxis(story: AxisStory): string | null {
  if (!story.terms.length) return null;
  return story.terms
    .map((term) =>
      term.part
        ? `${compactLabel(term)}[${term.part.index + 1}/${term.part.of}]`
        : compactLabel(term),
    )
    .join("×");
}

/**
 * A one-line origin for a whole tensor: each axis's badge origin, with runs
 * of the same origin said once ("computed by conv2d", not four times).
 */
export function originSummary(
  stories: AxisStory[],
  names: string[] = [],
): string {
  const parts = stories
    .filter((story, axis) => !restatesAxis(story, names[axis] ?? ""))
    .map((story) => shortAxis(story) ?? story.note ?? "unknown");
  return parts
    .filter((part, axis) => axis === 0 || part !== parts[axis - 1])
    .join(" · ");
}

const termKey = (term: AxisTerm) =>
  term.part ? `${term.label}#${term.part.index}/${term.part.of}` : term.label;

/**
 * Two axes that a product sums against each other but that hold different
 * things: both are traced to model inputs, and they are not the same axis.
 * A weight axis (x @ W) is never judged, because weights are made to match.
 */
export function unrelatedAxes(a: AxisStory, b: AxisStory): boolean {
  const known = (story: AxisStory) =>
    story.terms.length > 0 &&
    story.terms.every((term) => term.role === "input");
  if (!known(a) || !known(b)) return false;
  const key = (story: AxisStory) => rejoin(story.terms).map(termKey).join("×");
  return key(a) !== key(b);
}

/** An axis that is, untouched, an input's axis named batch (after any moves). */
export function isBatchAxis(story: AxisStory | undefined): boolean {
  const term = story?.terms.length === 1 ? story.terms[0] : null;
  return (
    !!term &&
    !term.part &&
    !story!.note &&
    term.role === "input" &&
    /\.batch\w*$/i.test(term.label)
  );
}

/**
 * Whether a tensor belongs to a trace: the trace's own object, or a copy of
 * it (lessons relabel axes with `{ ...tensor, axes }`). Ids repeat across
 * traces, so a copy must also match name, shape, dtype, and storage.
 */
export function inTrace(trace: Run["trace"], tensor: Tensor): boolean {
  const own = trace.tensors[tensor.id];
  return (
    !!own &&
    (own === tensor ||
      (own.name === tensor.name &&
        own.dtype === tensor.dtype &&
        own.storage_id === tensor.storage_id &&
        own.shape.length === tensor.shape.length &&
        own.shape.every((size, axis) => size === tensor.shape[axis])))
  );
}

/** One piece of where a result axis comes from: an input axis, or a piece of one. */
export type AxisPart = {
  /** The input axis. */
  axis: number;
  size: number;
  /** For a piece of a split input axis: which, out of how many. */
  piece?: { index: number; of: number };
};

/**
 * Where each axis of `toId` comes from among the axes of `fromId`, through
 * the steps between them: for each result axis, the input axes (or pieces
 * of them) it is made of, in order. Null when some axis cannot be traced
 * back to the input, such as one computed rather than rearranged.
 */
export function axisMap(
  trace: Run["trace"],
  fromId: string,
  toId: string,
): AxisPart[][] | null {
  const from = trace.tensors[fromId];
  if (!from || !trace.tensors[toId]) return null;
  const start: AxisStory[] = from.shape.map((size, axis) => ({
    size,
    terms: [{ label: `#${axis}`, size }],
    note: null,
  }));
  const stories = lineageOf(trace, new Map([[fromId, start]]))(toId);
  const map: AxisPart[][] = [];
  for (const story of stories) {
    if (story.size === 1 && !story.terms.length) {
      map.push([]);
      continue;
    }
    const parts: AxisPart[] = [];
    for (const term of story.terms) {
      const match = /^#(\d+)$/.exec(term.label);
      if (!match) return null;
      parts.push({
        axis: Number(match[1]),
        size: term.size,
        ...(term.part
          ? { piece: { index: term.part.index, of: term.part.of } }
          : {}),
      });
    }
    if (!parts.length) return null;
    map.push(parts);
  }
  return map;
}
