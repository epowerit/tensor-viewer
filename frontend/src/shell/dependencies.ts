import type {
  Dependencies,
  DependencyRequest,
  Run,
  Tensor,
} from "../api/client";
import { entryPath } from "../sources/files";
import { lineageOf, rootLabel } from "../tensors/axisLineage";
import type { Problem } from "./problems";

/**
 * Who sees whom: which input positions each position of a result reads,
 * held to the pattern the model promises. Examples of a batch read only
 * themselves (the diagonal); under causal attention a position reads only
 * itself and those before it (the lower triangle). The backend proves the
 * reads with gradients; this module chooses what to ask and what to expect,
 * and turns a breach into a run note on its step.
 */

export type Pattern = NonNullable<DependencyRequest["expect"]>;

/** What the model shows of itself that makes a position read only the past. */
export type CausalEvidence = {
  /** "the causal mask blocks.0.attention.causal" */
  label: string;
  tensorId: string;
  /** The positions it covers; null for a mask longer than any sequence. */
  size: number | null;
};

const GENERIC = /^axis \d+$/;
const BATCH = /^batch/i;
const SEQUENCE = /token|position|sequence|time|word|quer|frame/i;

const number = (value: unknown) => {
  if (typeof value === "number") return value;
  if (typeof value !== "string") return NaN;
  if (/^-inf/i.test(value)) return -Infinity;
  if (/^\+?inf/i.test(value)) return Infinity;
  return Number(value);
};
const inline = (tensor: Tensor) =>
  tensor.value_source !== "shape" &&
  tensor.value_source !== "paged" &&
  (tensor.values?.length ?? -1) === tensor.numel;
const floating = (tensor: Tensor) => /float|half|double/.test(tensor.dtype);

/** Each square plane of a tensor's last two axes, as rows of numbers. */
function planes(tensor: Tensor): number[][][] | null {
  const n = tensor.shape.at(-1) ?? 0;
  if (
    tensor.shape.length < 2 ||
    n < 2 ||
    tensor.shape.at(-2) !== n ||
    !inline(tensor)
  )
    return null;
  const values = tensor.values.map(number);
  return Array.from({ length: tensor.numel / (n * n) }, (_, plane) =>
    Array.from({ length: n }, (_, row) =>
      values.slice(plane * n * n + row * n, plane * n * n + (row + 1) * n),
    ),
  );
}

/**
 * A mask that sets the future apart: one value strictly above the diagonal
 * and another on and below it (a `tril` of ones, its `~`, or an additive
 * mask of −∞ and 0).
 */
export function isTriangularMask(tensor: Tensor): boolean {
  const found = planes(tensor);
  if (!found) return false;
  let upper: number | undefined;
  let lower: number | undefined;
  for (const plane of found)
    for (const [row, values] of plane.entries())
      for (const [column, value] of values.entries()) {
        if (Number.isNaN(value)) return false;
        if (column > row) {
          if (upper === undefined) upper = value;
          else if (value !== upper) return false;
        } else if (lower === undefined) lower = value;
        else if (value !== lower) return false;
      }
  return upper !== undefined && lower !== undefined && upper !== lower;
}

/** Attention weights where no query gives any weight to a later key. */
export function isTriangularMap(tensor: Tensor): boolean {
  const found = planes(tensor);
  if (!found) return false;
  let past = false;
  for (const plane of found)
    for (const [row, values] of plane.entries())
      for (const [column, value] of values.entries()) {
        if (column > row && value !== 0) return false;
        if (column < row && value !== 0) past = true;
      }
  return past;
}

/** The run's evidence of causal attention: mask buffers, then attention maps. */
export function causalEvidence(trace: Run["trace"]): CausalEvidence[] {
  const buffers = new Set(trace.buffer_names ?? []);
  const found: CausalEvidence[] = Object.values(trace.tensors)
    .filter(
      (tensor) =>
        tensor.role === "parameter" &&
        buffers.has(tensor.name) &&
        isTriangularMask(tensor),
    )
    .map((tensor) => ({
      label: `the causal mask ${tensor.name}`,
      tensorId: tensor.id,
      size: null,
    }));
  for (const op of trace.operations) {
    const tensor = op.kind === "softmax" ? trace.tensors[op.outputs[0]] : null;
    if (tensor && isTriangularMap(tensor))
      found.push({
        label: `the attention weights ${tensor.name} at step ${op.index + 1}, which give no later word any weight`,
        tensorId: tensor.id,
        size: tensor.shape.at(-1)!,
      });
  }
  return found;
}

const primaryInput = (trace: Run["trace"]) =>
  trace.tensors[trace.input_ids[0] ?? ""] ?? null;

/**
 * The input axis holding the same positions as a result's axis: the one of
 * the same name, the one its lineage traces back to, or the one of the same
 * size. Null when none is clear.
 */
export function inputAxisFor(
  trace: Run["trace"],
  tensor: Tensor,
  axis: number,
  lineage: (id: string) => { terms: { label: string; part?: unknown }[] }[] = (
    id,
  ) => lineageOf(trace)(id),
): number | null {
  const input = primaryInput(trace);
  if (!input) return null;
  const name = tensor.axes[axis] ?? "";
  if (!GENERIC.test(name) && input.axes.filter((a) => a === name).length === 1)
    return input.axes.indexOf(name);
  const story = lineage(tensor.id)[axis];
  const term = story?.terms.length === 1 ? story.terms[0] : null;
  if (term && !term.part) {
    const at = input.axes.findIndex(
      (_, each) => rootLabel(input, each) === term.label,
    );
    if (at >= 0) return at;
  }
  const same = input.shape.flatMap((size, each) =>
    size === tensor.shape[axis] ? [each] : [],
  );
  return same.length === 1 ? same[0] : null;
}

/** The pattern a result's axis is held to, and why; null for none. */
export function expectedPattern(
  trace: Run["trace"],
  tensor: Tensor,
  axis: number,
  inputAxis: number | null,
  evidence: CausalEvidence[] = causalEvidence(trace),
): { pattern: Pattern; why: string } | null {
  const input = primaryInput(trace);
  const names = [
    tensor.axes[axis] ?? "",
    inputAxis !== null && input ? (input.axes[inputAxis] ?? "") : "",
  ].filter((name) => name && !GENERIC.test(name));
  if (names.some((name) => BATCH.test(name)))
    return {
      pattern: "diagonal",
      why: "the examples of a batch should never read each other",
    };
  const fits = evidence.filter(
    (each) => each.size === null || each.size === tensor.shape[axis],
  );
  if (fits.length && names.some((name) => SEQUENCE.test(name)))
    return {
      pattern: "causal",
      why: `${fits[0].label} lets each position read only itself and those before it`,
    };
  return null;
}

/** Whether output slice `output` may read input slice `input`. */
export function allowed(
  pattern: Pattern | null | undefined,
  output: number,
  input: number,
): boolean {
  if (pattern === "diagonal") return output === input;
  if (pattern === "causal") return input <= output;
  return true;
}

/** Every cell read against the pattern, as [output, input], row by row. */
export function violationsOf(
  reads: boolean[][],
  pattern: Pattern | null | undefined,
): [number, number][] {
  return reads.flatMap((row, output) =>
    row.flatMap((read, input): [number, number][] =>
      read && !allowed(pattern, output, input) ? [[output, input]] : [],
    ),
  );
}

export type DependencyAsk = DependencyRequest & {
  /** Why the pattern is expected, for the panel and the note. */
  why: string | null;
};

/** A result's axes that line up with an input axis, with what they expect. */
export function askableAxes(
  trace: Run["trace"],
  tensor: Tensor,
  evidence: CausalEvidence[] = causalEvidence(trace),
): DependencyAsk[] {
  if (!floating(tensor) || tensor.role !== "intermediate") return [];
  const lineage = lineageOf(trace);
  return tensor.shape.flatMap((size, axis) => {
    if (size < 1) return [];
    const inputAxis = inputAxisFor(trace, tensor, axis, lineage);
    if (inputAxis === null) return [];
    const expected = expectedPattern(trace, tensor, axis, inputAxis, evidence);
    return [
      {
        tensor_id: tensor.id,
        axis,
        input_axis: inputAxis,
        expect: expected?.pattern ?? null,
        why: expected?.why ?? null,
      },
    ];
  });
}

/** The results that can be asked about, in the order they were made. */
export function askableResults(trace: Run["trace"]): Tensor[] {
  const seen = new Set<string>();
  return trace.operations.flatMap((op) =>
    op.outputs.flatMap((id) => {
      const tensor = trace.tensors[id];
      if (!tensor || seen.has(id) || !floating(tensor)) return [];
      seen.add(id);
      return tensor.role === "intermediate" ? [tensor] : [];
    }),
  );
}

const rank = (ask: DependencyAsk, trace: Run["trace"]) => {
  const size = trace.tensors[ask.tensor_id].shape[ask.axis];
  if (ask.expect === "causal") return 3;
  if (ask.expect === "diagonal" && size > 1) return 2;
  return ask.expect ? 1 : 0;
};

/**
 * What to ask first: about the selected step's result when it lines up with
 * the input, else the latest result that does; its axis with the most to
 * prove (a causal one, then a batch of more than one example).
 */
export function defaultAsk(
  trace: Run["trace"],
  selected: string | null = null,
): DependencyAsk | null {
  const evidence = causalEvidence(trace);
  const results = askableResults(trace);
  const step = selected
    ? trace.operations.find((op) => op.id === selected)
    : null;
  const ordered = [
    ...(step ? results.filter((t) => step.outputs.includes(t.id)) : []),
    ...[...results].reverse(),
  ];
  for (const tensor of ordered) {
    const asks = askableAxes(trace, tensor, evidence);
    if (!asks.length) continue;
    return asks.reduce((best, ask) =>
      rank(ask, trace) > rank(best, trace) ? ask : best,
    );
  }
  return null;
}

const unit = (pattern: Pattern | null | undefined) =>
  pattern === "diagonal" ? "example" : "position";
const capital = (text: string) => text[0].toUpperCase() + text.slice(1);

/** "position 3 (“time”)", with the word at a text input's position. */
export function positionName(
  pattern: Pattern | null | undefined,
  at: number,
  words?: string[] | null,
) {
  const word = pattern !== "diagonal" ? words?.[at] : undefined;
  return `${unit(pattern)} ${at}${word ? ` (“${word}”)` : ""}`;
}

/** "0 reads 1, 2, 3, …; 1 reads 2, 3, …", for the first few outputs. */
export function readList(violations: [number, number][], outputs = 3) {
  const byOutput = new Map<number, number[]>();
  for (const [output, input] of violations)
    byOutput.set(output, [...(byOutput.get(output) ?? []), input]);
  const shown = [...byOutput].slice(0, outputs).map(([output, inputs]) => {
    const listed = inputs.slice(0, 4).join(", ");
    return `${output} reads ${listed}${inputs.length > 4 ? ", …" : ""}`;
  });
  return `${shown.join("; ")}${byOutput.size > outputs ? "; …" : ""}`;
}

/**
 * A breach of the expected pattern as a run note pinned to the first step
 * whose result breaks it: "Position 0 reads position 1".
 */
export function dependencyNotes(
  run: Run,
  ask: DependencyAsk,
  answer: Dependencies,
  words?: string[] | null,
): Problem[] {
  const pattern = answer.expect ?? ask.expect;
  if (!pattern) return [];
  const reads = answer.reads ?? [];
  const violations = violationsOf(reads, pattern);
  if (!violations.length) return [];
  const tensor = run.trace.tensors[answer.tensor_id];
  if (!tensor) return [];
  const breach = answer.breach ?? null;
  const [output, input] = breach
    ? [breach.output, breach.input]
    : violations[0];
  const step = breach
    ? run.trace.operations.find((op) => op.id === breach.operation_id)
    : undefined;
  const before = breach?.after
    ? run.trace.operations.find((op) => op.id === breach.after)
    : undefined;
  const axisName = GENERIC.test(tensor.axes[answer.axis] ?? "")
    ? `axis ${answer.axis}`
    : tensor.axes[answer.axis];
  const rule =
    pattern === "diagonal"
      ? `read each example along ${axisName} only from itself`
      : `read each position along ${axisName} only from itself and the positions before it`;
  const cells = `${violations.length} ${violations.length === 1 ? "cell reads" : "cells read"} ${pattern === "diagonal" ? "another example" : "a later position"}`;
  const where = step
    ? ` The first step whose result reads it is step ${step.index + 1}, ${step.kind}${step.source ? ` (line ${step.source.line})` : ""}${before ? `; step ${before.index + 1}, ${before.kind}, before it still kept the pattern` : ", the first after the input"}.`
    : "";
  return [
    {
      id: `dependencies-${answer.tensor_id}-${answer.axis}`,
      severity: "warning",
      title: `${capital(positionName(pattern, output, words))} reads ${positionName(pattern, input, words)}`,
      detail: `${tensor.name} [${tensor.shape.join(", ")}] should ${rule}${ask.why ? `, since ${ask.why}` : ""}. Gradients show ${cells}: ${readList(violations)}.${where}`,
      file: step?.source ? (step.source.file ?? entryPath(run.project)) : null,
      line: step?.source?.line ?? null,
      node: step?.id ?? null,
      diagnosis: null,
    },
  ];
}
