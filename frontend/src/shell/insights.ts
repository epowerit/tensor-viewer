import type { LoopStep, Operation, Run, Tensor } from "../api/client";
import { alignShapes } from "../operations/relations";
import { entryPath } from "../sources/files";
import { producedTensorIds } from "../tensors/provenance";
import {
  describeAxis,
  isBatchAxis,
  lineageOf,
  scrambledAxes,
  unrelatedAxes,
  type AxisStory,
} from "../tensors/axisLineage";
import type { Problem } from "./problems";

/**
 * A linter for tensors. Each rule reads only the recorded trace — shapes,
 * axis names, data types, storage, and inline values — and points at the step
 * that deserves a second look. None of them is an error: the code ran.
 */

const ELEMENTWISE = new Set([
  "add",
  "sub",
  "rsub",
  "__rsub__",
  "mul",
  "div",
  "true_divide",
  "__rdiv__",
  "__rtruediv__",
  "floor_divide",
  "remainder",
  "pow",
  "maximum",
  "minimum",
  "eq",
  "ne",
  "gt",
  "ge",
  "lt",
  "le",
  "where",
  "logical_and",
  "logical_or",
  "__and__",
  "__or__",
]);
const REDUCTIONS = new Set([
  "sum",
  "mean",
  "prod",
  "amax",
  "amin",
  "max",
  "min",
  "argmax",
  "argmin",
  "std",
  "var",
  "logsumexp",
]);
// Operations that only regroup the same elements into a new shape.
const RESHAPES = new Set([
  "reshape",
  "view",
  "flatten",
  "unflatten",
  "ravel",
  "view_as",
  "reshape_as",
]);
const ZEROING = new Set(["relu", "relu_", "threshold", "hardtanh"]);
const NON_FINITE = new Set([
  "nan",
  "inf",
  "-inf",
  "NaN",
  "Infinity",
  "-Infinity",
]);

const shape = (tensor: Tensor) => `[${tensor.shape.join(", ")}]`;
const nonFinite = (value: unknown) =>
  (typeof value === "string" && NON_FINITE.has(value)) ||
  (typeof value === "number" && !Number.isFinite(value));
const hasNaN = (tensor: Tensor) =>
  tensor.values?.some(
    (value) =>
      (typeof value === "string" && /nan/i.test(value)) ||
      (typeof value === "number" && Number.isNaN(value)),
  ) ?? false;
const hasNonFinite = (tensor: Tensor) =>
  (tensor.values?.length ?? -1) === tensor.numel &&
  tensor.values.some(
    (value) =>
      (typeof value === "string" && NON_FINITE.has(value)) ||
      (typeof value === "number" && !Number.isFinite(value)),
  );
const axisList = (axes: number[]) =>
  axes.length === 1 ? `axis ${axes[0]}` : `axes ${axes.join(", ")}`;
const normalize = (dim: unknown, rank: number) =>
  typeof dim === "number" && Number.isInteger(dim) && dim >= -rank && dim < rank
    ? (dim + rank) % rank
    : null;
/** Reduced axes from recorded arguments, or null when they are not explicit. */
function reducedAxes(op: Operation, rank: number): number[] | null {
  const dim = op.arguments.dim;
  if (dim === undefined || dim === null) return null;
  const dims = Array.isArray(dim) ? dim : [dim];
  const axes = dims.map((d) => normalize(d, rank));
  return axes.every((axis) => axis !== null) ? (axes as number[]) : null;
}

/**
 * Features of an activation's result that are zero at every position: along
 * an axis named for channels, or the last axis. Null when there are none, or
 * too few positions per feature (under 2) to say.
 */
export function deadUnits(tensor: Tensor): {
  axis: number;
  size: number;
  per: number;
  units: number[];
} | null {
  const rank = tensor.shape.length;
  if (rank < 2 || tensor.values?.length !== tensor.numel) return null;
  const named = tensor.axes.findIndex((name) => /channel/i.test(name ?? ""));
  const axis = named >= 0 ? named : rank - 1;
  const size = tensor.shape[axis];
  const per = tensor.numel / size;
  if (size < 2 || per < 2) return null;
  const inner = tensor.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  const alive = new Array<boolean>(size).fill(false);
  tensor.values.forEach((value, at) => {
    if (value !== 0) alive[Math.floor(at / inner) % size] = true;
  });
  const units = alive.flatMap((live, unit) => (live ? [] : [unit]));
  return units.length ? { axis, size, per, units } : null;
}

/** ln of the largest finite value: exp overflows past this. */
const EXP_LIMIT: Record<string, number> = {
  float32: 88.72,
  float64: 709.78,
  bfloat16: 88.72,
  float16: 11.09,
};
const FLOAT16_MAX = 65504;
/** Decimal digits a float carries. */
const DIGITS: Record<string, number> = {
  float32: 7.2,
  float64: 15.9,
  bfloat16: 2.4,
  float16: 3.3,
};

const extent = (tensor: Tensor | undefined) => {
  if (!tensor) return null;
  const low = tensor.minimum ?? tensor.histogram?.low;
  const high = tensor.maximum ?? tensor.histogram?.high;
  return typeof low === "number" && typeof high === "number"
    ? { low, high, largest: Math.max(Math.abs(low), Math.abs(high)) }
    : null;
};
const short = (value: number) =>
  Math.abs(value) >= 1e4 || (value !== 0 && Math.abs(value) < 1e-3)
    ? value.toExponential(1)
    : String(Number(value.toPrecision(3)));

/**
 * Steps whose values sit near where float arithmetic breaks: an exp close to
 * overflow, a log of values close to 0, a subtraction that cancels most of its
 * digits, a division by values close to 0, a softmax so sharp its gradients
 * vanish, and float16 values near its largest. Read from recorded ranges, and
 * from inline values where a rule needs every value.
 */
function stability(
  op: Operation,
  inputs: Tensor[],
  outputs: Tensor[],
): Finding[] {
  const found: Finding[] = [];
  const kind = op.kind.replace(/^__|__$|_$/g, "");
  const input = inputs[0],
    output = outputs[0];
  const range = extent(input);
  const floating = (tensor: Tensor | undefined) =>
    !!tensor && /^(b?float)/.test(tensor.dtype);

  if (/^(exp|expm1)$/.test(kind) && range && floating(output)) {
    const limit = EXP_LIMIT[output!.dtype];
    if (limit && range.high > limit - 8 && range.high <= limit)
      found.push({
        rule: "exp-near-overflow",
        severity: "warning",
        title: `${kind} is close to overflowing ${output!.dtype}`,
        detail: `Its input reaches ${short(range.high)}, and exp overflows ${output!.dtype} past ${limit}: a little larger and this step returns ∞. Subtracting the largest value first (as softmax and logsumexp do) keeps exp in range.`,
      });
  }

  if (/^(log|log2|log10)$/.test(kind) && range && range.low > 0) {
    if (range.low < 1e-12)
      found.push({
        rule: "log-near-zero",
        severity: "info",
        title: `${kind} of values close to 0`,
        detail: `Its input goes down to ${short(range.low)}, where ${kind} reads ${short(Math.log(range.low) / (kind === "log2" ? Math.LN2 : kind === "log10" ? Math.LN10 : 1))}: one value of 0 would make it −∞. A small floor (x.clamp_min(1e-9)) or log_softmax keeps it finite.`,
      });
  }

  // A Python number subtracted (x - 1e6) is an argument, not an input.
  const scalar = Number(op.arguments?.other);
  const withScalar = inputs.length === 1 && Number.isFinite(scalar);
  if (
    /^(sub|add)$/.test(kind) &&
    (inputs.length >= 2 || withScalar) &&
    floating(output)
  ) {
    const ins = inputs.map(extent),
      out = extent(output);
    const largest = Math.max(
      ...ins.map((each) => each?.largest ?? 0),
      withScalar ? Math.abs(scalar) : 0,
    );
    if (out && largest > 1 && out.largest > 0 && out.largest < largest * 1e-4) {
      const lost = Math.log10(largest / out.largest);
      const left = Math.max(0, (DIGITS[output!.dtype] ?? 7.2) - lost);
      found.push({
        rule: "cancellation",
        severity: left < 2 ? "warning" : "info",
        title: "Nearly equal values cancel",
        detail: `The operands reach ${short(largest)} but the result only ${short(out.largest)}, so about ${lost.toFixed(1)} of ${output!.dtype}'s ${DIGITS[output!.dtype] ?? 7.2} significant digits cancel and about ${left.toFixed(1)} are left. Subtracting before the values grow, or computing in float64, keeps the digits.`,
      });
    }
  }

  if (/^(div|true_divide)$/.test(kind) && inputs[1] && floating(output)) {
    const divisor = inputs[1];
    const values =
      divisor.values?.length === divisor.numel
        ? divisor.values.filter(
            (value): value is number => typeof value === "number",
          )
        : null;
    const smallest = values?.length
      ? Math.min(...values.filter((value) => value !== 0).map(Math.abs))
      : null;
    if (smallest !== null && Number.isFinite(smallest) && smallest < 1e-6) {
      const out = extent(output);
      found.push({
        rule: "tiny-divisor",
        severity: "warning",
        title: "Dividing by values close to 0",
        detail: `${divisor.name} goes down to ${short(smallest)} in magnitude, so the quotient reaches ${out ? short(out.largest) : "a very large value"}. Adding a small eps to the divisor (as normalizations do) keeps it bounded.`,
      });
    }
  }

  if (
    kind === "softmax" &&
    output &&
    output.values?.length === output.numel &&
    output.shape.length
  ) {
    const width = output.shape.at(-1)!;
    const rows = output.numel / width;
    let sharp = 0;
    for (let row = 0; row < rows; row++) {
      let top = 0;
      for (let at = row * width; at < (row + 1) * width; at++) {
        const value = output.values[at];
        if (typeof value === "number" && value > top) top = value;
      }
      if (top > 0.999) sharp++;
    }
    if (width > 1 && rows >= 2 && sharp >= rows * 0.9)
      found.push({
        rule: "saturated-softmax",
        severity: "info",
        title: "softmax is saturated",
        detail: `${sharp} of ${rows} rows put more than 99.9% on one entry. A softmax this sharp passes almost no gradient to the rest, and its scores span ${range ? short(range.high - range.low) : "a wide range"}: scaling them down (÷√d, or a temperature) softens it.`,
      });
  }

  if (output?.dtype === "float16") {
    const out = extent(output);
    if (out && out.largest > FLOAT16_MAX / 8 && out.largest <= FLOAT16_MAX)
      found.push({
        rule: "float16-range",
        severity: "warning",
        title: "Close to float16's largest value",
        detail: `${output.name} reaches ${short(out.largest)}, within a factor of ${(FLOAT16_MAX / out.largest).toFixed(1)} of float16's largest finite value, 65504. A little more and it overflows to ∞; bfloat16 keeps float32's range.`,
      });
  }
  return found;
}

type Finding = Pick<Problem, "severity" | "title" | "detail"> & {
  rule: string;
};

function examine(
  op: Operation,
  inputs: Tensor[],
  outputs: Tensor[],
  trace: Run["trace"],
  lineage: (tensorId: string) => AxisStory[],
): Finding[] {
  const findings: Finding[] = [];
  const output = outputs[0];
  if (op.status !== "ok") return findings;

  // A matrix product that sums one input axis against a different one.
  if (
    output &&
    op.lesson?.interaction === "dot_product" &&
    inputs.length >= 2 &&
    inputs[0].shape.length >= 2 &&
    inputs[1].shape.length >= 2
  ) {
    const [left, right] = inputs;
    const leftAxes = lineage(left.id),
      rightAxes = lineage(right.id);
    const summed = leftAxes[left.shape.length - 1],
      against = rightAxes[right.shape.length - 2];
    if (summed && against && unrelatedAxes(summed, against)) {
      const rightLast = rightAxes[right.shape.length - 1];
      const transposed = rightLast && !unrelatedAxes(summed, rightLast);
      findings.push({
        rule: "unrelated-contraction",
        severity: "warning",
        title: `${op.kind} sums ${describeAxis(summed)} against ${describeAxis(against)}`,
        detail: `${left.name} ${shape(left)} @ ${right.name} ${shape(right)} multiplies along the last axis of ${left.name} (${describeAxis(summed)}) and the second-to-last axis of ${right.name} (${describeAxis(against)}). Their sizes agree (${summed.size}), but they hold different things, so each result adds up unrelated products.${transposed ? ` The last axis of ${right.name} is ${describeAxis(rightLast)}: ${right.name}.transpose(-2, -1) would pair matching axes.` : ""}`,
      });
    }
  }

  // A reshape that interleaves elements of different source axes.
  if (output && RESHAPES.has(op.kind) && inputs[0]) {
    const source = inputs[0];
    const [scrambled] = scrambledAxes(
      source.shape,
      output.shape,
      lineage(source.id),
    );
    if (scrambled) {
      const names = [
        ...new Set(scrambled.sources.map((text) => text.split(" piece")[0])),
      ];
      const where =
        scrambled.axes.length === 1
          ? `Axis ${scrambled.axes[0]}`
          : `Axes ${scrambled.axes.join(", ")}`;
      findings.push({
        rule: "scrambled-axes",
        severity: "warning",
        title:
          names.length > 1
            ? `${op.kind} mixes ${names.join(" and ")}`
            : `${op.kind} rebuilds ${names[0]} out of order`,
        detail:
          scrambled.reason === "regrouped"
            ? `${where} of ${output.name} ${shape(output)} cut across ${source.name}'s axes ${scrambled.sources.join(", ")}, so neighboring elements come from different positions of ${names.join(" and ")}. This usually means a permute or transpose was not undone: permute ${source.name} back so each axis is whole and in order, then reshape.`
            : `${where} of ${output.name} merges ${scrambled.sources.join(", ")}: the pieces of ${names.find((name) => scrambled.sources.some((text) => text.startsWith(`${name} piece`)))} are not next to each other in their original order, so the merged axis is not the original one. Permute the pieces back together, in order, before merging.`,
      });
    }
  }

  // Two operands each stretched along an axis the other has: an outer combination.
  if (output && ELEMENTWISE.has(op.kind) && inputs.length >= 2) {
    const rows = alignShapes(
      inputs.map((tensor) => tensor.shape),
      output.shape,
    );
    const stretched = rows?.map((row) =>
      row.flatMap((axis, i) => (axis.stretched ? [i] : [])),
    );
    // Size-1 axes written out on both sides (rows[:, None] * columns[None, :])
    // ask for a table on purpose; the surprise is an axis that was missing.
    const implicit = rows?.some((row) =>
      row.some((axis) => axis.stretched && axis.size === null),
    );
    const crossing =
      implicit &&
      stretched &&
      stretched.some((own, i) =>
        stretched.some(
          (other, j) =>
            j !== i &&
            own.length > 0 &&
            other.length > 0 &&
            own.some((axis) => !other.includes(axis)),
        ),
      );
    if (
      crossing &&
      output.numel > Math.max(...inputs.map((tensor) => tensor.numel))
    ) {
      const [a, b] = inputs;
      findings.push({
        rule: "outer-broadcast",
        severity: "warning",
        title: "Both operands were stretched",
        detail: `${a.name} ${shape(a)} and ${b.name} ${shape(b)} combine into ${shape(output)}: each is repeated along an axis the other has, which pairs every value of one with every value of the other. If you meant one result per position, give them the same shape first.`,
      });
    }
  }

  // The first step whose values are not finite, when its inputs were. An
  // infinity the step was asked to write, like masked_fill(mask, -inf) before
  // a softmax, is deliberate unless a NaN came with it.
  const asked =
    Object.values(op.arguments ?? {}).some(nonFinite) && !outputs.some(hasNaN);
  if (
    !asked &&
    outputs.some(hasNonFinite) &&
    inputs.every(
      (tensor) =>
        (tensor.values?.length ?? -1) === tensor.numel && !hasNonFinite(tensor),
    )
  ) {
    const cause =
      op.kind === "div" || op.kind === "true_divide"
        ? " A division by zero is the usual cause."
        : op.kind === "log"
          ? " The logarithm of zero is −∞ and of a negative number is NaN."
          : op.kind === "sqrt" || op.kind === "rsqrt"
            ? " The square root of a negative number is NaN."
            : op.kind === "pow"
              ? " A power can overflow or take a root of a negative number."
              : "";
    findings.push({
      rule: "non-finite",
      severity: "warning",
      title: "NaN or infinity first appears here",
      detail: `${output?.name ?? op.kind} contains values that are not finite, and none of this step's inputs did.${cause} Every later step that reads it carries them along.`,
    });
  }

  // A reduction over axes that only have one position changes nothing.
  if (output && REDUCTIONS.has(op.kind) && inputs[0]) {
    const source = inputs[0];
    const axes = reducedAxes(op, source.shape.length);
    if (axes?.length && axes.every((axis) => source.shape[axis] === 1))
      findings.push({
        rule: "trivial-reduction",
        severity: "info",
        title: `${op.kind} over a size-1 axis`,
        detail: `${source.name} ${shape(source)} has one position along ${axisList(axes)}, so ${op.kind} combines a single value. Check that dim names the axis you meant to reduce.`,
      });
    else if (axes?.length && output.shape.some((size) => size > 1)) {
      const stories = lineage(source.id);
      const batch = axes.find(
        (axis) => source.shape[axis] > 1 && isBatchAxis(stories[axis]),
      );
      if (batch !== undefined)
        findings.push({
          rule: "batch-reduction",
          severity: "info",
          title: `${op.kind} combines examples of the batch`,
          detail: `${op.kind} over axis ${batch} of ${source.name} ${shape(source)} (${describeAxis(stories[batch])}) combines different examples while keeping ${shape(output)} per position, so each result describes the whole batch rather than one example. That is right for batch statistics; for a per-example result, reduce the other axes instead.`,
        });
    }
  }

  // Softmax over one position always gives 1 (or 0 for log_softmax).
  if (
    output &&
    (op.kind === "softmax" || op.kind === "log_softmax") &&
    inputs[0]
  ) {
    const axis = normalize(op.arguments.dim, inputs[0].shape.length);
    if (axis !== null && inputs[0].shape[axis] === 1)
      findings.push({
        rule: "trivial-softmax",
        severity: "warning",
        title: `${op.kind} over a size-1 axis`,
        detail: `Axis ${axis} of ${inputs[0].name} ${shape(inputs[0])} has one position, so every weight is ${op.kind === "softmax" ? "1" : "0"} whatever the scores are. Softmax usually runs over the axis that holds the alternatives being compared.`,
      });
    else if (
      axis !== null &&
      inputs[0].shape[axis] > 1 &&
      isBatchAxis(lineage(inputs[0].id)[axis])
    )
      findings.push({
        rule: "batch-softmax",
        severity: "warning",
        title: `${op.kind} across the batch`,
        detail: `Axis ${axis} of ${inputs[0].name} ${shape(inputs[0])} is ${describeAxis(lineage(inputs[0].id)[axis])}, so the weights compare different examples with each other and every example's result depends on the rest of the batch. Softmax usually runs over the axis that holds the alternatives (classes, keys); check which axis dim names after any transpose or permute.`,
      });
  }

  // squeeze() without dim also removes a batch axis that happens to have size 1.
  if (
    output &&
    op.kind === "squeeze" &&
    op.arguments.dim === undefined &&
    inputs[0]
  ) {
    const lost = (inputs[0].axes ?? []).flatMap((name, axis) =>
      /batch/i.test(name) && inputs[0].shape[axis] === 1 ? [name] : [],
    );
    if (lost.length)
      findings.push({
        rule: "squeezed-batch",
        severity: "warning",
        title: "squeeze() removed the batch axis",
        detail: `${inputs[0].name} ${shape(inputs[0])} has a batch of one, and squeeze() without dim drops every size-1 axis, including ${lost.join(", ")}. With a larger batch the result would have a different rank; squeeze(dim) removes only the axis you name.`,
      });
  }

  // An activation that left nothing alive.
  if (
    output &&
    ZEROING.has(op.kind) &&
    (output.values?.length ?? -1) === output.numel &&
    output.numel > 1
  ) {
    const zeros = output.values.filter((value) => value === 0).length;
    if (zeros === output.numel)
      findings.push({
        rule: "all-zero",
        severity: "warning",
        title: `${op.kind} produced only zeros`,
        detail: `Every value reaching ${op.kind} was at or below its threshold, so ${output.name} is entirely zero and carries no information forward.`,
      });
    else if (zeros >= output.numel * 0.9)
      findings.push({
        rule: "mostly-zero",
        severity: "info",
        title: `${op.kind} zeroed ${Math.round((zeros / output.numel) * 100)}% of values`,
        detail: `${zeros} of ${output.numel} values in ${output.name} are zero after ${op.kind}.`,
      });
    const dead = zeros < output.numel ? deadUnits(output) : null;
    if (dead) {
      const name = output.axes[dead.axis] || `axis ${dead.axis}`;
      const listed = dead.units.slice(0, 6);
      findings.push({
        rule: "dead-units",
        severity: dead.units.length * 4 >= dead.size ? "warning" : "info",
        title: `${dead.units.length} of ${dead.size} ${name} never activate`,
        detail: `After ${op.kind}, ${output.name} is zero along ${name} at ${dead.units.length === 1 ? "index" : "indices"} ${listed.join(", ")}${dead.units.length > listed.length ? ` and ${dead.units.length - listed.length} more` : ""}, at every one of the ${dead.per} positions this run recorded. A unit that is zero everywhere passes no gradient back either (a dead ReLU); with more varied inputs it may wake, but a bias driven far negative or a learning rate that was too high keeps it dead.`,
      });
    }
  }

  // Numerical stability: trouble a few steps before it becomes NaN or ∞.
  findings.push(...stability(op, inputs, outputs));

  // Precision silently widened to float64.
  const floats = inputs.filter((tensor) => tensor.dtype?.startsWith("float"));
  if (
    output?.dtype === "float64" &&
    floats.some((tensor) => tensor.dtype !== "float64") &&
    floats.some((tensor) => tensor.dtype === "float64")
  )
    findings.push({
      rule: "float64",
      severity: "info",
      title: "Promoted to float64",
      detail: `${output.name} is float64 because one operand was. Mixing it with float32 tensors doubles memory from here on; .float() keeps everything float32.`,
    });

  // An in-place write that reaches the model's own input.
  const inputStorages = new Set(
    (trace.input_ids ?? []).map((id) => trace.tensors[id]?.storage_id),
  );
  const written = (op.mutations ?? []).find(
    (mutation) =>
      mutation.kind === "write" &&
      inputStorages.has(trace.tensors[mutation.before]?.storage_id),
  );
  if (written) {
    const original = (trace.input_ids ?? [])
      .map((id) => trace.tensors[id])
      .find(
        (tensor) =>
          tensor?.storage_id === trace.tensors[written.before]?.storage_id,
      );
    findings.push({
      rule: "input-written",
      severity: "warning",
      title: "This write changes the input",
      detail: `${op.kind} writes into storage that ${original?.name ?? "an input"} shares${trace.tensors[written.before]?.name !== original?.name ? ` (through ${trace.tensors[written.before]?.name})` : ""}. Code that reads the input afterwards sees the new values; clone() first to keep the original.`,
    });
  }
  return findings;
}

/** Results nobody reads: neither a later step nor the model's output. */
/** Steps whose results nothing reads and the model does not return. */
export function unused(trace: Run["trace"]): Operation[] {
  const read = new Set(trace.output_ids ?? []);
  trace.operations.forEach((op) => {
    (op.inputs ?? []).forEach((id) => read.add(id));
    op.mutations?.forEach((mutation) => read.add(mutation.before));
  });
  // A failed run stops early; what came before the failure was not abandoned.
  if (trace.error) return [];
  return trace.operations.filter((op) => {
    const produced = producedTensorIds(op);
    return (
      op.status === "ok" &&
      (op.outputs ?? []).length > 0 &&
      !op.mutations?.length &&
      produced.every((id) => !read.has(id)) &&
      // Storage shared with something read later is not dead.
      produced.every(
        (id) =>
          ![...read].some(
            (other) =>
              trace.tensors[other]?.storage_id ===
                trace.tensors[id]?.storage_id && other !== id,
          ),
      )
    );
  });
}

/** Lint a recorded run, or a shape check of the current code. */
export function tensorInsights(run: Run): Problem[] {
  const trace = run.trace;
  const entry = entryPath(run.project);
  const located = (op: Operation) => ({
    node: op.id,
    file: op.source?.file ?? entry,
    line: op.source?.line ?? null,
    diagnosis: null,
  });
  const findings: Problem[] = [];
  // Built on first use: most traces have no reshape worth tracing.
  let lookup: ((tensorId: string) => AxisStory[]) | null = null;
  const lineage = (tensorId: string) => (lookup ??= lineageOf(trace))(tensorId);
  for (const op of trace.operations) {
    const inputs = (op.inputs ?? [])
      .map((id) => trace.tensors[id])
      .filter(Boolean);
    const outputs = (op.outputs ?? [])
      .map((id) => trace.tensors[id])
      .filter(Boolean);
    for (const finding of examine(op, inputs, outputs, trace, lineage))
      findings.push({
        id: `insight-${finding.rule}-${op.id}`,
        severity: finding.severity,
        title: finding.title,
        detail: finding.detail,
        ...located(op),
      });
  }
  for (const drift of loopDrift(trace)) {
    const last = drift.passes.at(-1)!;
    findings.push({
      id: `insight-loop-drift-${last.op.id}`,
      severity: "warning",
      title: `${last.name} ${drift.grows ? "grows" : "shrinks"} every pass of ${drift.step.text}`,
      detail: `The value this loop carries into its next pass, ${last.name}, reaches a largest magnitude of ${drift.passes.map((pass) => formatMagnitude(pass.magnitude)).join(" → ")} over ${drift.passes.length} passes (${drift.grows ? "×" : "÷"}${formatMagnitude(drift.grows ? drift.ratio : 1 / drift.ratio)}). ${drift.grows ? "With more passes it can overflow or saturate later steps; a normalization or a scaled residual inside the body usually keeps it steady." : "With more passes it can fade toward zero and stop carrying signal; check scaling, gating, or a missing residual connection."}`,
      node: last.op.id,
      file: drift.step.file ?? entry,
      line: drift.step.line,
      diagnosis: null,
    });
  }
  for (const op of unused(trace)) {
    const names = (op.outputs ?? []).map((id) => trace.tensors[id]?.name ?? id);
    findings.push({
      id: `insight-unused-${op.id}`,
      severity: "info",
      title: `Computed but never used: ${names.join(", ")}`,
      detail: `No later tensor operation reads ${names.length === 1 ? "this result" : "these results"}, and ${names.length === 1 ? "it is" : "they are"} not returned, so ${names.length === 1 ? "it does" : "they do"} not affect the output. Uses outside tensor operations, such as an if condition, print, or .item(), are not recorded.`,
      ...located(op),
    });
  }
  return findings;
}

const formatMagnitude = (value: number) =>
  value >= 1000 || (value > 0 && value < 0.01)
    ? value.toExponential(1)
    : value.toFixed(2);

/**
 * Loops whose carried value (each pass's last result) grows or shrinks at
 * least fourfold, steadily, from the first pass to the last. Needs values.
 */
export function loopDrift(trace: Run["trace"]) {
  const runs = new Map<
    string,
    { step: LoopStep; passes: Map<number, Operation> }
  >();
  for (const op of trace.operations)
    for (const step of op.loops ?? []) {
      const run = runs.get(step.id) ?? { step, passes: new Map() };
      // The last operation of each pass is the one recorded last.
      run.passes.set(step.iteration, op);
      runs.set(step.id, run);
    }
  return [...runs.values()].flatMap(({ step, passes }) => {
    const ordered = [...passes.entries()]
      .sort(([a], [b]) => a - b)
      .map(([, op]) => {
        const tensor = trace.tensors[producedTensorIds(op)[0]];
        const low = tensor?.minimum,
          high = tensor?.maximum;
        return typeof low === "number" &&
          typeof high === "number" &&
          Number.isFinite(low) &&
          Number.isFinite(high)
          ? {
              op,
              name: tensor.name,
              magnitude: Math.max(Math.abs(low), Math.abs(high)),
            }
          : null;
      });
    if (ordered.length < 2 || ordered.some((pass) => !pass)) return [];
    const list = ordered as {
      op: Operation;
      name: string;
      magnitude: number;
    }[];
    const first = list[0].magnitude,
      last = list.at(-1)!.magnitude;
    if (!first || !last) return [];
    const ratio = last / first;
    const steady = (grows: boolean) =>
      list.every(
        (pass, i) =>
          !i ||
          (grows
            ? pass.magnitude >= list[i - 1].magnitude * 0.98
            : pass.magnitude <= list[i - 1].magnitude * 1.02),
      );
    if (ratio >= 4 && steady(true))
      return [{ step, passes: list, ratio, grows: true }];
    if (ratio <= 0.25 && steady(false))
      return [{ step, passes: list, ratio, grows: false }];
    return [];
  });
}
