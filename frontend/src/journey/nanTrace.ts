import type { Operation, Run, Tensor } from "../api/client";
import { asNumber } from "../tensors/margins";
import { traceCellContributors } from "./cellContributors";

type Trace = Run["trace"];

/** A cell on the way back: the value it holds and the step that wrote it. */
export type NanCell = {
  tensorId: string;
  index: number;
  value: number;
  /** The step that wrote it; null for an input or a parameter. */
  opId: string | null;
};

export type NanTrace = {
  /** From the chosen cell back to the origin, the origin last. */
  cells: NanCell[];
  /** The values the origin was computed from, as far as they are recorded. */
  from: { tensorId: string; index: number; value: number; role: string }[];
  /** Why the origin's value is not finite, in a sentence. */
  cause: string;
  /** An infinite cell that a NaN was made from, to trace in its turn. */
  infinity: NanCell | null;
  /**
   * Somewhere a cell relation was not recorded and the trail went on through
   * the first non-finite value of the step's input instead.
   */
  approximate: boolean;
  /** Set when the trail went cold before finding where the value began. */
  stopped: string | null;
};

/** Values of cells of a tensor; null when they cannot be read. */
export type ReadValues = (
  tensorId: string,
  indices: number[],
) => Promise<(number | string | undefined)[] | null>;

const isNaNValue = (value: number) => Number.isNaN(value);
const infinite = (value: number) =>
  !Number.isNaN(value) && !Number.isFinite(value);

/** "NaN", "+∞", "−∞", or the number. */
export function nonFiniteText(value: number) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Infinity) return "+∞";
  if (value === -Infinity) return "−∞";
  return String(Number(value.toPrecision(4)));
}

/** Largest finite float32. */
const FLOAT32_MAX = 3.4028234663852886e38;

/**
 * Why a step made a value that is not finite from the values it read, in
 * words: log(0), 0 / 0, an exp that overflows, ∞ − ∞.
 */
export function nanCause(
  op: Operation,
  value: number,
  from: { value: number; role: string }[],
): string {
  const kind = op.kind.toLowerCase().replace(/^__|__$|_$/g, "");
  const values = from.map((each) => each.value);
  const finite = values.filter(Number.isFinite);
  const zero = finite.includes(0);
  const negative = finite.some((each) => each < 0);
  const largest = finite.reduce(
    (top, each) => Math.max(top, Math.abs(each)),
    0,
  );
  const result = nonFiniteText(value);
  const fromInfinity = values.some(infinite);
  if (op.kind === "masked_fill" || op.kind === "fill_" || op.kind === "full") {
    const fill = asNumber(op.arguments?.value as number | string | undefined);
    if (!Number.isFinite(fill))
      return `${op.kind} writes ${nonFiniteText(fill)} here on purpose, where the mask is set`;
  }
  if (isNaNValue(value) && fromInfinity) {
    if (/softmax/.test(kind))
      return "every score in this row is −∞, so softmax divides 0 by 0";
    if (/^(add|sub|subtract|sum|mean|cumsum)$/.test(kind))
      return "∞ − ∞ has no value: infinities of both signs met in a sum";
    if (/^(mul|multiply|matmul|bmm|mm|linear|einsum)$/.test(kind) && zero)
      return "0 · ∞ has no value";
    if (/^(div|true_divide)$/.test(kind)) return "∞ / ∞ has no value";
    return `${kind} turned an infinity into NaN`;
  }
  if (/^log/.test(kind)) {
    if (zero && value === -Infinity) return "log(0) is −∞";
    if (negative) return "the logarithm of a negative number is NaN";
  }
  if (/^sqrt$/.test(kind) && negative)
    return "the square root of a negative number is NaN";
  if (/^rsqrt$/.test(kind)) {
    if (negative) return "1 / √x of a negative number is NaN";
    if (zero) return "1 / √0 is +∞";
  }
  if (/^(div|true_divide|reciprocal|floor_divide)$/.test(kind)) {
    const divisor =
      kind === "reciprocal"
        ? from[0]
        : from.find((each) => each.role === "other");
    const dividend = from.find((each) => each.role === "input");
    if (divisor?.value === 0 || (!divisor && zero))
      return dividend?.value === 0 || (isNaNValue(value) && zero)
        ? "0 / 0 has no value"
        : `dividing by 0 gives ${result}`;
  }
  if (/^pow$/.test(kind) && negative && isNaNValue(value))
    return "a negative number to a fractional power is NaN";
  if (/^(exp|expm1|softmax|log_softmax)$/.test(kind) && largest > 88)
    return `exp(${String(Number(largest.toPrecision(3)))}) is beyond float32's largest value (3.4·10³⁸)`;
  if (/^(acos|asin)$/.test(kind) && finite.some((each) => Math.abs(each) > 1))
    return `${kind} of a value beyond ±1 is NaN`;
  if (infinite(value) && largest > Math.sqrt(FLOAT32_MAX))
    return `${kind} of values this large (${largest.toExponential(1)}) is beyond float32's largest value (3.4·10³⁸)`;
  if (infinite(value))
    return `${kind} made ${result}: the result is beyond float32's range`;
  return `${kind} made ${result} from finite values`;
}

/** The step that wrote each tensor state. */
function writers(trace: Trace) {
  const made = new Map<string, Operation>();
  for (const op of trace.operations) {
    op.outputs.forEach((id) => made.set(id, op));
    op.mutations?.forEach((mutation) => made.set(mutation.after, op));
  }
  return made;
}

/** The first cell of a tensor that matches, when its values are inline. */
function firstWhere(tensor: Tensor, wanted: (value: number) => boolean) {
  if (tensor.values?.length !== tensor.numel) return null;
  const index = tensor.values.findIndex((value) => wanted(asNumber(value)));
  return index < 0 ? null : index;
}

/** Steps the trail follows before giving up, to bound a pathological graph. */
const HOPS = 512;

/**
 * Follows a NaN or an infinity back to where it began: through each step's
 * exact cell relation, to the cell it was computed from that is not finite
 * either, until a step made it from finite values (or an input already held
 * it). A NaN follows NaNs; when a step made a NaN from infinities, that step
 * is the NaN's origin and the infinity is offered to trace in its turn.
 */
export async function traceNonFinite(
  trace: Trace,
  tensorId: string,
  index: number,
  read: ReadValues,
): Promise<NanTrace> {
  const made = writers(trace);
  const run = { trace } as Run;
  const result: NanTrace = {
    cells: [],
    from: [],
    cause: "",
    infinity: null,
    approximate: false,
    stopped: null,
  };
  const [first] = (await read(tensorId, [index])) ?? [];
  let at: NanCell = {
    tensorId,
    index,
    value: asNumber(first),
    opId: made.get(tensorId)?.id ?? null,
  };
  if (Number.isFinite(at.value) || first === undefined) {
    result.stopped = "This value is finite.";
    return result;
  }
  const looksFor = isNaNValue(at.value) ? isNaNValue : infinite;
  for (let hop = 0; hop < HOPS; hop++) {
    result.cells.push(at);
    const op = made.get(at.tensorId);
    const tensor = trace.tensors[at.tensorId];
    if (!op) {
      result.cause =
        tensor?.role === "parameter"
          ? `the parameter ${tensor.name} already holds ${nonFiniteText(at.value)}`
          : `the input ${tensor?.name ?? ""} already holds ${nonFiniteText(at.value)}`;
      return result;
    }
    const contributors = traceCellContributors(run, op, at.tensorId, at.index);
    let sources: NanTrace["from"] = [];
    if (contributors.status === "mapped") {
      const byTensor = new Map<string, number[]>();
      for (const source of contributors.sources)
        byTensor.set(source.tensorId, [
          ...(byTensor.get(source.tensorId) ?? []),
          source.index,
        ]);
      const fetched = new Map<string, (number | string | undefined)[] | null>();
      for (const [id, indices] of byTensor)
        fetched.set(id, await read(id, indices));
      const position = new Map<string, number>();
      sources = contributors.sources.flatMap((source) => {
        const nth = position.get(source.tensorId) ?? 0;
        position.set(source.tensorId, nth + 1);
        const value = fetched.get(source.tensorId)?.[nth];
        return value === undefined
          ? []
          : [{ ...source, value: asNumber(value) }];
      });
    }
    const same = sources.find((source) => looksFor(source.value));
    if (same) {
      at = {
        tensorId: same.tensorId,
        index: same.index,
        value: same.value,
        opId: made.get(same.tensorId)?.id ?? null,
      };
      continue;
    }
    // Not every source was read: go on through the first matching value of
    // an input, which may not be this cell's own.
    const unread =
      contributors.status !== "mapped" ||
      contributors.truncated ||
      sources.length < contributors.sources.length;
    if (unread) {
      const inputs = [
        ...op.inputs,
        ...(op.mutations ?? []).map((mutation) => mutation.before),
      ];
      let next: NanCell | null = null;
      for (const id of inputs) {
        const input = trace.tensors[id];
        if (!input || (input.histogram?.non_finite ?? 1) === 0) continue;
        const found = firstWhere(input, looksFor);
        if (found !== null) {
          next = {
            tensorId: id,
            index: found,
            value: asNumber(input.values[found]),
            opId: made.get(id)?.id ?? null,
          };
          break;
        }
      }
      if (next) {
        result.approximate = true;
        at = next;
        continue;
      }
      // An input whose values are not inline might hold it: say so.
      const unknown = inputs.some((id) => {
        const input = trace.tensors[id];
        return (
          !!input &&
          input.values?.length !== input.numel &&
          (input.histogram?.non_finite ?? 1) > 0
        );
      });
      if (unknown) {
        result.stopped = `The values ${op.kind} read are not all recorded here, so the trail stops at this step.`;
        result.cause = `${op.kind} wrote ${nonFiniteText(at.value)}; one of its inputs may already have held it`;
        return result;
      }
    }
    // This step made it.
    result.from = sources.slice(0, 8);
    result.cause = nanCause(op, at.value, sources);
    const infinity = isNaNValue(at.value)
      ? sources.find((source) => infinite(source.value))
      : undefined;
    if (infinity)
      result.infinity = {
        tensorId: infinity.tensorId,
        index: infinity.index,
        value: infinity.value,
        opId: made.get(infinity.tensorId)?.id ?? null,
      };
    return result;
  }
  result.stopped = `The trail is longer than ${HOPS} steps.`;
  return result;
}
