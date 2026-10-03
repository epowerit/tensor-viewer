import type { LineResult } from "../console/script";
import type { Tensor } from "../api/client";
import { asNumber } from "../tensors/margins";
import { formatValue } from "../tensors/coordinates";
import type { ContractCheck } from "./contracts";

/**
 * Value contracts: clauses in a line's comment that state what its tensor's
 * values must be, checked against the recorded run like shape contracts.
 *
 *   probs = scores.softmax(-1)   # range: 0..1; sums(-1): 1
 *   h = layer(x)                 # finite; mean: 0 ± 0.5
 *
 * `range: a..b` bounds every finite value (either end may be left open),
 * `finite` forbids NaN and infinity, `sums(axis): v` requires every sum over
 * that axis to be v, `mean: m ± t` bounds the mean, `std: a..b` the spread,
 * `zeros: ..50%` the share of exact zeros (dead activations), and
 * `dtype: float32` the type. Clauses are separated by `;` and may share a
 * comment with `shape:`.
 */
export type ValueClause =
  | { kind: "range"; low: number | null; high: number | null }
  | { kind: "finite" }
  | { kind: "sums"; axis: number; value: number }
  | { kind: "mean"; value: number; tolerance: number }
  | { kind: "std"; low: number | null; high: number | null }
  | { kind: "zeros"; low: number | null; high: number | null }
  | { kind: "dtype"; dtype: string };

const NUMBER = String.raw`-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?`;
const RANGE = new RegExp(
  String.raw`^range:\s*(${NUMBER})?\s*\.\.\s*(${NUMBER})?$`,
);
const SUMS = new RegExp(String.raw`^sums\(\s*(-?\d+)\s*\):\s*(${NUMBER})$`);
const BOUNDS = new RegExp(
  String.raw`^(std|zeros):\s*(${NUMBER})?(%?)\s*\.\.\s*(${NUMBER})?(%?)$`,
);
const DTYPE = /^dtype:\s*(?:torch\.)?([a-z]+\d*)$/;
const MEAN = new RegExp(
  String.raw`^mean:\s*(${NUMBER})(?:\s*(?:±|\+-|\+/-)\s*(${NUMBER}))?$`,
);

/** The value clauses of a line's comment, an error, or null with none. */
export function parseValueContract(
  source: string,
  line: number,
):
  | { line: number; clauses: ValueClause[]; text: string }
  | { line: number; error: string }
  | null {
  const hash = source.indexOf("#");
  if (hash < 0) return null;
  const parts = source
    .slice(hash + 1)
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean);
  const clauses: ValueClause[] = [];
  const texts: string[] = [];
  for (const part of parts) {
    // Only the contract forms count, so a comment such as "mean pooling
    // over tokens" or "finite differences" stays a comment.
    const keyword =
      part === "finite"
        ? "finite"
        : /^(range|mean|std|zeros|dtype):|^(sums)\(/
            .exec(part)
            ?.slice(1)
            .find(Boolean);
    if (!keyword) continue;
    let match: RegExpExecArray | null;
    if (keyword === "finite") clauses.push({ kind: "finite" });
    else if (
      keyword === "range" &&
      (match = RANGE.exec(part)) &&
      (match[1] || match[2])
    )
      clauses.push({
        kind: "range",
        low: match[1] === undefined ? null : Number(match[1]),
        high: match[2] === undefined ? null : Number(match[2]),
      });
    else if (keyword === "sums" && (match = SUMS.exec(part)))
      clauses.push({
        kind: "sums",
        axis: Number(match[1]),
        value: Number(match[2]),
      });
    else if (
      (keyword === "std" || keyword === "zeros") &&
      (match = BOUNDS.exec(part)) &&
      (match[2] || match[4])
    ) {
      // A share of zeros may be written as a fraction or a percentage.
      const bound = (text: string | undefined, percent: string) =>
        text === undefined
          ? null
          : Number(text) /
            (percent || (keyword === "zeros" && Number(text) > 1) ? 100 : 1);
      clauses.push({
        kind: keyword,
        low: bound(match[2], match[3]),
        high: bound(match[4], match[5]),
      });
    } else if (keyword === "dtype" && (match = DTYPE.exec(part)))
      clauses.push({ kind: "dtype", dtype: match[1] });
    else if (keyword === "mean" && (match = MEAN.exec(part)))
      clauses.push({
        kind: "mean",
        value: Number(match[1]),
        tolerance: match[2] === undefined ? 0 : Math.abs(Number(match[2])),
      });
    else
      return {
        line,
        error: `“${part}” is not a value contract: try range: 0..1, finite, sums(-1): 1, mean: 0 ± 0.1, std: 0.5..2, zeros: ..50%, or dtype: float32.`,
      };
    texts.push(part);
  }
  return clauses.length ? { line, clauses, text: texts.join("; ") } : null;
}

/** Every sum over `axis`, when every value is inline. */
function sumsOver(tensor: Tensor, axis: number) {
  if (tensor.values.length !== tensor.numel) return null;
  const rank = tensor.shape.length;
  const along = axis < 0 ? rank + axis : axis;
  if (along < 0 || along >= rank) return undefined;
  const size = tensor.shape[along];
  const inner = tensor.shape.slice(along + 1).reduce((a, b) => a * b, 1);
  const sums = new Array(tensor.numel / size).fill(0);
  tensor.values.forEach((value, flat) => {
    const outer = Math.floor(flat / (size * inner));
    sums[outer * inner + (flat % inner)] += asNumber(value);
  });
  return sums as number[];
}

/** Why a tensor breaks a clause, or null when it keeps it (or cannot be told). */
/** The backend's answer to a sums contract on a tensor kept in a snapshot. */
export type SumsAnswer = {
  count: number;
  off: number;
  worst: number | string | null;
};

/** The key a sums contract's backend answer is kept under. */
export const sumsKey = (tensorId: string, axis: number, value: number) =>
  `${tensorId}|${axis}|${value}`;

function breach(
  tensor: Tensor,
  clause: ValueClause,
  answers?: ReadonlyMap<string, SumsAnswer>,
): string | null | undefined {
  const name = tensor.name;
  if (clause.kind === "finite") {
    const broken = tensor.histogram?.non_finite;
    if (broken === undefined) return undefined;
    return broken
      ? `${name} holds ${broken.toLocaleString()} NaN or infinite values.`
      : null;
  }
  if (clause.kind === "range") {
    const { minimum, maximum } = tensor;
    if (typeof minimum !== "number" || typeof maximum !== "number")
      return undefined;
    if (clause.low !== null && minimum < clause.low)
      return `${name} reaches ${formatValue(minimum)}, below ${clause.low}.`;
    if (clause.high !== null && maximum > clause.high)
      return `${name} reaches ${formatValue(maximum)}, above ${clause.high}.`;
    return null;
  }
  if (clause.kind === "dtype") {
    const dtype = tensor.dtype.replace(/^torch\./, "");
    return dtype === clause.dtype
      ? null
      : `${name} is ${dtype}, not ${clause.dtype}.`;
  }
  if (clause.kind === "std" || clause.kind === "zeros") {
    const histogram = tensor.histogram;
    if (!histogram) return undefined;
    const total =
      histogram.counts.reduce((sum, count) => sum + count, 0) +
      histogram.non_finite;
    const value =
      clause.kind === "std"
        ? histogram.std
        : total
          ? histogram.zeros / total
          : null;
    if (typeof value !== "number") return undefined;
    const shown = (x: number) =>
      clause.kind === "zeros"
        ? `${Math.round(x * 1000) / 10}%`
        : formatValue(x);
    const what = clause.kind === "std" ? `${name}'s σ is` : `${name} is`;
    const unit = clause.kind === "zeros" ? " zeros" : "";
    if (clause.low !== null && value < clause.low)
      return `${what} ${shown(value)}${unit}, below ${shown(clause.low)}.`;
    if (clause.high !== null && value > clause.high)
      return `${what} ${shown(value)}${unit}, above ${shown(clause.high)}.`;
    return null;
  }
  if (clause.kind === "mean") {
    // As in torch, any NaN or infinity makes the mean itself not finite.
    const broken = tensor.histogram?.non_finite ?? 0;
    if (broken)
      return `${name} holds ${broken.toLocaleString()} NaN or infinite values, so its mean is not a number near ${clause.value}.`;
    const mean = tensor.histogram?.mean;
    if (typeof mean !== "number") return undefined;
    // Float32 rounding is not a breach.
    const slack = clause.tolerance + 1e-6 * Math.max(1, Math.abs(clause.value));
    return Math.abs(mean - clause.value) > slack
      ? `${name}'s mean is ${formatValue(mean)}, not ${clause.value}${clause.tolerance ? ` ± ${clause.tolerance}` : ""}.`
      : null;
  }
  const sums = sumsOver(tensor, clause.axis);
  if (sums === null) {
    // Values kept in a snapshot: the backend adds them up.
    const answer = answers?.get(sumsKey(tensor.id, clause.axis, clause.value));
    if (!answer) return undefined;
    return answer.off
      ? `${answer.off.toLocaleString()} of ${answer.count.toLocaleString()} sums over axis ${clause.axis} are not ${clause.value}; one is ${formatValue(asNumber(answer.worst ?? undefined))}.`
      : null;
  }
  if (sums === undefined) return `${name} has no axis ${clause.axis}.`;
  const tolerance = 1e-4 * Math.max(1, Math.abs(clause.value));
  const off = sums.filter(
    (sum) => !(Math.abs(sum - clause.value) <= tolerance),
  );
  if (!off.length) return null;
  const worst = off.reduce((a, b) =>
    Math.abs(b - clause.value) > Math.abs(a - clause.value) || Number.isNaN(b)
      ? b
      : a,
  );
  return `${off.length.toLocaleString()} of ${sums.length.toLocaleString()} sums over axis ${clause.axis} are not ${clause.value}; one is ${formatValue(worst)}.`;
}

/**
 * Check every value contract against the line's recorded tensor. A line
 * without fresh recorded values (stale, failed, or a shape check) is skipped,
 * and so is a clause the recording cannot answer, such as sums over a tensor
 * whose values stay in a snapshot.
 */
export function checkValueContracts(
  lines: string[],
  results: ReadonlyMap<number, LineResult>,
  answers?: ReadonlyMap<string, SumsAnswer>,
): ContractCheck[] {
  const checks: ContractCheck[] = [];
  lines.forEach((source, index) => {
    const line = index + 1;
    const contract = parseValueContract(source, line);
    if (!contract) return;
    if ("error" in contract) {
      checks.push({ line, text: "", ok: false, message: contract.error });
      return;
    }
    const result = results.get(line);
    const tensor =
      result?.fresh && !result.error && !result.predicted
        ? result.output
        : null;
    if (!tensor || tensor.value_source === "shape") return;
    const outcomes = contract.clauses.map((clause) =>
      breach(tensor, clause, answers),
    );
    if (outcomes.every((outcome) => outcome === undefined)) return;
    const failed = outcomes.filter((outcome): outcome is string => !!outcome);
    checks.push({
      line,
      text: contract.text,
      ok: !failed.length,
      message: failed.length
        ? failed.join(" ")
        : `${tensor.name} keeps ${contract.text}.`,
    });
  });
  return checks;
}

/** One mark per line: a line keeps its contracts only if it keeps all of them. */
export function mergeChecks(...lists: ContractCheck[][]) {
  const merged = new Map<number, ContractCheck>();
  for (const check of lists.flat()) {
    const earlier = merged.get(check.line);
    merged.set(
      check.line,
      earlier
        ? {
            line: check.line,
            text: [earlier.text, check.text].filter(Boolean).join("; "),
            ok: earlier.ok && check.ok,
            // A broken line explains only what it breaks.
            message: [earlier, check]
              .filter((item) => earlier.ok === check.ok || !item.ok)
              .map((item) => item.message)
              .join(" "),
          }
        : check,
    );
  }
  return merged;
}

/** A bound rounded outward to two significant digits, so it still holds. */
function outward(value: number, up: boolean) {
  if (value === 0 || !Number.isFinite(value)) return value;
  const scale = 10 ** (Math.floor(Math.log10(Math.abs(value))) - 1);
  return Number(
    ((up ? Math.ceil : Math.floor)(value / scale) * scale).toPrecision(2),
  );
}

/**
 * A contract stating what a recorded tensor is: its shape, its range rounded
 * outward, and `finite` when it holds no NaN or infinity. Pasted after the
 * line, it keeps later runs honest.
 */
export function contractFor(
  tensor: Pick<Tensor, "shape" | "minimum" | "maximum" | "histogram">,
) {
  const clauses = [`shape: ${tensor.shape.join(", ") || "()"}`];
  const { minimum, maximum } = tensor;
  if (typeof minimum === "number" && typeof maximum === "number")
    clauses.push(
      `range: ${outward(minimum, false)}..${outward(maximum, true)}`,
    );
  if (tensor.histogram && !tensor.histogram.non_finite) clauses.push("finite");
  return `# ${clauses.join("; ")}`;
}

/**
 * The sums contracts the browser cannot add up itself, because the line's
 * tensor keeps its values in a snapshot: what to ask the backend.
 */
export function sumsQuestions(
  lines: string[],
  results: ReadonlyMap<number, LineResult>,
) {
  const questions: {
    key: string;
    tensorId: string;
    axis: number;
    value: number;
  }[] = [];
  lines.forEach((source, index) => {
    const contract = parseValueContract(source, index + 1);
    if (!contract || "error" in contract) return;
    const result = results.get(index + 1);
    const tensor =
      result?.fresh && !result.error && !result.predicted
        ? result.output
        : null;
    if (!tensor || tensor.value_source !== "paged") return;
    for (const clause of contract.clauses)
      if (clause.kind === "sums")
        questions.push({
          key: sumsKey(tensor.id, clause.axis, clause.value),
          tensorId: tensor.id,
          axis: clause.axis,
          value: clause.value,
        });
  });
  return questions;
}

/**
 * The line rewritten so its value contracts describe what the run recorded:
 * each broken bound is redrawn around the recorded value, like accepting a
 * new snapshot. `finite` and `sums` cannot be made true by rewording, so a
 * broken one is dropped. Null when nothing is broken.
 */
export function acceptRecorded(source: string, tensor: Tensor) {
  const hash = source.indexOf("#");
  if (hash < 0) return null;
  const parts = source
    .slice(hash + 1)
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean);
  let changed = false;
  const kept: string[] = [];
  for (const part of parts) {
    const parsed = parseValueContract(`x  # ${part}`, 1);
    const clause = parsed && "clauses" in parsed ? parsed.clauses[0] : null;
    if (!clause || !breach(tensor, clause)) {
      kept.push(part);
      continue;
    }
    changed = true;
    const recorded = describe(tensor, clause);
    if (recorded) kept.push(recorded);
  }
  if (!changed) return null;
  const code = source.slice(0, hash).trimEnd();
  return kept.length ? `${code}  # ${kept.join("; ")}` : code;
}

/** A clause of the same kind stating the recorded value; null when none can. */
function describe(tensor: Tensor, clause: ValueClause): string | null {
  const histogram = tensor.histogram;
  switch (clause.kind) {
    case "range":
      return typeof tensor.minimum === "number" &&
        typeof tensor.maximum === "number"
        ? `range: ${outward(tensor.minimum, false)}..${outward(tensor.maximum, true)}`
        : null;
    case "dtype":
      return `dtype: ${tensor.dtype.replace(/^torch\./, "")}`;
    case "std":
      return typeof histogram?.std === "number"
        ? `std: ${outward(histogram.std / 2, false)}..${outward(histogram.std * 2, true)}`
        : null;
    case "zeros": {
      if (!histogram) return null;
      const total =
        histogram.counts.reduce((sum, count) => sum + count, 0) +
        histogram.non_finite;
      const share = total ? histogram.zeros / total : 0;
      return `zeros: ..${Math.min(100, Math.ceil((share * 100 + 0.01) / 5) * 5)}%`;
    }
    case "mean":
      return typeof histogram?.mean === "number" && !histogram.non_finite
        ? `mean: ${Number(histogram.mean.toPrecision(2))} ± ${outward(Math.max(Math.abs(histogram.mean) * 0.1, (histogram.std ?? 0) * 0.1, 1e-3), true)}`
        : null;
    default:
      return null;
  }
}
