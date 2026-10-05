import type { Run } from "../api/client";
import { kindName } from "../operations/kindName";

export type FlowRow = {
  id: string;
  step: number;
  operation: string;
  name: string;
  shape: number[] | null;
  dtype: string;
  numel: number;
  spread: number | null;
  zeros: number | null;
  low: number | null;
  high: number | null;
  line: number | null;
  failed: boolean;
  /** How long the PyTorch call took, in µs; null for runs before timing. */
  time: number | null;
  /** NaN and infinite values in the result. */
  broken: number;
  /** A contract on the step's line: kept, broken, or none. */
  contract?: "kept" | "broken";
  /** Names of the tensors this step reads, in operand order. */
  inputs: string[];
  /** The module call it ran inside, such as "GPT / blocks.0 / attention". */
  module: string;
  /**
   * How much it changed since a compared run: 0 when the same, the largest
   * difference otherwise, Infinity when its shape or step changed.
   */
  change?: number;
};

export type FlowSort =
  "step" | "size" | "spread" | "zeros" | "magnitude" | "time" | "change";

/** One row per recorded step: what it made, and how its values look. */
export function flowRows(trace: Run["trace"]): FlowRow[] {
  return trace.operations.map((op) => {
    const tensor = trace.tensors[op.outputs[0]];
    const histogram = tensor?.histogram;
    const total = histogram
      ? histogram.counts.reduce((sum, count) => sum + count, 0) +
        histogram.non_finite
      : 0;
    return {
      id: op.id,
      step: op.index + 1,
      operation: kindName(op.kind),
      name: tensor?.name ?? "",
      shape: tensor ? tensor.shape : null,
      dtype: tensor?.dtype ?? "",
      numel: tensor?.numel ?? 0,
      spread: typeof histogram?.std === "number" ? histogram.std : null,
      zeros: histogram && total ? histogram.zeros / total : null,
      low: histogram?.counts.length ? histogram.low : null,
      high: histogram?.counts.length ? histogram.high : null,
      line: op.source?.line ?? null,
      failed: op.status === "error",
      time: typeof op.duration_us === "number" ? op.duration_us : null,
      broken: histogram?.non_finite ?? 0,
      inputs: (op.inputs ?? []).map((id) => trace.tensors[id]?.name ?? id),
      module: op.module ?? "",
    };
  });
}

const magnitude = (row: FlowRow) =>
  row.low === null || row.high === null
    ? null
    : Math.max(Math.abs(row.low), Math.abs(row.high));

/**
 * Rows in the chosen order. Step order ascends; the measures descend, so the
 * largest tensors, widest spreads, or emptiest results come first. Rows
 * without a value sort last.
 */
export function sortFlow(rows: FlowRow[], by: FlowSort): FlowRow[] {
  if (by === "step") return [...rows].sort((a, b) => a.step - b.step);
  const value = (row: FlowRow) =>
    by === "size"
      ? row.numel
      : by === "spread"
        ? row.spread
        : by === "zeros"
          ? row.zeros
          : by === "time"
            ? row.time
            : by === "change"
              ? (row.change ?? null)
              : magnitude(row);
  return [...rows].sort((a, b) => {
    const x = value(a),
      y = value(b);
    if (x === null) return y === null ? a.step - b.step : 1;
    if (y === null) return -1;
    return (x === y ? 0 : y > x ? 1 : -1) || a.step - b.step;
  });
}

/**
 * Rows whose operation, result, inputs, or module contain every word of the
 * filter, in any case: "attention softmax" keeps the attention softmaxes.
 */
export function filterFlow(rows: FlowRow[], query: string): FlowRow[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return rows;
  return rows.filter((row) => {
    // "nan" or "inf" finds the steps whose results hold non-finite values,
    // and "contract" (or "broken") the steps with contracts.
    const text =
      `${row.operation} ${row.name} ${row.inputs.join(" ")} ${row.module}${row.broken ? " nan inf" : ""}${row.contract ? ` contract ${row.contract}` : ""}`.toLowerCase();
    return words.every((word) => text.includes(word));
  });
}

/** A module call and everything that ran inside it. */
export type ModuleGroup = {
  /** The full path, such as "CLIP / image_encoder / block". */
  path: string;
  /** Its own name, the last part of the path. */
  name: string;
  depth: number;
  /** Steps run directly in this module, not in a child. */
  rows: FlowRow[];
  children: ModuleGroup[];
  /** Totals over the module and its children. */
  steps: number;
  values: number;
  spread: number | null;
  change: number | null;
};

/**
 * Rows grouped into the tree of modules they ran in, in step order. Totals
 * cover each module and its children: how many steps, how many values, the
 * widest spread, and the largest change when comparing runs.
 */
export function moduleTree(rows: FlowRow[]): ModuleGroup[] {
  const root: ModuleGroup = {
    path: "",
    name: "",
    depth: -1,
    rows: [],
    children: [],
    steps: 0,
    values: 0,
    spread: null,
    change: null,
  };
  for (const row of [...rows].sort((a, b) => a.step - b.step)) {
    let group = root;
    const parts = row.module ? row.module.split(" / ") : [];
    parts.forEach((part, i) => {
      const path = parts.slice(0, i + 1).join(" / ");
      let child = group.children.find((item) => item.path === path);
      if (!child) {
        // "image_encoder.block.norm1" under "image_encoder.block" reads "norm1".
        const parent = parts[i - 1];
        child = {
          path,
          name:
            parent && part.startsWith(`${parent}.`)
              ? part.slice(parent.length + 1)
              : part,
          depth: i,
          rows: [],
          children: [],
          steps: 0,
          values: 0,
          spread: null,
          change: null,
        };
        group.children.push(child);
      }
      group = child;
    });
    group.rows.push(row);
  }
  // A call that ran a single step and nothing inside it, such as a Linear
  // layer's one matmul, reads as a row of its parent, as in the explorer's
  // outline: groups mark blocks, not layers.
  const lift = (group: ModuleGroup) => {
    group.children.forEach(lift);
    if (group === root) return;
    const single = group.children.filter(
      (child) => !child.children.length && child.rows.length === 1,
    );
    group.rows.push(...single.flatMap((child) => child.rows));
    group.children = group.children.filter((child) => !single.includes(child));
  };
  lift(root);
  const total = (group: ModuleGroup) => {
    group.children.forEach(total);
    const all = [
      ...group.rows.map((row) => ({
        steps: 1,
        values: row.numel,
        spread: row.spread,
        change: row.change ?? null,
      })),
      ...group.children,
    ];
    const max = (values: (number | null)[]) => {
      const known = values.filter((v): v is number => v !== null);
      return known.length ? Math.max(...known) : null;
    };
    group.steps = all.reduce((sum, item) => sum + item.steps, 0);
    group.values = all.reduce((sum, item) => sum + item.values, 0);
    group.spread = max(all.map((item) => item.spread));
    group.change = max(all.map((item) => item.change));
  };
  total(root);
  // Steps outside any module sit at the top level as their own group.
  return root.rows.length
    ? [
        { ...root, name: "(top level)", depth: 0, children: [] },
        ...root.children,
      ]
    : root.children;
}

/** The rows as CSV, with a header, for a spreadsheet or notebook. */
export function flowCsv(rows: FlowRow[]): string {
  const cell = (value: string | number | null | undefined) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const header = [
    "step",
    "operation",
    "result",
    "inputs",
    "shape",
    "dtype",
    "values",
    "spread",
    "zeros",
    "low",
    "high",
    "line",
    "module",
    "change",
    "non_finite",
    "time_us",
  ];
  const lines = rows.map((row) =>
    [
      row.step,
      row.operation,
      row.name,
      row.inputs.join(" "),
      row.shape ? row.shape.join("x") : "",
      row.dtype,
      row.numel,
      row.spread,
      row.zeros,
      row.low,
      row.high,
      row.line,
      row.module,
      row.change === Infinity ? "shape" : row.change,
      row.broken,
      row.time,
    ]
      .map(cell)
      .join(","),
  );
  return [header.join(","), ...lines].join("\n");
}
