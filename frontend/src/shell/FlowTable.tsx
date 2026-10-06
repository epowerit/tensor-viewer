import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, type Run } from "../api/client";
import { ContractContext } from "../editor/ContractContext";
import { durationText } from "../journey/cost";
import { changeValues, lensText } from "../journey/flow";
import { useComparison } from "../workspace/useComparison";
import {
  filterFlow,
  flowCsv,
  flowRows,
  moduleTree,
  sortFlow,
  type FlowRow,
  type FlowSort,
  type ModuleGroup,
} from "./flowRows";
import "./flowTable.css";
import { useStepPreview } from "./stepPreview";

const number = (value: number | null) =>
  value === null
    ? "—"
    : value !== 0 && (Math.abs(value) >= 1000 || Math.abs(value) < 0.01)
      ? value.toExponential(1)
      : String(Number(value.toPrecision(3)));

const COLUMNS: { key: FlowSort | null; label: string; title?: string }[] = [
  { key: "step", label: "Step", title: "The step's place in the run" },
  { key: null, label: "Operation", title: "What the step did" },
  { key: null, label: "Result", title: "The name its result was given" },
  { key: null, label: "Inputs", title: "The tensors it read" },
  {
    key: "size",
    label: "Shape",
    title: "The result's shape. Sort by how many values it holds",
  },
  {
    key: "spread",
    label: "σ",
    title: "Spread: the standard deviation of the result's values. Sort by it",
  },
  {
    key: "zeros",
    label: "Zeros",
    title: "The share of the result's values that are exactly zero. Sort by it",
  },
  {
    key: "magnitude",
    label: "Range",
    title:
      "The result's smallest … largest value. Sort by the largest magnitude",
  },
  {
    key: "time",
    label: "Time",
    title:
      "How long the PyTorch call took, measured without the recording; a kind's first call includes one-time setup. Sort slowest first",
  },
  { key: null, label: "Line", title: "The line of code that ran the step" },
];

/**
 * The run's tensor flow as a table: one row per step, in execution order or
 * sorted by a measure, so the largest tensors, widest spreads, or emptiest
 * results are a click away. A row opens its step.
 */
export function FlowTable({
  run,
  selected,
  range,
  changed,
  onSelect,
  onPreview,
  hovered = null,
  previousRunId = null,
}: {
  run: Run;
  selected: string | null;
  /** A folded card's steps, played as one: their rows share a band. */
  range?: string[];
  /** Steps the last save added, edited, or reshaped, lit for a moment. */
  changed?: ReadonlySet<string> | null;
  onSelect: (id: string) => void;
  onPreview?: (id: string | null) => void;
  /** Steps under the pointer on the canvas (a stage: all of its steps). */
  hovered?: string[] | null;
  /** The run recorded before this one, offered for comparison. */
  previousRunId?: string | null;
}) {
  const [sort, setSort] = useState<FlowSort>("step");
  // The tools stay above the column heads as the table scrolls: the heads
  // stick just below them, however tall the tools are drawn.
  const scroller = useRef<HTMLDivElement>(null);
  const tools = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = tools.current;
    if (!element) return;
    const measure = () =>
      scroller.current?.style.setProperty(
        "--flow-tools",
        `${element.offsetHeight}px`,
      );
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [run]);
  // Pointing at a row traces its step on the canvas.
  const { preview } = useStepPreview(onPreview);
  const [query, setQuery] = useState("");
  // Comparing with the run before adds a Δ column and its profile.
  const [comparing, setComparing] = useState(false);
  const [earlier, setEarlier] = useState<Run | null>(null);
  useEffect(() => {
    if (!comparing || !previousRunId || earlier?.id === previousRunId) return;
    let current = true;
    api
      .getRun(previousRunId)
      .then((found) => current && setEarlier(found))
      .catch(() => current && setComparing(false));
    return () => {
      current = false;
    };
  }, [comparing, previousRunId, earlier?.id]);
  const compared = comparing && earlier?.id === previousRunId ? earlier : null;
  // Large tensors are compared by the backend over their snapshots.
  const comparison = useComparison(run, compared).steps;
  const contracts = useContext(ContractContext);
  const all = useMemo(() => {
    const rows = flowRows(run.trace).map((row) => {
      const check = contracts?.byOperation.get(row.id);
      return check
        ? {
            ...row,
            contract: check.ok ? ("kept" as const) : ("broken" as const),
          }
        : row;
    });
    if (!compared || !comparison) return rows;
    const changes = changeValues(comparison);
    return rows.map((row) => ({ ...row, change: changes.get(row.id) }));
  }, [run, compared, comparison, contracts]);
  const before = useMemo(
    () => (compared ? flowRows(compared.trace) : null),
    [compared],
  );
  const rows = useMemo(
    () => sortFlow(filterFlow(all, query), sort),
    [all, query, sort],
  );
  const body = useRef<HTMLTableSectionElement>(null);
  useEffect(() => {
    body.current
      ?.querySelector(".flow-hovered")
      ?.scrollIntoView({ block: "nearest" });
  }, [hovered]);
  useEffect(() => {
    body.current
      ?.querySelector('[aria-current="step"], .flow-in-range')
      ?.scrollIntoView({ block: "nearest" });
  }, [selected, range, sort]);
  // Steps sit in the modules they ran in, as on the canvas; off, one flat list.
  const [grouped, setGrouped] = useState(true);
  const [folded, setFolded] = useState<Set<string>>(new Set());
  const toggleGroup = (path: string) =>
    setFolded((previous) => {
      const next = new Set(previous);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  // A module's rows follow the table's sort; folded modules hide theirs.
  // In step order a module's own steps and its submodules interleave as
  // they ran; sorted by a measure, its own steps come first.
  const firstStep = (group: ModuleGroup): number =>
    Math.min(
      ...group.rows.map((row) => row.step),
      ...group.children.map(firstStep),
    );
  const flatten = (
    groups: ModuleGroup[],
  ): ({ group: ModuleGroup } | { row: FlowRow; depth: number })[] =>
    groups.flatMap((group) => {
      if (folded.has(group.path)) return [{ group }];
      const own = sortFlow(group.rows, sort).map((row) => ({
        row,
        depth: group.depth + 1,
        at: row.step,
      }));
      const nested = group.children.map((child) => ({
        child,
        at: firstStep(child),
      }));
      const parts: ((typeof own)[number] | (typeof nested)[number])[] =
        sort === "step"
          ? [...own, ...nested].sort((a, b) => a.at - b.at)
          : [...own, ...nested];
      return [
        { group },
        ...parts.flatMap((part) =>
          "child" in part
            ? flatten([part.child])
            : [{ row: part.row, depth: part.depth }],
        ),
      ];
    });
  const columnCount = COLUMNS.length + (compared ? 1 : 0);
  const [copied, setCopied] = useState(false);
  const renderRow = (row: FlowRow, depth: number) => (
    <tr
      key={row.id}
      aria-current={row.id === selected ? "step" : undefined}
      className={`${row.failed ? "flow-failed" : ""} ${hovered?.includes(row.id) ? "flow-hovered" : ""} ${range?.includes(row.id) ? "flow-in-range" : ""} ${changed?.has(row.id) ? "flow-just-changed" : ""}`}
      tabIndex={0}
      onClick={() => onSelect(row.id)}
      onMouseEnter={() => preview(row.id)}
      onMouseLeave={() => preview(null)}
      onFocus={() => preview(row.id)}
      onBlur={() => preview(null)}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(row.id);
        }
      }}
    >
      <td style={depth ? { paddingLeft: 10 + depth * 14 } : undefined}>
        {row.step}
      </td>
      <td>
        <code>{row.operation}</code>
      </td>
      <td title={row.module ? `${row.name} · ${row.module}` : row.name}>
        {row.failed ? "failed" : row.name}
        {row.contract && (
          <small
            className={`flow-contract flow-contract-${row.contract}`}
            title={contracts?.byOperation.get(row.id)?.message}
          >
            {" "}
            {row.contract === "kept" ? "✓" : "✗"}
          </small>
        )}
      </td>
      <td className="flow-inputs" title={row.inputs.join(", ")}>
        {row.inputs.length ? `← ${row.inputs.join(", ")}` : ""}
      </td>
      <td>
        {row.shape ? `[${row.shape.join(", ")}]` : "—"}
        {row.dtype && row.dtype !== "float32" && <small> {row.dtype}</small>}
      </td>
      <td>{number(row.spread)}</td>
      <td>{row.zeros === null ? "—" : `${Math.round(row.zeros * 100)}%`}</td>
      <td>
        {row.low === null ? "—" : `${number(row.low)} … ${number(row.high)}`}
        {row.broken > 0 && (
          <small
            className="flow-broken"
            title={`${row.broken.toLocaleString()} NaN or infinite values`}
          >
            {" "}
            {row.broken.toLocaleString()} NaN/∞
          </small>
        )}
      </td>
      <td>{row.time === null ? "—" : durationText(row.time)}</td>
      <td>{row.line ?? ""}</td>
      {compared && (
        <td
          className={
            row.change ? "flow-changed" : row.change === 0 ? "flow-same" : ""
          }
        >
          {row.change === undefined
            ? "—"
            : row.change === 0
              ? "same"
              : lensText(row.change, "change")}
        </td>
      )}
    </tr>
  );
  if (!all.length)
    return <p className="panel-empty">This run recorded no steps.</p>;
  return (
    <div className="flow-table-scroll" ref={scroller}>
      <div className="flow-table-tools" ref={tools}>
        <FlowProfile
          rows={all}
          before={before}
          metric={sort === "zeros" || sort === "magnitude" ? sort : "spread"}
          selected={selected}
          onSelect={onSelect}
        />
        <input
          type="search"
          value={query}
          placeholder="Filter: attention, softmax, q…"
          aria-label="Filter steps by operation, result, input, or module"
          onChange={(event) => setQuery(event.target.value)}
        />
        {query && (
          <small>
            {rows.length} of {all.length} steps
          </small>
        )}
        <button
          className="text-button"
          aria-pressed={comparing}
          disabled={!previousRunId}
          title={
            previousRunId
              ? "Add how much each step changed since the run recorded before this one"
              : "No earlier run of this project to compare with"
          }
          onClick={() => {
            setComparing(!comparing);
            if (comparing && sort === "change") setSort("step");
          }}
        >
          {comparing ? "Stop comparing" : "Compare with the run before"}
        </button>
        <button
          className="text-button"
          aria-pressed={grouped}
          title="Group steps into the modules they ran in"
          onClick={() => setGrouped(!grouped)}
        >
          Group by module
        </button>
        <button
          className="text-button"
          title="Copy the rows shown, in this order, as CSV"
          onClick={() => {
            void navigator.clipboard
              .writeText(flowCsv(rows))
              .then(() => {
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1400);
              })
              .catch(() => {});
          }}
        >
          {copied ? "Copied" : "Copy as CSV"}
        </button>
      </div>
      <table className="flow-table">
        <thead>
          <tr>
            {[
              ...COLUMNS,
              ...(compared
                ? [
                    {
                      key: "change" as FlowSort,
                      label: "Δ",
                      title: "Sort by how much the step changed",
                    },
                  ]
                : []),
            ].map((column) => (
              <th
                key={column.label}
                scope="col"
                title={column.key ? undefined : column.title}
                aria-sort={
                  column.key && column.key === sort
                    ? column.key === "step"
                      ? "ascending"
                      : "descending"
                    : undefined
                }
              >
                {column.key ? (
                  <button
                    title={column.title ?? `Sort by ${column.label}`}
                    onClick={() => setSort(column.key!)}
                  >
                    {column.label}
                  </button>
                ) : (
                  column.label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody ref={body}>
          {grouped
            ? flatten(moduleTree(rows)).map((item) =>
                "group" in item ? (
                  <tr
                    key={`module-${item.group.path}`}
                    className="flow-module"
                    onClick={() => toggleGroup(item.group.path)}
                  >
                    <td
                      colSpan={columnCount}
                      style={{ paddingLeft: 10 + item.group.depth * 14 }}
                    >
                      <button
                        aria-expanded={!folded.has(item.group.path)}
                        aria-label={`${folded.has(item.group.path) ? "Expand" : "Collapse"} ${item.group.path || item.group.name}`}
                      >
                        {folded.has(item.group.path) ? "▸" : "▾"}
                      </button>
                      <b>{item.group.name}</b>
                      <small>
                        {item.group.steps}{" "}
                        {item.group.steps === 1 ? "step" : "steps"} ·{" "}
                        {item.group.values.toLocaleString()} values
                        {item.group.spread !== null &&
                          ` · widest σ ${number(item.group.spread)}`}
                        {compared &&
                          item.group.change !== null &&
                          ` · ${item.group.change === 0 ? "unchanged" : lensText(item.group.change, "change")}`}
                      </small>
                    </td>
                  </tr>
                ) : (
                  renderRow(item.row, item.depth)
                ),
              )
            : rows.map((row) => renderRow(row, 0))}
        </tbody>
      </table>
    </div>
  );
}

type ProfileMetric = "spread" | "zeros" | "magnitude";

const metricOf = (row: FlowRow, metric: ProfileMetric) =>
  metric === "spread"
    ? row.spread
    : metric === "zeros"
      ? row.zeros
      : row.low === null || row.high === null
        ? null
        : Math.max(Math.abs(row.low), Math.abs(row.high));

const METRIC_LABEL = {
  spread: "Spread of values (σ, log scale)",
  zeros: "Share of zeros",
  magnitude: "Largest |value| (log scale)",
};

/**
 * One measure at every step in execution order: the run's flow profile. It
 * plots the table's sort measure (σ unless sorted by zeros or range). A
 * tenfold jump between neighbouring steps, the signature of exploding or
 * vanishing values, is marked; the current step is marked too, and a click
 * opens the nearest step.
 */
function FlowProfile({
  rows,
  before,
  metric,
  selected,
  onSelect,
}: {
  rows: FlowRow[];
  /** A compared run's rows, drawn faintly behind. */
  before?: FlowRow[] | null;
  metric: ProfileMetric;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  const [pointer, setPointer] = useState<number | null>(null);
  const logScale = metric !== "zeros";
  const measured = (list: FlowRow[]) =>
    list
      .map((row) => ({ row, value: metricOf(row, metric) }))
      .filter(
        (point): point is { row: FlowRow; value: number } =>
          point.value !== null && (!logScale || point.value > 0),
      );
  const points = measured(rows);
  if (points.length < 2) return null;
  const earlierPoints = measured(before ?? []);
  const width = 600,
    height = 40;
  const toScale = (value: number) => (logScale ? Math.log10(value) : value);
  const scaled = [...points, ...earlierPoints].map((point) =>
    toScale(point.value),
  );
  const low = Math.min(...scaled),
    high = Math.max(...scaled);
  const last = rows.length - 1 || 1;
  const earlierLast = (before?.length ?? 1) - 1 || 1;
  const x = (row: FlowRow, of = last) => ((row.step - 1) / of) * width;
  const y = (value: number) =>
    height - 4 - ((toScale(value) - low) / (high - low || 1)) * (height - 8);
  const jumps = logScale
    ? points.flatMap((point, i) =>
        i > 0 &&
        (point.value / points[i - 1].value >= 10 ||
          point.value / points[i - 1].value <= 0.1)
          ? [i]
          : [],
      )
    : [];
  const nearestTo = (step: number) =>
    points.reduce((best, point) =>
      Math.abs(point.row.step - 1 - step) < Math.abs(best.row.step - 1 - step)
        ? point
        : best,
    );
  const stepAt = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    return Math.round(((event.clientX - box.left) / box.width) * last);
  };
  const read = pointer === null ? null : nearestTo(pointer);
  const current = rows.find((row) => row.id === selected);
  const text = (value: number) =>
    metric === "zeros"
      ? `${Math.round(value * 100)}% zeros`
      : `${metric === "spread" ? "σ" : "|x| ≤"} ${Number(value.toPrecision(3))}`;
  return (
    <div className="flow-profile-wrap">
      <svg
        className="flow-profile"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${METRIC_LABEL[metric]} across ${rows.length} steps${jumps.length ? `, with ${jumps.length} tenfold jumps` : ""}${earlierPoints.length ? ", with the run before drawn behind" : ""}. Select a point to open its step.`}
        onClick={(event) => onSelect(nearestTo(stepAt(event)).row.id)}
        onMouseMove={(event) => setPointer(stepAt(event))}
        onMouseLeave={() => setPointer(null)}
      >
        <title>{`${METRIC_LABEL[metric]} along the run`}</title>
        {earlierPoints.length > 1 && (
          <polyline
            className="flow-profile-before"
            points={earlierPoints
              .map((point) => `${x(point.row, earlierLast)},${y(point.value)}`)
              .join(" ")}
          />
        )}
        <polyline
          points={points
            .map((point) => `${x(point.row)},${y(point.value)}`)
            .join(" ")}
        />
        {jumps.map((i) => (
          <circle
            key={points[i].row.id}
            className="flow-profile-jump"
            cx={x(points[i].row)}
            cy={y(points[i].value)}
            r={2.6}
          >
            <title>{`Step ${points[i].row.step}: ${(points[i].value / points[i - 1].value).toPrecision(2)}× the step before`}</title>
          </circle>
        ))}
        {current && <line x1={x(current)} x2={x(current)} y1={0} y2={height} />}
        {read && (
          <line
            className="flow-profile-pointer"
            x1={x(read.row)}
            x2={x(read.row)}
            y1={0}
            y2={height}
          />
        )}
      </svg>
      {read && (
        <span className="flow-profile-readout">
          {read.row.step} · {read.row.operation} · {text(read.value)}
        </span>
      )}
    </div>
  );
}
