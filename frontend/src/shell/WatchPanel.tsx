import { X } from "lucide-react";
import { useEffect, useState } from "react";
import {
  api,
  type Evaluation,
  type Run,
  type WatchSeries,
} from "../api/client";
import { formatValue } from "../tensors/coordinates";
import { holdingSteps, loadSeries, setPauses, usePauses } from "./watchStore";

const storageKey = (projectId: string) => `tensorviewer.watch.${projectId}`;

function stored(projectId: string): string[] {
  try {
    const saved = JSON.parse(
      localStorage.getItem(storageKey(projectId)) ?? "[]",
    );
    return Array.isArray(saved)
      ? saved.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

/** How a watch's value reads in one line: a tensor's shape and first values. */
export function watchText(result: Evaluation): string {
  if (result.kind !== "tensor") return result.text ?? "";
  const shape = `[${(result.shape ?? []).join(", ")}]`;
  const values = (result.values ?? [])
    .slice(0, 8)
    .map((value) =>
      typeof value === "number" ? formatValue(value) : String(value),
    );
  const more = (result.numel ?? 0) > values.length ? ", …" : "";
  return result.shape?.length
    ? `${shape} ${result.dtype} · ${values.join(", ")}${more}`
    : `${values[0] ?? ""} (${result.dtype})`;
}

/**
 * Watch expressions, as a debugger keeps them: Python over the run's named
 * tensors (`weights.sum(-1)`, `params["head.weight"].norm()`), evaluated at
 * the playback position, each name reading its state at that step. The list
 * is kept per project, and the backend evaluates; nothing is saved.
 */
export function WatchPanel({
  run,
  projectId,
  selected,
  onSelect,
}: {
  run: Run | null;
  projectId: string | null;
  selected: string | null;
  /** Opens a step (or an input's node) on the canvas. */
  onSelect?: (node: string) => void;
}) {
  // Watches drawn across the run, and their series, by run and expression.
  const [plotted, setPlotted] = useState<Set<string>>(new Set());
  // Watches that pause playback wherever they hold; playback reads them too,
  // with this tab closed.
  const pausing = usePauses(projectId);
  const togglePause = (expression: string) =>
    projectId &&
    setPauses(
      projectId,
      pausing.includes(expression)
        ? pausing.filter((each) => each !== expression)
        : [...pausing, expression],
    );
  const [series, setSeries] = useState<Record<string, WatchSeries>>({});
  const seriesKey = (expression: string) => `${run?.id}\n${expression}`;
  useEffect(() => {
    if (!run) return;
    for (const expression of new Set([...plotted, ...pausing])) {
      const key = `${run.id}\n${expression}`;
      if (series[key]) continue;
      void loadSeries(run.id, expression).then((result) =>
        setSeries((found) => ({ ...found, [key]: result })),
      );
    }
    // Each expression is asked for once per run (the store keeps it).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.id, plotted, pausing]);
  // The steps where a pausing watch holds: playback pauses there.
  const holds = (expression: string) => {
    const found = series[seriesKey(expression)];
    return found && run ? holdingSteps(run, found) : null;
  };
  const [watches, setWatches] = useState<string[]>(() =>
    projectId ? stored(projectId) : [],
  );
  const [draft, setDraft] = useState("");
  const [results, setResults] = useState<Record<string, Evaluation>>({});
  const [names, setNames] = useState<string[]>([]);
  useEffect(() => {
    setWatches(projectId ? stored(projectId) : []);
  }, [projectId]);
  const save = (next: string[]) => {
    setWatches(next);
    if (!projectId) return;
    try {
      localStorage.setItem(storageKey(projectId), JSON.stringify(next));
    } catch {
      // Storage can be unavailable; the list still works for this visit.
    }
  };
  const at =
    run && selected && run.trace.operations.some((op) => op.id === selected)
      ? selected
      : null;
  const step = at
    ? run!.trace.operations.find((op) => op.id === at)
    : undefined;
  const key = `${run?.id}/${at}/${watches.join("\n")}`;
  useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    // Evaluated once playback settles on a step, one after another.
    const timer = setTimeout(async () => {
      const found: Record<string, Evaluation> = {};
      for (const expression of watches.length ? watches : ["0"]) {
        const result = await api
          .evaluate(run.id, expression, at, controller.signal)
          .catch((error: Error) =>
            controller.signal.aborted
              ? null
              : ({
                  kind: "error",
                  text: error.message,
                  names: [],
                } as Evaluation),
          );
        if (controller.signal.aborted || !result) return;
        found[expression] = result;
        setNames(result.names ?? []);
      }
      setResults(found);
    }, 350);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
    // `key` names the run, the step, and the expressions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (!run)
    return (
      <p className="panel-empty">Run to watch expressions over its tensors.</p>
    );
  return (
    <div className="watch-panel">
      <p className="watch-where">
        {step
          ? `At step ${step.index + 1} · ${step.kind}${step.source?.line ? ` · line ${step.source.line}` : ""}`
          : "At the end of the run"}
        {" · "}names read their state here
      </p>
      <ul className="watch-list">
        {watches.map((expression, at) => {
          const result = results[expression];
          return (
            <li key={`${at}/${expression}`} className="watch-row">
              <code className="watch-expression">{expression}</code>
              <button
                type="button"
                className={`watch-plot${plotted.has(expression) ? " watch-plot-on" : ""}`}
                aria-pressed={plotted.has(expression)}
                title="Draw it across the run: one number at every step its names change"
                onClick={() =>
                  setPlotted((current) => {
                    const next = new Set(current);
                    if (!next.delete(expression)) next.add(expression);
                    return next;
                  })
                }
              >
                ∿
              </button>
              <button
                type="button"
                className={`watch-plot${pausing.includes(expression) ? " watch-pause-on" : ""}`}
                aria-pressed={pausing.includes(expression)}
                title="Pause playback at every step where this holds (is true or non-zero)"
                onClick={() => togglePause(expression)}
              >
                ⏸
              </button>
              <span
                className={`watch-value${result?.kind === "error" ? " watch-error" : ""}`}
                title={
                  result?.stats
                    ? `min ${formatValue(result.stats.min)} · max ${formatValue(result.stats.max)} · mean ${formatValue(result.stats.mean)} · σ ${formatValue(result.stats.std)}${result.non_finite ? ` · ${result.non_finite} NaN/∞` : ""}`
                    : undefined
                }
              >
                {result ? watchText(result) : "…"}
              </span>
              <button
                type="button"
                className="watch-remove"
                aria-label={`Stop watching ${expression}`}
                onClick={() => {
                  save(watches.filter((_, i) => i !== at));
                  if (projectId && pausing.includes(expression))
                    setPauses(
                      projectId,
                      pausing.filter((each) => each !== expression),
                    );
                }}
              >
                <X size={12} />
              </button>
              {pausing.includes(expression) && (
                <WatchHolds
                  points={holds(expression)}
                  selected={selected}
                  onSelect={onSelect}
                />
              )}
              {plotted.has(expression) && (
                <WatchCurve
                  series={series[seriesKey(expression)]}
                  selected={selected}
                  onSelect={onSelect}
                />
              )}
            </li>
          );
        })}
      </ul>
      <form
        className="watch-add"
        onSubmit={(event) => {
          event.preventDefault();
          const expression = draft.trim();
          if (!expression || watches.includes(expression)) return;
          save([...watches, expression]);
          setDraft("");
        }}
      >
        <input
          value={draft}
          spellCheck={false}
          placeholder='Add a watch: weights.sum(-1), x.std(), params["head.weight"].norm()'
          aria-label="Add a watch expression"
          onChange={(event) => setDraft(event.target.value)}
        />
      </form>
      {names.length > 0 && (
        <p className="watch-names">
          Names here: {names.slice(0, 24).join(", ")}
          {names.length > 24 ? ", …" : ""} · torch, F, math, params["…"]
        </p>
      )}
    </div>
  );
}

/**
 * A watch across the run: its number at every step where a name it reads
 * changed, step by step. The playback step is marked, and a point opens its
 * step.
 */
function WatchCurve({
  series,
  selected,
  onSelect,
}: {
  series: WatchSeries | { loading: true } | undefined;
  selected: string | null;
  onSelect?: (node: string) => void;
}) {
  if (!series || "loading" in series)
    return (
      <span className="watch-curve watch-curve-note">
        Evaluating across the run…
      </span>
    );
  const points = (series.points ?? []).filter(
    (point): point is typeof point & { value: number } =>
      typeof point.value === "number",
  );
  if (series.error || !points.length)
    return (
      <span className="watch-curve watch-curve-note watch-error">
        {series.error ??
          series.points?.find((point) => point.error)?.error ??
          "No step changes the names it reads"}
      </span>
    );
  const values = points.map((point) => point.value);
  const low = Math.min(...values),
    high = Math.max(...values);
  const width = 420,
    height = 44;
  const x = (at: number) =>
    6 + (points.length === 1 ? 0.5 : at / (points.length - 1)) * (width - 12);
  const y = (value: number) =>
    high === low
      ? height / 2
      : 4 + (1 - (value - low) / (high - low)) * (height - 8);
  const current = points.findIndex((point) => point.at === selected);
  return (
    <span className="watch-curve">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <polyline
          points={points
            .map(
              (point, at) => `${x(at).toFixed(1)},${y(point.value).toFixed(1)}`,
            )
            .join(" ")}
        />
        {points.map((point, at) => (
          <circle
            key={point.at}
            cx={x(at)}
            cy={y(point.value)}
            r={at === current ? 4 : 2.5}
            className={at === current ? "watch-curve-now" : undefined}
            role="button"
            tabIndex={0}
            aria-label={`Step ${point.step}${point.line ? `, line ${point.line}` : ""}: ${formatValue(point.value)}`}
            onClick={() => onSelect?.(point.at)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect?.(point.at);
              }
            }}
          >
            <title>{`${point.step ? `Step ${point.step}${point.line ? ` · line ${point.line}` : ""}` : "The input"}: ${formatValue(point.value)}`}</title>
          </circle>
        ))}
      </svg>
      <span className="watch-curve-note">
        {formatValue(values[0])} → {formatValue(values.at(-1)!)} over{" "}
        {points.length} states{series.truncated ? " (the first 200)" : ""}
      </span>
    </span>
  );
}

/** Where a pausing watch holds: the steps playback pauses at, to open. */
function WatchHolds({
  points,
  selected,
  onSelect,
}: {
  points: { at: string; step: number; line?: number | null }[] | null;
  selected: string | null;
  onSelect?: (node: string) => void;
}) {
  if (!points)
    return (
      <span className="watch-holds watch-curve-note">
        Finding where it holds…
      </span>
    );
  if (!points.length)
    return (
      <span className="watch-holds watch-curve-note">
        Never holds in this run, so playback does not pause for it
      </span>
    );
  return (
    <span className="watch-holds">
      <span className="watch-curve-note">
        Pauses at {points.length} {points.length === 1 ? "step" : "steps"}:
      </span>
      {points.slice(0, 12).map((point) => (
        <button
          type="button"
          key={point.at}
          className={point.at === selected ? "watch-hold-now" : undefined}
          onClick={() => onSelect?.(point.at)}
          title={point.line ? `Line ${point.line}` : undefined}
        >
          step {point.step}
        </button>
      ))}
      {points.length > 12 && (
        <span className="watch-curve-note">and {points.length - 12} more</span>
      )}
    </span>
  );
}
