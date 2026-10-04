import { useContext, useEffect, useMemo, useState } from "react";
import type { Run, Tensor } from "../api/client";
import { since } from "../shell/since";
import {
  RunHistoryContext,
  tensorAcrossRuns,
  type RunPoint,
} from "./runTimeline";
import { formatValue } from "./coordinates";
import { loadRun } from "./diff";
import { asNumber } from "./margins";

/** Runs further back than this are not read. */
const RUNS = 6;

/**
 * A run's mean and spread of the tensor: from its histogram, or from its
 * values when it was recorded before histograms kept them.
 */
function statsOf(tensor: Tensor) {
  const histogram = tensor.histogram;
  let mean = histogram?.mean ?? null,
    std = histogram?.std ?? null;
  const broken = (histogram?.non_finite ?? 0) > 0;
  if (
    (mean === null || std === null) &&
    tensor.values?.length === tensor.numel
  ) {
    const values = tensor.values.map(asNumber).filter(Number.isFinite);
    if (values.length) {
      mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const centre = mean;
      std = Math.sqrt(
        values.reduce((sum, value) => sum + (value - centre) ** 2, 0) /
          values.length,
      );
    }
  }
  return { mean, std, broken };
}

const shapeText = (tensor: Tensor) => `[${tensor.shape.join(", ")}]`;

/** The change worth a sentence: a new shape, or how far the spread moved. */
function summary(points: RunPoint[]): string {
  const [now] = points;
  const reshaped = points.findIndex((point) => point.reshaped);
  if (reshaped >= 0) {
    const was = points[reshaped + 1].tensor;
    return reshaped === 0
      ? `was ${shapeText(was)} last run`
      : `${shapeText(points[reshaped].tensor)} for ${reshaped + 1} runs, ${shapeText(was)} before`;
  }
  const first = points.at(-1)!;
  const std = statsOf(now.tensor).std,
    before = statsOf(first.tensor).std;
  if (typeof std !== "number" || typeof before !== "number")
    return `${shapeText(now.tensor)} in each`;
  if (Math.abs(std - before) <= 1e-6 * Math.max(1, Math.abs(before)))
    return `σ ${formatValue(std)} in each`;
  return `σ ${formatValue(before)} → ${formatValue(std)}`;
}

/**
 * This tensor in the project's earlier runs: its mean and spread run by run,
 * oldest on the left, so an edit that changed what it holds shows as a jump.
 * A ring marks a run where its shape changed; red, one with values that were
 * not finite. Choosing an earlier run opens it in Run compare.
 */
export function AcrossRuns({
  tensor,
  runId,
  index,
}: {
  tensor: Tensor;
  runId?: string;
  index: number;
}) {
  const context = useContext(RunHistoryContext);
  const current = context && context.run.id === runId ? context.run : null;
  const older = useMemo(() => {
    if (!current || !context) return [];
    const at = context.history.findIndex((each) => each.id === current.id);
    return at < 0 ? [] : context.history.slice(at + 1, at + RUNS);
  }, [context, current]);
  const key = older.map((each) => each.id).join();
  const [runs, setRuns] = useState<{ key: string; runs: Run[] }>({
    key: "",
    runs: [],
  });
  useEffect(() => {
    if (!key) return;
    let live = true;
    Promise.all(older.map((each) => loadRun(each.id).catch(() => null))).then(
      (loaded) => {
        if (live)
          setRuns({
            key,
            runs: loaded.filter((run): run is Run => !!run),
          });
      },
    );
    return () => {
      live = false;
    };
    // `key` names the runs.
  }, [key]);
  const points = useMemo(
    () =>
      current && runs.key === key
        ? tensorAcrossRuns([current, ...runs.runs], tensor.id)
        : [],
    [current, runs, key, tensor.id],
  );
  if (points.length < 2 || !context) return null;
  // Oldest on the left.
  const shown = [...points].reverse();
  const stats = shown.map(({ tensor: each }) => statsOf(each));
  const extent = stats.flatMap(({ mean, std }) =>
    typeof mean === "number" ? [mean - (std ?? 0), mean + (std ?? 0)] : [],
  );
  const low = Math.min(...extent),
    high = Math.max(...extent);
  const width = 132,
    height = 22;
  const x = (i: number) => 5 + (i / (shown.length - 1)) * (width - 10);
  const y = (value: number) =>
    !extent.length || high === low
      ? height / 2
      : 3 + (1 - (value - low) / (high - low)) * (height - 6);
  const sameShape = (each: Tensor) =>
    each.shape.join() === tensor.shape.join() &&
    each.values?.length === each.numel;
  const cellText = (each: Tensor) =>
    sameShape(each)
      ? ` · this cell ${formatValue(asNumber(each.values[index]))}`
      : "";
  const describe = (point: RunPoint, i: number) => {
    const { mean, std } = stats[i];
    const when =
      point.runId === current!.id
        ? "This run"
        : since(new Date(point.createdAt));
    return `${when}: ${shapeText(point.tensor)}${typeof mean === "number" ? ` · mean ${formatValue(mean)}` : ""}${typeof std === "number" ? ` · σ ${formatValue(std)}` : ""}${cellText(point.tensor)}${point.reshaped ? " · new shape" : ""}${stats[i].broken ? " · not finite" : ""}`;
  };
  const path = stats
    .map(({ mean }, i) =>
      typeof mean === "number"
        ? `${x(i).toFixed(1)},${y(mean).toFixed(1)}`
        : null,
    )
    .filter(Boolean)
    .join(" ");
  return (
    <div
      className="across-runs"
      aria-label={`${tensor.name} across the last ${points.length} runs`}
    >
      <span className="across-runs-label">{points.length} runs</span>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="group"
      >
        {path.includes(" ") && <polyline points={path} />}
        {shown.map((point, i) => {
          const { mean, std, broken } = stats[i];
          const now = point.runId === current!.id;
          const open = () => !now && context.compare(point.runId);
          const cy = typeof mean === "number" ? y(mean) : height / 2;
          return (
            <g
              key={point.runId}
              className={`across-runs-point${now ? " across-runs-now" : ""}${point.reshaped ? " across-runs-reshaped" : ""}${broken ? " across-runs-broken" : ""}`}
              role={now ? undefined : "button"}
              tabIndex={now ? undefined : 0}
              aria-label={describe(point, i)}
              onClick={open}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  open();
                }
              }}
            >
              <title>
                {describe(point, i)}
                {now ? "" : " · open in Run compare"}
              </title>
              {typeof mean === "number" && typeof std === "number" && (
                <line
                  x1={x(i)}
                  x2={x(i)}
                  y1={y(mean - std)}
                  y2={y(mean + std)}
                />
              )}
              <circle cx={x(i)} cy={cy} r={now ? 3 : 2.2} />
              {point.reshaped && (
                <circle className="across-runs-ring" cx={x(i)} cy={cy} r={5} />
              )}
            </g>
          );
        })}
      </svg>
      <span className="across-runs-summary">{summary(points)}</span>
    </div>
  );
}
