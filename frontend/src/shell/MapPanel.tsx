import { useEffect, useMemo, useState } from "react";
import type { Run, Tensor } from "../api/client";
import { unravel } from "../tensors/coordinates";
import { layerStates } from "../tensors/layerStates";
import { readNpy } from "../tensors/npy";
import { project2d } from "../tensors/pca";
import { lensWords } from "./LogitLensPanel";

/** At most this many rows are mapped; more read as a cloud, not points. */
const ROWS = 400;
/** Larger tensors are not fetched whole to map. */
const VALUES = 400_000;

const loaded = new Map<string, Promise<Float64Array | null>>();

/** A tensor's recorded values: inline, or its snapshot read as a .npy. */
function valuesOf(runId: string, tensor: Tensor): Promise<Float64Array | null> {
  if (
    tensor.value_source === "inline" &&
    tensor.values?.length === tensor.numel
  )
    return Promise.resolve(Float64Array.from(tensor.values, Number));
  if (tensor.value_source === "shape" || tensor.numel > VALUES)
    return Promise.resolve(null);
  const key = `${runId}/${tensor.id}`;
  let asked = loaded.get(key);
  if (!asked) {
    asked = fetch(`/api/v1/runs/${runId}/tensors/${tensor.id}/npy`)
      .then((response) => (response.ok ? response.arrayBuffer() : null))
      .then((buffer) => (buffer ? (readNpy(buffer)?.values ?? null) : null))
      .catch(() => null);
    loaded.set(key, asked);
  }
  return asked;
}

/** The rows of a tensor's last axis, at most ROWS of them. */
function rowsOf(values: Float64Array, tensor: Tensor) {
  const width = tensor.shape.at(-1) ?? 0;
  const count = Math.min(ROWS, width ? Math.floor(values.length / width) : 0);
  return Array.from({ length: count }, (_, row) =>
    values.subarray(row * width, (row + 1) * width),
  );
}

const hue = (at: number, count: number) =>
  `hsl(${Math.round(200 + (160 * at) / Math.max(1, count - 1))} 70% 66%)`;

type Mode = "step" | "layers";

/**
 * A map of what a tensor's rows have become: each row of its last axis (a
 * word's vector, a patch's) as a point on the two directions the rows spread
 * along most (principal components), labelled by the word it stands for.
 * Rows near each other the model treats alike. Across layers, each word's
 * path runs through the state entering the block stack and each block's
 * result, all on one shared pair of directions, so you can watch the
 * representations move.
 */
export function MapPanel({
  run,
  selected,
}: {
  run: Run | null;
  selected: string | null;
}) {
  const [mode, setMode] = useState<Mode>("step");
  const states = useMemo(() => (run ? layerStates(run.trace) : []), [run]);
  // The selected step's result, or the last block's when none is selected.
  const tensor = useMemo(() => {
    if (!run) return null;
    const op = selected
      ? run.trace.operations.find((each) => each.id === selected)
      : undefined;
    const id =
      op?.outputs[0] ?? states.at(-1)?.tensorId ?? run.trace.output_ids[0];
    return id ? (run.trace.tensors[id] ?? null) : null;
  }, [run, selected, states]);
  const layered = useMemo(() => {
    if (!run || states.length < 2) return null;
    const tensors = states.map((state) => run.trace.tensors[state.tensorId]);
    const shape = JSON.stringify(tensors[0]?.shape);
    return tensors.every((each) => each && JSON.stringify(each.shape) === shape)
      ? tensors
      : null;
  }, [run, states]);
  const showing: Tensor[] =
    mode === "layers" && layered ? layered : tensor ? [tensor] : [];
  const [values, setValues] = useState<{
    key: string;
    rows?: Float64Array[][];
  } | null>(null);
  const key = `${run?.id}|${showing.map((each) => each.id).join(",")}`;
  useEffect(() => {
    if (!run || !showing.length) return;
    let live = true;
    setValues({ key });
    Promise.all(showing.map((each) => valuesOf(run.id, each))).then((found) => {
      if (!live) return;
      if (found.some((each) => !each)) return setValues({ key, rows: [] });
      setValues({
        key,
        rows: found.map((each, at) => rowsOf(each!, showing[at])),
      });
    });
    return () => {
      live = false;
    };
    // `key` names the run and the tensors mapped.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const rows = values?.key === key ? values.rows : undefined;
  const projection = useMemo(
    () => (rows?.length ? project2d(rows.flat()) : null),
    [rows],
  );
  if (!run)
    return <p className="panel-empty">Run a model to map its tensors' rows.</p>;
  const first = showing[0];
  if (!first)
    return <p className="panel-empty">Select a step to map its result.</p>;
  if ((first.shape.at(-1) ?? 0) < 2 || first.shape.length < 2)
    return (
      <p className="panel-empty">
        {first.name} {`[${first.shape.join(", ")}]`} has no rows of two or more
        values to map. Select a step whose result has a feature axis.
      </p>
    );
  const switcher = (
    <span className="map-modes" role="group" aria-label="What to map">
      <button
        type="button"
        aria-pressed={mode === "step"}
        onClick={() => setMode("step")}
      >
        {tensor?.name ?? "This step"}
      </button>
      <button
        type="button"
        aria-pressed={mode === "layers"}
        disabled={!layered}
        title={
          layered
            ? "Each word's path through the block stack, on one shared pair of directions"
            : "Needs a stack of repeated blocks whose states share a shape"
        }
        onClick={() => setMode("layers")}
      >
        Across layers
      </button>
    </span>
  );
  if (!rows)
    return (
      <div className="map-panel">
        <p className="lens-summary">{switcher} Reading the values…</p>
      </div>
    );
  if (!projection)
    return (
      <div className="map-panel">
        <p className="lens-summary">
          {switcher} This run recorded no values to map here (a shapes-only run,
          or a tensor too large to fetch whole).
        </p>
      </div>
    );
  const perState = rows[0].length;
  const leading = first.shape.slice(0, -1);
  const positions = leading.at(-1) ?? perState;
  const words = lensWords(run, positions);
  const label = (row: number) => {
    const coords = unravel(row, leading);
    const at = coords.at(-1) ?? row;
    const others = coords
      .slice(0, -1)
      .map((value, axis) => (leading[axis] > 1 ? `${value}·` : ""))
      .join("");
    return `${others}${words?.tokens[at] ?? at}`;
  };
  // Fit every point into the drawing, with room for labels.
  const xs = projection.points.map(([x]) => x);
  const ys = projection.points.map(([, y]) => y);
  const [left, right] = [Math.min(...xs), Math.max(...xs)];
  const [bottom, top] = [Math.min(...ys), Math.max(...ys)];
  const width = 640,
    height = 300,
    pad = 28;
  const sx = (x: number) =>
    pad + ((x - left) / (right - left || 1)) * (width - 2 * pad - 60);
  const sy = (y: number) =>
    height - pad - ((y - bottom) / (top - bottom || 1)) * (height - 2 * pad);
  const point = (state: number, row: number) =>
    projection.points[state * perState + row];
  const layers = mode === "layers" && layered;
  return (
    <div className="map-panel">
      <p className="lens-summary">
        {switcher}{" "}
        {layers
          ? `${perState} rows through ${showing.length} layers (${states[0].name} to ${states.at(-1)!.name}), on one pair of directions:`
          : `${first.name} [${first.shape.join(", ")}]: ${perState} rows on their two main directions:`}{" "}
        PC1 holds {Math.round(projection.explained[0] * 100)}% of the spread,
        PC2 {Math.round(projection.explained[1] * 100)}%. Rows close together,
        the model treats alike.
      </p>
      <svg
        className="map-plot"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Map of ${layers ? "each word across layers" : first.name}`}
      >
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} />
        <text x={width - pad} y={height - 8} textAnchor="end">
          PC1
        </text>
        <text x={6} y={pad - 8}>
          PC2
        </text>
        {Array.from({ length: perState }, (_, row) => {
          const color = hue(row, perState);
          if (!layers) {
            const [x, y] = point(0, row);
            return (
              <g key={row}>
                <circle cx={sx(x)} cy={sy(y)} r={4} fill={color}>
                  <title>{`${label(row)} · row ${row}`}</title>
                </circle>
                <text x={sx(x) + 6} y={sy(y) + 3} fill={color}>
                  {label(row)}
                </text>
              </g>
            );
          }
          const path = showing.map((_, state) => point(state, row));
          const [lx, ly] = path.at(-1)!;
          return (
            <g key={row}>
              <polyline
                points={path.map(([x, y]) => `${sx(x)},${sy(y)}`).join(" ")}
                stroke={color}
                fill="none"
              />
              {path.slice(0, -1).map(([x, y], state) => (
                <circle
                  key={state}
                  cx={sx(x)}
                  cy={sy(y)}
                  r={state === 0 ? 2.5 : 2}
                  fill={state === 0 ? "none" : color}
                  stroke={color}
                >
                  <title>{`${label(row)} · ${states[state].name}`}</title>
                </circle>
              ))}
              <circle cx={sx(lx)} cy={sy(ly)} r={4} fill={color}>
                <title>{`${label(row)} · ${states.at(-1)!.name}`}</title>
              </circle>
              <text x={sx(lx) + 6} y={sy(ly) + 3} fill={color}>
                {label(row)}
              </text>
            </g>
          );
        })}
      </svg>
      {layers && (
        <p className="map-legend">
          ○ {states[0].name} → ● {states.at(-1)!.name}
        </p>
      )}
    </div>
  );
}
