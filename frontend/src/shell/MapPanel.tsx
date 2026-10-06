import { useEffect, useMemo, useState } from "react";
import type { Run, Tensor } from "../api/client";
import { unravel } from "../tensors/coordinates";
import { layerStates } from "../tensors/layerStates";
import { valuesOf } from "../tensors/loadValues";
import { project2d } from "../tensors/pca";
import { hoverPosition, usePositionFocus } from "../tensors/positionFocus";
import { lensWords } from "./LogitLensPanel";
import { PanelLoading } from "./PanelLoading";

/** At most this many rows are mapped; more read as a cloud, not points. */
const ROWS = 400;
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

type Mode = "step" | "layers" | "similarity";

/** Similarity is drawn for up to this many rows. */
const SIMILAR = 64;

/** The cosine similarity of each pair of rows. */
export function similarity(rows: ArrayLike<number>[]): number[][] {
  const norms = rows.map((row) =>
    Math.sqrt(Array.from(row).reduce((sum, v) => sum + v * v, 0)),
  );
  return rows.map((a, i) =>
    rows.map((b, j) => {
      if (!norms[i] || !norms[j]) return 0;
      let dot = 0;
      for (let k = 0; k < a.length; k++) dot += a[k] * b[k];
      return dot / (norms[i] * norms[j]);
    }),
  );
}

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
  const focused = usePositionFocus();
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
        title="Each row as a point on the two directions the rows vary most"
        onClick={() => setMode("step")}
      >
        Points
      </button>
      <button
        type="button"
        aria-pressed={mode === "similarity"}
        title="How alike each pair of rows is: the cosine of the angle between them"
        onClick={() => setMode("similarity")}
      >
        Similarity
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
  // What is mapped, and for points, what their colours mean.
  const toolbar = (points = 0) => (
    <div className="map-toolbar">
      {switcher}
      <span className="map-source">
        {mode === "layers" && layered ? (
          <>
            Mapping the state entering {states[0].name.replace("before ", "")}{" "}
            and each block's result
          </>
        ) : (
          <>
            Mapping <b>{first.name}</b> [{first.shape.join(", ")}],{" "}
            {selected
              ? "the step playback is on"
              : states.length
                ? "the last block's result"
                : "the model's result"}
          </>
        )}
      </span>
      {points > 1 && (
        <span
          className="map-key"
          title="Each point's colour follows its row's order"
        >
          <i style={{ background: hue(0, points) }} /> first row …{" "}
          <i style={{ background: hue(points - 1, points) }} /> last
        </span>
      )}
      {points > 1 && mode === "layers" && layered && (
        <span
          className="map-key"
          title="Each word's path starts where the stack begins and ends at the last block"
        >
          ○ {states[0].name} → ● {states.at(-1)!.name}
        </span>
      )}
    </div>
  );
  if (!rows)
    return (
      <div className="map-panel">
        {toolbar()}
        <PanelLoading>Reading the values…</PanelLoading>
      </div>
    );
  // NaN or infinity has no place on a map: say so rather than draw nothing.
  const broken = showing.find((_, at) =>
    rows[at]?.some((row) => row.some((value) => !Number.isFinite(value))),
  );
  if (broken)
    return (
      <div className="map-panel">
        {toolbar()}
        <p className="lens-summary">
          {broken.name} holds NaN or infinite values, which have no place on a
          map. Select a step before they appear.
        </p>
      </div>
    );
  if (!projection)
    return (
      <div className="map-panel">
        {toolbar()}
        <p className="lens-summary">
          This run recorded no values to map here (a shapes-only run, or a
          tensor too large to fetch whole).
        </p>
      </div>
    );
  const perState = rows[0].length;
  const leading = first.shape.slice(0, -1);
  const positions = leading.at(-1) ?? perState;
  const words = lensWords(run, positions);
  const positionOf = (row: number) => unravel(row, leading).at(-1) ?? row;
  // The word under the pointer in any panel stands out; the rest step back.
  const focusClass = (row: number) =>
    focused === null
      ? undefined
      : positionOf(row) === focused
        ? "is-focused"
        : "is-dimmed";
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
  if (mode === "similarity") {
    const shown = rows[0].slice(0, SIMILAR);
    const matrix = similarity(shown);
    const cell = Math.max(8, Math.min(20, Math.floor(300 / shown.length)));
    const labelled = shown.length <= SIMILAR;
    return (
      <div className="map-panel">
        {toolbar()}
        <p className="lens-summary">
          How alike each pair of the {shown.length} rows is (cosine): warm
          alike, cool opposite
          {rows[0].length > shown.length
            ? `, for the first ${shown.length} of ${rows[0].length} rows`
            : ""}
          .
        </p>
        <svg
          className="similarity-plot"
          width={110 + shown.length * cell}
          height={70 + shown.length * cell}
          role="img"
          aria-label={`Similarity of the rows of ${first.name}`}
        >
          <g transform="translate(70 70)">
            {matrix.map((row, i) =>
              row.map((value, j) => (
                <rect
                  key={`${i}-${j}`}
                  x={j * cell}
                  y={i * cell}
                  width={cell - 1}
                  height={cell - 1}
                  fill={
                    value >= 0
                      ? `rgba(242, 166, 108, ${Math.min(1, value).toFixed(3)})`
                      : `rgba(110, 168, 217, ${Math.min(1, -value).toFixed(3)})`
                  }
                >
                  <title>{`${label(i)} · ${label(j)}: ${value.toFixed(2)}`}</title>
                </rect>
              )),
            )}
            {labelled &&
              shown.map((_, i) => (
                <text
                  key={`r${i}`}
                  x={-6}
                  y={i * cell + cell / 2 + 3}
                  textAnchor="end"
                  className={focusClass(i)}
                  {...hoverPosition(positionOf(i), label(i))}
                >
                  {label(i)}
                </text>
              ))}
            {labelled &&
              shown.map((_, j) => (
                <text
                  key={`c${j}`}
                  transform={`translate(${j * cell + cell / 2 + 3} -6) rotate(-60)`}
                  className={focusClass(j)}
                  {...hoverPosition(positionOf(j), label(j))}
                >
                  {label(j)}
                </text>
              ))}
            {shown.map((_, i) =>
              positionOf(i) === focused ? (
                <g key={`f${i}`} className="focus-outline">
                  <rect
                    x={0}
                    y={i * cell}
                    width={shown.length * cell - 1}
                    height={cell - 1}
                  />
                  <rect
                    x={i * cell}
                    y={0}
                    width={cell - 1}
                    height={shown.length * cell - 1}
                  />
                </g>
              ) : null,
            )}
          </g>
        </svg>
      </div>
    );
  }
  return (
    <div className="map-panel">
      {toolbar(perState)}
      <p className="lens-summary">
        {layers
          ? `${perState} rows through ${showing.length} layers (${states[0].name} to ${states.at(-1)!.name}), on one pair of directions`
          : `${perState} rows on the two directions they vary most`}
        . These hold {Math.round(projection.explained[0] * 100)}% (PC1) and{" "}
        {Math.round(projection.explained[1] * 100)}% (PC2) of the variation.
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
              <g
                key={row}
                className={focusClass(row)}
                {...hoverPosition(positionOf(row), label(row))}
              >
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
            <g
              key={row}
              className={focusClass(row)}
              {...hoverPosition(positionOf(row), label(row))}
            >
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
