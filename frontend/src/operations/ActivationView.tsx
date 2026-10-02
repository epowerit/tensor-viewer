import { useId, useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue, unravel } from "../tensors/coordinates";
import {
  activationFormula,
  activationNames,
  activationPlot,
  activationValue,
  activationWindow,
  finiteNumber,
  type Activation,
} from "./activation";
import { useCellPaint } from "../tensors/InkShape";
import "../tensors/gridInk.css";

const coord = (index: number, p: Activation) =>
  `[${unravel(index, p.input.shape).join(", ")}]`;

export function ActivationView({
  activation: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  activation: Activation;
  run: Run;
  showValues: boolean;
  onShowValues: (value: boolean) => void;
  onDetails: () => void;
}) {
  const [selected, setSelected] = useState(0);
  const [fit, setFit] = useState(false);
  const plotId = useId();
  const window = activationWindow(p.input.numel, selected);
  const shapeOnly =
    p.input.value_source === "shape" || p.output.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const inputData = useTensorValues(p.input, run.id, numeric ? window : []);
  // Plotted points wear their cells' ink; the selected one burns.
  const paint = useCellPaint();
  const outputData = useTensorValues(p.output, run.id, numeric ? window : []);
  const datasets = [inputData, outputData];
  const loading = numeric && datasets.some((d) => d.loading);
  const error = datasets.find((d) => d.error)?.error;
  const x = numeric ? inputData.valueAt(selected) : undefined;
  const y = numeric ? outputData.valueAt(selected) : undefined;
  const estimate = activationValue(p, x);
  const pairs = numeric
    ? window.map((index) => ({
        index,
        input: inputData.valueAt(index),
        output: outputData.valueAt(index),
      }))
    : [];
  const plot = activationPlot(p, pairs, fit);
  const zero = plot.project(0, 0);
  const path = plot.samples
    .map((point, i) => `${i ? "L" : "M"}${point.x},${point.y}`)
    .join(" ");
  const name = activationNames[p.kind];
  const formula = activationFormula(p);
  const frame = {
    rows: Math.min(8, p.input.shape.at(-2) ?? 1),
    columns: Math.min(8, p.input.shape.at(-1) ?? 1),
  };
  const note = {
    relu: "Negative values become zero. Positive values pass through unchanged.",
    gelu:
      p.approximate === "tanh"
        ? "This call uses the tanh approximation. Small negative values can remain negative."
        : "Φ(x) is the standard normal cumulative probability. Small negative values can remain negative.",
    sigmoid:
      "Values approach zero or one at the tails. Recorded results may round to these endpoints; this is independent per cell, not a sum-to-one normalization.",
    tanh: "Values approach −1 or 1 at the tails. Recorded results may round to these endpoints; zero stays zero.",
  }[p.kind];
  return (
    <section
      className="linear-lesson activation-lesson"
      aria-label={`${name} activation lesson`}
    >
      <header className="linear-heading">
        <div>
          <code>y = {formula}</code>
          <p>{note}</p>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div className="linear-controls activation-controls">
        <IndexControl
          label="Element"
          name="Activation element"
          value={selected}
          size={p.input.numel}
          onChange={setSelected}
        />
        <span className="activation-location">
          <code>
            x{coord(selected, p)} → y{coord(selected, p)}
          </code>
          <small>
            Same coordinate · {p.input.numel.toLocaleString()} elements
          </small>
        </span>
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="normalization-tensors">
        <TensorCard
          tensor={p.input}
          label="Activation input"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected}
          highlights={[selected]}
          onSelect={setSelected}
        />
        <ArrowRight className="normalization-direction" size={20} />
        <TensorCard
          tensor={p.output}
          label="Activated output"
          tone="output"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected}
          highlights={[selected]}
          onSelect={setSelected}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select either tensor, a point on the curve, or a captured pair below.
          Enlarged 3D views stay linked. Each output uses only the input at the
          same coordinate.
        </p>
      </div>
      <section
        className="linear-calculation activation-calculation"
        aria-label="Activation calculation"
      >
        <header>
          <div>
            <span className="eyebrow">How this cell changes</span>
            <code>
              x{coord(selected, p)} → y{coord(selected, p)}
            </code>
          </div>
          <span>
            {p.kind === "gelu"
              ? `GELU · ${p.approximate === "tanh" ? "tanh approximation" : "Gaussian CDF"}`
              : name}
          </span>
        </header>
        <div className="activation-results">
          <div>
            <span className="eyebrow">01 · Captured input</span>
            <strong title={String(x ?? "unavailable")}>
              {numeric ? formatValue(x) : "—"}
            </strong>
          </div>
          <div>
            <span className="eyebrow">02 · Apply {name}</span>
            <code>{formula}</code>
            <small>
              {numeric && estimate !== undefined
                ? `Reference ≈ ${formatValue(estimate)}`
                : "One independent calculation per element"}
            </small>
          </div>
          <div>
            <span className="eyebrow">03 · Recorded output</span>
            <strong title={String(y ?? "unavailable")}>
              {numeric ? formatValue(y) : "—"}
            </strong>
          </div>
        </div>
        <p className="linear-numeric-note" role="status">
          {shapeOnly
            ? "This run captured shapes only. The curve is a function reference; no tensor values or recorded points are available."
            : !showValues
              ? "Values are hidden. The curve shows the function only."
              : loading
                ? "Loading the selected input and output window…"
                : !finiteNumber(x) || !finiteNumber(y)
                  ? "A selected value is missing, non-finite, or outside browser numeric precision. Inspect the recorded values; no finite curve point is invented."
                  : "The reference curve is approximate. Values are rounded here; inspect a tensor cell for its exact captured value. The recorded PyTorch output is authoritative."}
        </p>
        {numeric && error && (
          <div className="linear-load-error" role="alert">
            {error}{" "}
            <button
              className="text-button"
              onClick={() => datasets.forEach((d) => d.retry())}
            >
              Retry values
            </button>
          </div>
        )}
        <figure className="activation-figure">
          <figcaption>
            <div>
              <span className="eyebrow">Function and captured cells</span>
              <p>
                Line: {name} reference · dots: captured pairs in this window
              </p>
            </div>
            <div
              className="activation-chart-controls"
              role="group"
              aria-label="Activation curve range"
            >
              <button
                className="secondary-button small"
                aria-pressed={!fit}
                onClick={() => setFit(false)}
              >
                Near zero
              </button>
              <button
                className="secondary-button small"
                aria-pressed={fit}
                disabled={
                  !pairs.some(
                    (pair) =>
                      finiteNumber(pair.input) && finiteNumber(pair.output),
                  )
                }
                onClick={() => setFit(true)}
              >
                Fit visible cells
              </button>
            </div>
          </figcaption>
          <svg
            className="activation-plot"
            viewBox="0 0 640 302"
            role="group"
            aria-labelledby={`${plotId}-title ${plotId}-description`}
          >
            <title id={`${plotId}-title`}>{`${name} activation curve`}</title>
            <desc id={`${plotId}-description`}>
              Input on the horizontal axis; output on the vertical axis. The
              line is a mathematical reference. Select a captured point with
              click or Enter, or use the captured-pairs table.
            </desc>
            <rect
              x="58"
              y="36"
              width="548"
              height="214"
              className="activation-plot-area"
            />
            <line
              x1="58"
              y1={zero.y}
              x2="606"
              y2={zero.y}
              className="activation-axis"
            />
            <line
              x1={zero.x}
              y1="36"
              x2={zero.x}
              y2="250"
              className="activation-axis"
            />
            <text x="58" y="270" textAnchor="start">
              {formatValue(plot.xmin)}
            </text>
            <text x="606" y="270" textAnchor="end">
              {formatValue(plot.xmax)}
            </text>
            {zero.x > 90 && zero.x < 574 && (
              <text x={zero.x} y="270" textAnchor="middle">
                0
              </text>
            )}
            <text x="49" y="41" textAnchor="end">
              {formatValue(plot.ymax)}
            </text>
            <text x="49" y="250" textAnchor="end">
              {formatValue(plot.ymin)}
            </text>
            <text x="332" y="295" textAnchor="middle">
              Input x
            </text>
            <text
              x="16"
              y="143"
              textAnchor="middle"
              transform="rotate(-90 16 143)"
            >
              Output y
            </text>
            <path d={path} className="activation-curve" />
            {[
              ...plot.points.filter((point) => point.index !== selected),
              ...plot.points.filter((point) => point.index === selected),
            ].map((point) => (
              <g
                key={point.index}
                role="button"
                tabIndex={0}
                aria-label={`Select activation cell ${coord(point.index, p)}, input ${point.input}, output ${point.output}`}
                aria-pressed={point.index === selected}
                onClick={() => setSelected(point.index)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    event.stopPropagation();
                    setSelected(point.index);
                  }
                }}
                className={`activation-point ${point.index === selected ? "selected" : ""}`}
              >
                <title>{`Cell ${coord(point.index, p)}: ${point.input} → ${point.output}`}</title>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="12"
                  className="activation-point-hit"
                />
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={point.index === selected ? 6 : 4}
                  className="activation-point-dot"
                  style={
                    point.index !== selected && paint?.(p.input, point.index)
                      ? { fill: paint(p.input, point.index)! }
                      : undefined
                  }
                />
              </g>
            ))}
          </svg>
          <p className="activation-chart-note" role="status">
            {plot.omitted
              ? `${plot.omitted} finite captured ${plot.omitted === 1 ? "pair is" : "pairs are"} outside this range. Use Fit visible cells to include them.`
              : numeric
                ? `${plot.points.length} finite captured pairs shown · overlapping points can be selected in the table.`
                : "Function reference only · captured points are hidden."}
          </p>
        </figure>
        <details className="linear-terms">
          <summary>
            Inspect captured pairs{" "}
            <span>
              {window[0].toLocaleString()}–{window.at(-1)!.toLocaleString()} /{" "}
              {p.input.numel.toLocaleString()}
            </span>
          </summary>
          <div className="linear-term-navigation">
            <p>Logical element order · at most eight pairs are loaded.</p>
            <IndexControl
              label="Cell window"
              name="Activation cell window"
              value={window[0]}
              size={p.input.numel}
              step={8}
              onChange={setSelected}
            />
          </div>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>Element</th>
                  <th>Coordinate</th>
                  <th>Input</th>
                  <th>Recorded output</th>
                </tr>
              </thead>
              <tbody>
                {window.map((index) => (
                  <tr
                    key={index}
                    className={index === selected ? "selected" : ""}
                  >
                    <th scope="row">
                      <button
                        aria-label={`Inspect activation element ${index}`}
                        aria-pressed={index === selected}
                        onClick={() => setSelected(index)}
                      >
                        {index.toLocaleString()}
                      </button>
                    </th>
                    <td>{coord(index, p)}</td>
                    <td
                      title={
                        numeric
                          ? String(inputData.valueAt(index) ?? "unavailable")
                          : undefined
                      }
                    >
                      {numeric ? formatValue(inputData.valueAt(index)) : "—"}
                    </td>
                    <td
                      title={
                        numeric
                          ? String(outputData.valueAt(index) ?? "unavailable")
                          : undefined
                      }
                    >
                      {numeric ? formatValue(outputData.valueAt(index)) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>
    </section>
  );
}
