import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue, unravel } from "../tensors/coordinates";
import { normalizationNumbers } from "./normalizationStatistics";
import { useNormalizationStatistics } from "./useNormalizationStatistics";
import {
  layerNormSelection,
  layerNormWindow,
  layerNormStatistics,
  layerNormCalculation,
  type LayerNormalization,
} from "./layerNormalization";

const coord = (a: number[]) => `[${a.join(", ")}]`;
export function LayerNormalizationView({
  normalization: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  normalization: LayerNormalization;
  run: Run;
  showValues: boolean;
  onShowValues: (value: boolean) => void;
  onDetails: () => void;
}) {
  const [selected, setSelected] = useState(0);
  const selection = layerNormSelection(p, selected);
  const window = layerNormWindow(p, selected),
    samples = layerNormWindow(p, selected, true);
  const complete = samples.length === p.size,
    shapeOnly = p.input.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const summary = useNormalizationStatistics(
    {
      runId: run.id,
      operationId: p.operationId,
      tensorId: p.input.id,
      group: selection.group,
      size: p.size,
    },
    numeric && !complete,
  );
  const inputData = useTensorValues(p.input, run.id, numeric ? samples : []);
  const outputData = useTensorValues(
    p.output,
    run.id,
    numeric ? [selected] : [],
  );
  const scaleData = useTensorValues(
    p.weight ?? p.input,
    run.id,
    numeric && p.weight ? [selection.feature] : [],
  );
  const biasData = useTensorValues(
    p.bias ?? p.input,
    run.id,
    numeric && p.bias ? [selection.feature] : [],
  );
  const datasets = [
    inputData,
    outputData,
    ...(p.weight ? [scaleData] : []),
    ...(p.bias ? [biasData] : []),
    summary,
  ];
  const error = datasets.find((d) => d.error)?.error;
  const loading = datasets.some((d) => d.loading);
  const x = numeric ? inputData.valueAt(selected) : undefined;
  const scale = p.weight ? scaleData.valueAt(selection.feature) : 1;
  const bias = p.bias ? biasData.valueAt(selection.feature) : 0;
  const stats = numeric
    ? !complete
      ? normalizationNumbers(summary.data)
      : layerNormStatistics(
          samples.map((i) => inputData.valueAt(i)),
          p.size,
          p.eps,
        )
    : null;
  const calculation = layerNormCalculation(stats, x, scale, bias);
  const y = outputData.valueAt(selected);
  const selectFeature = (feature: number) =>
    setSelected(selection.start + feature);
  const axes = p.input.axes.slice(-p.shape.length);
  const firstAxis = p.input.shape.length - p.shape.length;
  const gridFrame = {
    rows: Math.min(8, p.input.shape.at(-2) ?? 1),
    columns: Math.min(8, p.input.shape.at(-1)!),
  };
  const result = (value: number | undefined) =>
    numeric && (
      <strong title={String(value ?? "unavailable")}>
        {formatValue(value)}
      </strong>
    );
  return (
    <section
      className="layer-normalization-lesson"
      aria-label="Layer normalization lesson"
    >
      <header className="normalization-heading">
        <div>
          <code>y = (x − μ) / √(σ² + ε) × γ + β</code>
          <p>
            Each trailing group has its own mean and variance. Scale and bias,
            when supplied, are shared across groups.
          </p>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div className="normalization-controls">
        <IndexControl
          label="Group"
          name="Normalization group"
          value={selection.group}
          size={p.groups}
          onChange={(group) => setSelected(group * p.size + selection.feature)}
        />
        <IndexControl
          label="Feature in group"
          name="Normalization feature"
          value={selection.feature}
          size={p.size}
          onChange={selectFeature}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="normalization-location" aria-live="polite">
        <code>
          x{coord(selection.coordinates)} → y{coord(selection.coordinates)}
        </code>
        <span>
          {p.size.toLocaleString()} elements per group ·{" "}
          {p.groups.toLocaleString()} groups · ε = {p.eps}
        </span>
        <span>
          Normalized axes:{" "}
          {p.shape
            .map(
              (n, i) =>
                `${axes[i] || `axis ${firstAxis + i}`} (${n.toLocaleString()})`,
            )
            .join(" × ")}
        </span>
        <small>
          {selection.prefix.length
            ? `Fixed leading coordinates ${coord(selection.prefix)}`
            : "The entire input is one normalization group"}
          . Parameter coordinate {coord(selection.parameterCoordinates)}.
        </small>
      </div>
      <div className="normalization-tensors">
        <TensorCard
          tensor={p.input}
          label="Normalization input"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={selected}
          highlights={samples}
          onSelect={setSelected}
        />
        <ArrowRight className="normalization-direction" size={20} />
        <TensorCard
          tensor={p.output}
          label="Normalized output"
          tone="output"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={selected}
          highlights={samples}
          onSelect={setSelected}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select either tensor, including its enlarged 3D view. Highlights cover{" "}
          {complete
            ? "the full group"
            : `${samples.length} selected group elements`}{" "}
          across the current slices. Each output depends on the whole group,
          even though its shape and coordinates stay the same.
        </p>
      </div>
      <section
        className="linear-calculation"
        aria-label="Layer normalization calculation"
      >
        <header>
          <div>
            <span className="eyebrow">How this cell is formed</span>
            <code>y{coord(selection.coordinates)}</code>
          </div>
          <span>Population variance · divide by {p.size.toLocaleString()}</span>
        </header>
        {!complete && numeric && (
          <p className="normalization-coverage" role="status">
            {summary.loading
              ? `Calculating statistics across all ${p.size.toLocaleString()} group elements…`
              : stats
                ? `Statistics cover all ${p.size.toLocaleString()} group elements. The visible window shows ${samples.length}.`
                : "Complete group statistics are unavailable."}
          </p>
        )}
        <ol className="normalization-progression">
          <li>
            <span className="eyebrow">01 · Mean</span>
            <code>μ = Σ xᵢ / N</code>
            {result(stats?.mean)}
            <p>One mean over the entire trailing group.</p>
          </li>
          <li>
            <span className="eyebrow">02 · Variance</span>
            <code>σ² = Σ (xᵢ − μ)² / N</code>
            {result(stats?.variance)}
            <p>Squared distances from the mean, divided by N.</p>
          </li>
          <li>
            <span className="eyebrow">03 · Normalize</span>
            <code>z = (x − μ) / √(σ² + ε)</code>
            {result(calculation?.normalized)}
            {numeric && stats && (
              <small>
                Centered ≈ {formatValue(calculation?.centered)}
                <br />
                Denominator ≈ {formatValue(stats.denominator)}
              </small>
            )}
            <p>Epsilon is inside the square root.</p>
          </li>
          <li>
            <span className="eyebrow">04 · Scale and shift</span>
            <code>y = z × γ + β</code>
            {result(calculation?.output)}
            <small>
              γ ={" "}
              {p.weight
                ? numeric
                  ? formatValue(scale)
                  : "recorded scale"
                : "1 (no scale)"}
              <br />β ={" "}
              {p.bias
                ? numeric
                  ? formatValue(bias)
                  : "recorded bias"
                : "0 (no bias)"}
            </small>
            <p>One parameter value per normalized coordinate.</p>
          </li>
        </ol>
        <div className="normalization-recorded">
          <div>
            <span className="eyebrow">Recorded PyTorch output</span>
            <code>y{coord(selection.coordinates)}</code>
          </div>
          {numeric ? (
            <strong title={String(y ?? "unavailable")}>{formatValue(y)}</strong>
          ) : (
            <span>
              {shapeOnly ? "Shapes only · no numeric values" : "Values hidden"}
            </span>
          )}
        </div>
        <p className="linear-numeric-note" role="status">
          {shapeOnly
            ? "This run records group boundaries and parameter coordinates without numeric statistics."
            : !showValues
              ? "Show values to inspect the calculation."
              : loading
                ? "Loading the complete group and selected parameters…"
                : summary.data?.status === "non_finite"
                  ? "This group contains a non-finite input, possibly outside the visible window. No partial-group statistics are substituted. Inspect the recorded output."
                  : summary.data?.status === "overflow"
                    ? "The complete group statistics overflow floating-point precision. Inspect the recorded output."
                    : !stats
                      ? "A required group value is unavailable, non-finite, or overflows this calculation. Inspect the recorded output."
                      : stats.denominator === 0
                        ? "Variance and epsilon are both zero. Division by zero has no finite normalized result; inspect the recorded output."
                        : calculation?.output === undefined
                          ? "A selected input or affine parameter is unavailable, non-finite, or overflows this calculation. Inspect the recorded output."
                          : "These are explanatory calculations from captured values, not additional execution nodes. Calculation precision can differ from PyTorch; the recorded output is authoritative."}
        </p>
        <p className="linear-numeric-note">
          Epsilon can keep the normalized variance below one. Scale and bias can
          change both mean and variance. LayerNorm uses current input statistics
          in evaluation mode too.
        </p>
        {numeric && error && (
          <div className="linear-load-error" role="alert">
            {error}{" "}
            <button
              className="text-button"
              onClick={() => datasets.forEach((d) => d.retry())}
            >
              Retry calculation
            </button>
          </div>
        )}
        <details className="linear-terms">
          <summary>
            Inspect group elements{" "}
            <span>
              {window[0] - selection.start}–{window.at(-1)! - selection.start} /{" "}
              {p.size.toLocaleString()}
            </span>
          </summary>
          <div className="linear-term-navigation">
            <p>Feature indices follow the original trailing-axis order.</p>
            <IndexControl
              label="Group window"
              name="Normalization group window"
              value={window[0] - selection.start}
              size={p.size}
              step={8}
              onChange={selectFeature}
            />
          </div>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Input coordinate</th>
                  <th>Input</th>
                  <th>Centered</th>
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
                        aria-label={`Inspect normalization feature ${index - selection.start}`}
                        aria-pressed={index === selected}
                        onClick={() => setSelected(index)}
                      >
                        {index - selection.start}
                      </button>
                    </th>
                    <td>{coord(unravel(index, p.input.shape))}</td>
                    <td>
                      {numeric ? formatValue(inputData.valueAt(index)) : "—"}
                    </td>
                    <td>
                      {numeric
                        ? formatValue(
                            layerNormCalculation(
                              stats,
                              inputData.valueAt(index),
                              1,
                              0,
                            )?.centered,
                          )
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>
      {(p.weight || p.bias) && (
        <details className="normalization-parameters">
          <summary>
            Inspect scale and bias{" "}
            <span>{coord(selection.parameterCoordinates)}</span>
          </summary>
          <p>
            Select a parameter to follow its feature while keeping the current
            group.
          </p>
          <div className="normalization-parameter-tensors">
            {p.weight && (
              <TensorCard
                tensor={p.weight}
                label="Scale γ"
                gridFrame={{
                  rows: Math.min(8, p.shape.at(-2) ?? 1),
                  columns: Math.min(8, p.shape.at(-1)!),
                }}
                runId={run.id}
                showValues={showValues}
                focusIndex={selection.feature}
                highlights={[selection.feature]}
                onSelect={selectFeature}
              />
            )}
            {p.bias && (
              <TensorCard
                tensor={p.bias}
                label="Bias β"
                gridFrame={{
                  rows: Math.min(8, p.shape.at(-2) ?? 1),
                  columns: Math.min(8, p.shape.at(-1)!),
                }}
                runId={run.id}
                showValues={showValues}
                focusIndex={selection.feature}
                highlights={[selection.feature]}
                onSelect={selectFeature}
              />
            )}
          </div>
        </details>
      )}
    </section>
  );
}
