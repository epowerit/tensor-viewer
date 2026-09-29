import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue } from "../tensors/coordinates";
import {
  contributionSum,
  finiteProduct,
  linearInputSelection,
  linearSelection,
  linearWeightSelection,
  linearWindow,
  type LinearProjection,
} from "./linear";

type Props = {
  projection: LinearProjection;
  run: Run;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onDetails: () => void;
};

const coordinate = (values: number[]) => `[${values.join(", ")}]`;

export function LinearProjectionView({
  projection: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: Props) {
  const [cursor, setCursor] = useState({ output: 0, term: 0 });
  const selected = linearSelection(p, cursor.output, cursor.term);
  const window = linearWindow(p, cursor.output, cursor.term);
  const complete = p.features <= 256;
  const samples = complete
    ? Array.from({ length: p.features }, (_, term) =>
        linearSelection(p, cursor.output, term),
      )
    : window;
  const shapeOnly = p.input.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const inputData = useTensorValues(
    p.input,
    run.id,
    numeric ? samples.map((s) => s.input) : [],
  );
  const weightData = useTensorValues(
    p.weight,
    run.id,
    numeric ? samples.map((s) => s.weight) : [],
  );
  const biasData = useTensorValues(
    p.bias ?? p.weight,
    run.id,
    numeric && p.bias ? [selected.feature] : [],
  );
  const outputData = useTensorValues(
    p.output,
    run.id,
    numeric ? [cursor.output] : [],
  );
  const datasets = [
    inputData,
    weightData,
    outputData,
    ...(p.bias ? [biasData] : []),
  ];
  const error = datasets.find((d) => d.error)?.error;
  const loading = datasets.some((d) => d.loading);
  const x = inputData.valueAt(selected.input),
    w = weightData.valueAt(selected.weight);
  const y = outputData.valueAt(cursor.output);
  const bias = p.bias ? biasData.valueAt(selected.feature) : 0;
  const product = finiteProduct(x, w);
  const sum = contributionSum(
    samples.map((s) =>
      finiteProduct(inputData.valueAt(s.input), weightData.valueAt(s.weight)),
    ),
  );
  const reconstructed =
    complete &&
    typeof bias === "number" &&
    Number.isFinite(bias) &&
    sum !== undefined &&
    Number.isFinite(sum + bias)
      ? sum + bias
      : undefined;
  const gridFrame = {
    rows: Math.max(
      ...[p.input, p.weight, p.output].map((t) =>
        Math.min(8, t.shape.at(-2) ?? 1),
      ),
    ),
    columns: Math.max(
      ...[p.input, p.weight, p.output].map((t) =>
        Math.min(8, t.shape.at(-1) ?? 1),
      ),
    ),
  };
  const selectTerm = (term: number) => setCursor((c) => ({ ...c, term }));
  const selectFeature = (feature: number) =>
    setCursor((c) => ({
      ...c,
      output: Math.floor(c.output / p.outputs) * p.outputs + feature,
    }));
  const reference = `y${coordinate(selected.coordinates)}`;

  return (
    <section className="linear-lesson" aria-label="Linear projection lesson">
      <header className="linear-heading">
        <div>
          <code>Y = X Wᵀ{p.bias ? " + b" : ""}</code>
          <p>
            One input vector × one <strong>row</strong> of weights. Leading
            coordinates stay the same.
          </p>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div className="linear-controls">
        <IndexControl
          label="Output feature · j"
          name="Projection output feature"
          value={selected.feature}
          size={p.outputs}
          onChange={selectFeature}
        />
        <IndexControl
          label="Input feature · k"
          name="Projection input feature"
          value={cursor.term}
          size={p.features}
          onChange={selectTerm}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="linear-tensors">
        <TensorCard
          tensor={p.input}
          label="Input vector"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={selected.input}
          highlights={window.map((s) => s.input)}
          onSelect={(index) =>
            setCursor(linearInputSelection(p, index, cursor.output))
          }
        />
        <TensorCard
          tensor={{ ...p.weight, axes: ["out_features", "in_features"] }}
          label="Weights"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={selected.weight}
          highlights={window.map((s) => s.weight)}
          onSelect={(index) =>
            setCursor(linearWeightSelection(p, index, cursor.output))
          }
        />
        <TensorCard
          tensor={p.output}
          label="Projected output"
          tone="output"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={cursor.output}
          highlights={[cursor.output]}
          onSelect={(output) => setCursor((c) => ({ ...c, output }))}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select any output, input, or weight cell. The other views follow its
          coordinates. Highlighted pairs show input features {window[0].term}–
          {window.at(-1)!.term} of {p.features.toLocaleString()}.
        </p>
      </div>
      <section
        className="linear-calculation"
        aria-label="Projection calculation"
      >
        <header>
          <div>
            <span className="eyebrow">How this cell is formed</span>
            <code>{reference}</code>
          </div>
          <span>
            {p.features.toLocaleString()} products
            {p.bias ? " + bias" : " · no bias"}
          </span>
        </header>
        <div className="linear-equation">
          <div>
            <span>Selected product · k = {cursor.term}</span>
            <code>
              x{coordinate(selected.inputCoordinates)} × W
              {coordinate(selected.weightCoordinates)}
            </code>
            {numeric && (
              <strong title={`${x ?? "unavailable"} × ${w ?? "unavailable"}`}>
                {formatValue(x)} × {formatValue(w)} ≈ {formatValue(product)}
              </strong>
            )}
          </div>
          <div>
            <span>
              {complete
                ? "Sum of all products"
                : `Window sum · k = ${window[0].term}–${window.at(-1)!.term}`}
            </span>
            <code>Σ x[…, k] × W[{selected.feature}, k]</code>
            {numeric && (
              <strong>
                {sum === undefined ? "—" : `≈ ${formatValue(sum)}`}
              </strong>
            )}
            {!complete && (
              <small>Partial sum; other features also contribute.</small>
            )}
          </div>
          <div className="linear-recorded-result">
            <span>Recorded output</span>
            <code>{reference}</code>
            {numeric && (
              <strong title={String(y ?? "unavailable")}>
                {formatValue(y)}
              </strong>
            )}
            {p.bias && (
              <small
                title={numeric ? String(bias ?? "unavailable") : undefined}
              >
                Bias b[{selected.feature}]
                {numeric ? ` = ${formatValue(bias)}` : " is added once"}
              </small>
            )}
            {numeric && reconstructed !== undefined && (
              <small>
                Sum{p.bias ? " + bias" : ""} ≈ {formatValue(reconstructed)}
              </small>
            )}
          </div>
        </div>
        {numeric && (
          <p className="linear-numeric-note" role="status">
            {loading
              ? "Loading recorded values…"
              : error
                ? "Some values could not be loaded."
                : sum === undefined
                  ? "A required value is missing or non-finite; no numerical sum is shown."
                  : "Calculations use captured values; displayed numbers are rounded. Floating-point accumulation can differ from PyTorch."}
          </p>
        )}
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
        <details className="linear-terms">
          <summary>
            Inspect matching products{" "}
            <span>
              {window[0].term}–{window.at(-1)!.term} /{" "}
              {p.features.toLocaleString()} features
            </span>
          </summary>
          <div className="linear-term-navigation">
            <p>
              One input entry × one weight from row {selected.feature}. Indices
              are zero-based.
            </p>
            <IndexControl
              label="Contribution window"
              name="Projection contribution window"
              value={window[0].term}
              size={p.features}
              step={8}
              onChange={selectTerm}
            />
          </div>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>k</th>
                  <th>Input x[…, k]</th>
                  <th>Weight W[j, k]</th>
                  <th>Product</th>
                </tr>
              </thead>
              <tbody>
                {window.map((s) => {
                  const input = inputData.valueAt(s.input),
                    weight = weightData.valueAt(s.weight);
                  return (
                    <tr
                      key={s.term}
                      className={cursor.term === s.term ? "selected" : ""}
                    >
                      <th scope="row">
                        <button
                          aria-label={`Inspect input feature ${s.term}`}
                          aria-pressed={cursor.term === s.term}
                          onClick={() => selectTerm(s.term)}
                        >
                          {s.term}
                        </button>
                      </th>
                      <td
                        title={
                          numeric ? String(input ?? "unavailable") : undefined
                        }
                      >
                        {numeric
                          ? formatValue(input)
                          : coordinate(s.inputCoordinates)}
                      </td>
                      <td
                        title={
                          numeric ? String(weight ?? "unavailable") : undefined
                        }
                      >
                        {numeric
                          ? formatValue(weight)
                          : coordinate(s.weightCoordinates)}
                      </td>
                      <td>
                        {numeric
                          ? formatValue(finiteProduct(input, weight))
                          : "x × W"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </details>
      </section>
    </section>
  );
}
