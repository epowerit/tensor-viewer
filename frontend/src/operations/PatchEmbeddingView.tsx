import { useState } from "react";
import { ArrowRight, Grid2X2, Layers3, MousePointer2 } from "lucide-react";
import type { Run, Tensor } from "../api/client";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatCellValue, formatValue } from "../tensors/coordinates";
import {
  patchCell,
  patchGridWindow,
  patchPosition,
  type PatchJourney,
} from "./patches";

type Props = {
  journey: PatchJourney;
  run: Run;
  operationId: string;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onDetails: () => void;
};

function cellLabel(value: number | string | undefined, length = 5) {
  const label = formatCellValue(value, length);
  return label.length <= length ? label : "…";
}

export function PatchEmbeddingView({
  journey: j,
  run,
  operationId,
  showValues,
  onShowValues,
  onDetails,
}: Props) {
  const [batch, setBatch] = useState(0);
  const [token, setToken] = useState(0);
  const [feature, setFeature] = useState(0);
  const [channel, setChannel] = useState(0);
  const [pixel, setPixel] = useState(0);
  const [ph, pw] = j.patch;
  const [batches, channels, height, width] = j.image.shape;
  const [, features, gridRows, gridColumns] = j.projected.shape;
  const count = gridRows * gridColumns,
    terms = channels * ph * pw;
  const position = patchPosition(j, token),
    grid = patchGridWindow(j, token);
  const dy = Math.floor(pixel / pw),
    dx = pixel % pw;
  const rowStart = Math.floor(dy / 8) * 8,
    colStart = Math.floor(dx / 8) * 8;
  const rows = Math.min(8, ph - rowStart),
    columns = Math.min(8, pw - colStart);
  const cells = Array.from({ length: rows * columns }, (_, i) => {
    const y = rowStart + Math.floor(i / columns),
      x = colStart + (i % columns);
    return { y, x, ...patchCell(j, batch, token, feature, channel, y, x) };
  });
  const selected = patchCell(j, batch, token, feature, channel, dy, dx);
  const featureStart = Math.floor(feature / 8) * 8;
  const featureIndices = Array.from(
    { length: Math.min(8, features - featureStart) },
    (_, i) => featureStart + i,
  );
  const result = j.tokens ?? j.projected;
  const resultIndex = (f: number) => {
    const cell = patchCell(j, batch, token, f, channel, dy, dx);
    return cell.token ?? cell.projected;
  };
  const inputData = useTensorValues(
    j.image,
    run.id,
    cells.map((c) => c.input),
  );
  const weightData = useTensorValues(
    j.weight,
    run.id,
    cells.map((c) => c.weight),
  );
  const outputData = useTensorValues(
    result,
    run.id,
    featureIndices.map(resultIndex),
  );
  const biasData = useTensorValues(
    j.bias ?? j.weight,
    run.id,
    j.bias ? [feature] : [],
  );
  const shapeOnly = j.image.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const x = inputData.valueAt(selected.input),
    w = weightData.valueAt(selected.weight),
    y = outputData.valueAt(resultIndex(feature));
  const contribution =
    typeof x === "number" &&
    typeof w === "number" &&
    Number.isFinite(x) &&
    Number.isFinite(w)
      ? x * w
      : undefined;
  const stages: {
    title: string;
    tensor: Tensor;
    coordinates: number[];
    operation?: string;
    caption: string;
  }[] = [
    {
      title: "Image",
      tensor: j.image,
      coordinates: selected.inputCoordinates,
      caption: "B · C · H · W",
    },
    {
      title: "Project patches",
      tensor: j.projected,
      coordinates: [batch, feature, position.row, position.column],
      operation: j.projection.id,
      caption: "B · D · patch rows · patch cols",
    },
    ...(j.flattened
      ? [
          {
            title: "Flatten grid",
            tensor: j.flattened,
            coordinates: [batch, feature, token],
            operation: j.flatten?.id,
            caption: "B · D · N",
          },
        ]
      : []),
    ...(j.tokens
      ? [
          {
            title: "Token rows",
            tensor: j.tokens,
            coordinates: [batch, token, feature],
            operation: j.transpose?.id,
            caption: "B · N · D",
          },
        ]
      : []),
  ];

  function renderCells(kind: "input" | "weight") {
    const data = kind === "input" ? inputData : weightData;
    return (
      <div
        className="patch-matrix"
        role="group"
        aria-label={
          kind === "input"
            ? "Selected patch pixels"
            : "Matching projection weights"
        }
      >
        <div className="patch-matrix-label">
          {kind === "input" ? "Input pixels" : `Kernel · feature ${feature}`}
        </div>
        <div
          className="patch-cell-grid"
          style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
        >
          {cells.map((c) => {
            const coordinates =
              kind === "input" ? c.inputCoordinates : c.weightCoordinates;
            const value = data.valueAt(c[kind]);
            return (
              <button
                key={c[kind]}
                className={pixel === c.y * pw + c.x ? "selected" : ""}
                style={{
                  fontSize: `min(11px, calc((100cqw / ${columns} - 4px) / 3.1))`,
                }}
                aria-label={`${kind === "input" ? "Patch pixel" : "Patch weight"} [${coordinates.join(", ")}]${numeric ? ` = ${value ?? "loading"}` : ""}`}
                aria-pressed={pixel === c.y * pw + c.x}
                title={`[${coordinates.join(", ")}]${numeric ? ` = ${value ?? "Loading…"}` : ""}`}
                onClick={() => setPixel(c.y * pw + c.x)}
                onKeyDown={(e) => {
                  const next =
                    e.key === "ArrowRight"
                      ? Math.min(pw - 1, dx + 1) + dy * pw
                      : e.key === "ArrowLeft"
                        ? Math.max(0, dx - 1) + dy * pw
                        : e.key === "ArrowDown"
                          ? Math.min(ph - 1, dy + 1) * pw + dx
                          : e.key === "ArrowUp"
                            ? Math.max(0, dy - 1) * pw + dx
                            : null;
                  if (next !== null) {
                    e.preventDefault();
                    setPixel(next);
                    const group = e.currentTarget.closest('[role="group"]');
                    requestAnimationFrame(() =>
                      group
                        ?.querySelector<HTMLButtonElement>(
                          '[aria-pressed="true"]',
                        )
                        ?.focus({ preventScroll: true }),
                    );
                  }
                }}
              >
                {numeric ? cellLabel(value) : `${c.y},${c.x}`}
              </button>
            );
          })}
        </div>
        <small>
          Channel {channel} ·{" "}
          {numeric
            ? kind === "input"
              ? "image values"
              : "recorded weights"
            : "patch coordinates"}
        </small>
      </div>
    );
  }

  return (
    <section className="patch-lesson" aria-label="Patch to token lesson">
      <header className="patch-lesson-heading">
        <div>
          <span className="eyebrow">FOLLOW A SPATIAL PATCH</span>
          <h3>
            {j.tokens ? "From pixels to a token" : "From pixels to features"}
          </h3>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          <Layers3 size={14} /> Tensor details
        </button>
      </header>
      <div
        className="patch-progression"
        aria-label="Recorded patch embedding progression"
      >
        {stages.map((stage, i) => (
          <div
            className={`patch-stage ${stage.operation === operationId ? "active" : ""}`}
            key={stage.tensor.id}
          >
            <span className="patch-stage-step">
              {String(i + 1).padStart(2, "0")}{" "}
              {i > 0 && <ArrowRight size={12} />}
            </span>
            <b>{stage.title}</b>
            <code>[{stage.tensor.shape.join(", ")}]</code>
            <small>{stage.caption}</small>
            <small className="patch-stage-coordinate">
              {i === 0 ? "Pixel" : "Feature"} [{stage.coordinates.join(", ")}]
            </small>
          </div>
        ))}
      </div>
      <div className="patch-control-row">
        <IndexControl
          label="Batch"
          name="Patch batch index"
          value={batch}
          size={batches}
          onChange={setBatch}
        />
        <IndexControl
          label={j.tokens ? "Patch / token" : "Patch"}
          name="Patch token index"
          value={token}
          size={count}
          onChange={setToken}
        />
        <IndexControl
          label="Output feature"
          name="Patch output feature"
          value={feature}
          size={features}
          onChange={setFeature}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
          description="Show recorded pixel, kernel, and output values. Cell text is rounded; hover a pixel or weight for its full value. Color marks the selected position."
        />
      </div>
      <div className="patch-panels">
        <section
          className="patch-panel patch-image-panel"
          aria-label="Spatial patch positions"
        >
          <header>
            <Grid2X2 size={15} />
            <h4>Image regions</h4>
            <span>
              {gridRows.toLocaleString()} × {gridColumns.toLocaleString()}
            </span>
          </header>
          <p>
            One numbered tile is a {ph} × {pw} region across all{" "}
            {channels.toLocaleString()} channels.
          </p>
          <div
            className="patch-position-grid"
            style={{
              gridTemplateColumns: `repeat(${grid.columns}, minmax(0, 1fr))`,
            }}
          >
            {grid.tokens.map((t) => (
              <button
                key={t}
                aria-label={`Select spatial patch ${t}`}
                aria-pressed={token === t}
                className={token === t ? "selected" : ""}
                onClick={() => setToken(t)}
              >
                <span>{t.toLocaleString()}</span>
              </button>
            ))}
          </div>
          <small className="patch-window-note">
            Patch rows {grid.startRow}–{grid.startRow + grid.rows - 1} · cols{" "}
            {grid.startColumn}–{grid.startColumn + grid.columns - 1}
            {grid.tokens.length < count
              ? ` · ${grid.tokens.length} of ${count.toLocaleString()} patches shown`
              : " · complete grid"}
          </small>
          <div className="patch-region-readout">
            <span>SELECTED IMAGE REGION</span>
            <b>
              Rows {position.y}–{position.y + ph - 1} · cols {position.x}–
              {position.x + pw - 1}
            </b>
            <small>
              Inside {height.toLocaleString()} × {width.toLocaleString()} pixels
              · batch {batch}
            </small>
          </div>
          <code className="patch-order-formula">
            token = {position.row} × {gridColumns} + {position.column} = {token}
          </code>
          <p className="patch-footnote">
            Token order follows patch rows, then columns.
          </p>
        </section>
        <section
          className="patch-panel patch-projection-panel"
          aria-label="Patch projection contributors"
        >
          <header>
            <Layers3 size={15} />
            <h4>Project this patch</h4>
            <span>{terms.toLocaleString()} terms</span>
          </header>
          <p>
            Every output feature combines all {channels} × {ph} × {pw} entries
            with its own kernel weights{j.bias ? " and bias" : ""}.
          </p>
          <IndexControl
            label="Input channel"
            name="Patch input channel"
            value={channel}
            size={channels}
            onChange={setChannel}
          />
          <div className="patch-matrix-pair">
            {renderCells("input")}
            <span className="patch-multiply">×</span>
            {renderCells("weight")}
          </div>
          {(ph > 8 || pw > 8) && (
            <div className="patch-pixel-navigation">
              {ph > 8 && (
                <IndexControl
                  label="Pixel row"
                  name="Patch pixel row"
                  value={dy}
                  size={ph}
                  onChange={(v) => setPixel(v * pw + dx)}
                />
              )}
              {pw > 8 && (
                <IndexControl
                  label="Pixel column"
                  name="Patch pixel column"
                  value={dx}
                  size={pw}
                  onChange={(v) => setPixel(dy * pw + v)}
                />
              )}
            </div>
          )}
          <small className="patch-window-note">
            Patch rows {rowStart}–{rowStart + rows - 1} · cols {colStart}–
            {colStart + columns - 1} · {rows * columns} of {ph * pw} positions
            in this channel
          </small>
          <div className="patch-contribution" aria-live="polite">
            <span>
              ONE CONTRIBUTION · INDEX {selected.patchVectorIndex} /{" "}
              {(terms - 1).toLocaleString()}
            </span>
            <code>
              x[{selected.inputCoordinates.join(", ")}] × W[
              {selected.weightCoordinates.join(", ")}]
            </code>
            {numeric && (
              <strong>
                {formatValue(x)} × {formatValue(w)}
                {contribution !== undefined
                  ? ` ≈ ${formatValue(contribution)}`
                  : ""}
              </strong>
            )}
            <p>
              Sum all {terms.toLocaleString()} contributions
              {j.bias
                ? `, then add bias${numeric ? ` ${formatValue(biasData.valueAt(feature))}` : ""}`
                : ""}
              . This selected term is only one part of the result.
            </p>
          </div>
        </section>
        <section
          className="patch-panel patch-token-panel"
          aria-label="Projected token features"
        >
          <header>
            <ArrowRight size={15} />
            <h4>{j.tokens ? `Token ${token}` : `Patch ${token} features`}</h4>
            <span>{features.toLocaleString()} features</span>
          </header>
          <p>
            {j.tokens
              ? "Flatten and transpose place this patch’s projected features in one token row. Values are unchanged by those layout steps."
              : "The recorded convolution stores these features at this spatial grid position."}
          </p>
          <div className="patch-feature-list">
            {featureIndices.map((f) => (
              <button
                key={f}
                aria-label={`Select patch feature ${f}`}
                aria-pressed={f === feature}
                className={f === feature ? "selected" : ""}
                onClick={() => setFeature(f)}
              >
                <span>Feature {f}</span>
                <code>
                  {numeric
                    ? cellLabel(outputData.valueAt(resultIndex(f)), 8)
                    : "—"}
                </code>
              </button>
            ))}
          </div>
          <small className="patch-window-note">
            Features {featureStart}–{featureStart + featureIndices.length - 1}{" "}
            of {features.toLocaleString()}
          </small>
          <div className="patch-result-readout">
            <span>
              {shapeOnly ? "SHAPE ONLY · NO NUMERIC VALUES" : "RECORDED OUTPUT"}
            </span>
            <code>
              [
              {(j.tokens
                ? [batch, token, feature]
                : [batch, feature, position.row, position.column]
              ).join(", ")}
              ]
            </code>
            {numeric && (
              <strong title="Full recorded value">
                {y === undefined ? "Loading…" : String(y)}
              </strong>
            )}
          </div>
        </section>
      </div>
      {[inputData, weightData, outputData, biasData].some(
        (data) => data.error,
      ) && (
        <div className="patch-load-error" role="alert">
          Some recorded values could not be loaded.
          <button
            className="text-button"
            onClick={() =>
              [inputData, weightData, outputData, biasData].forEach((data) =>
                data.retry(),
              )
            }
          >
            Retry values
          </button>
        </div>
      )}
      <footer className="patch-lesson-footer">
        <MousePointer2 size={14} />
        <p>
          Select a patch, pixel, or feature to follow its role. Pixel windows
          are views into the recorded input; the convolution computes the
          projection in one operation.
        </p>
      </footer>
    </section>
  );
}
