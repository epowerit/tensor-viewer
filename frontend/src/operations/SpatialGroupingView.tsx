import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import {
  groupingNeighborhood,
  groupingSelection,
  type SpatialGrouping,
} from "./spatialGrouping";
import { kindName } from "./kindName";

export function SpatialGroupingView({
  grouping: p,
  run,
  showValues,
  onShowValues,
  onStep,
  onDetails,
}: {
  grouping: SpatialGrouping;
  run: Run;
  showValues: boolean;
  onShowValues: (value: boolean) => void;
  onStep: (operation: string) => void;
  onDetails?: () => void;
}) {
  const [output, setOutput] = useState(0);
  const selected = groupingSelection(p, output),
    neighborhood = groupingNeighborhood(p, output);
  const input = p.tensors[0],
    result = p.tensors[4];
  const grid = p.kind === "restore" ? result : input;
  const [, features, height, width] = grid.shape;
  const count = (height / p.size) * (width / p.size);
  const labels =
    p.kind === "restore"
      ? ["Window sequences", "Restored grid"]
      : p.kind === "merge"
        ? ["Spatial grid", "Four neighbor vectors"]
        : ["Spatial grid", "Window sequences"];
  const frame = {
    rows: Math.max(
      ...[input, result].map((t) => Math.min(8, t.shape.at(-2) ?? 1)),
    ),
    columns: Math.max(
      ...[input, result].map((t) => Math.min(8, t.shape.at(-1) ?? 1)),
    ),
  };
  const coordinate = (values: number[]) => `[${values.join(", ")}]`;
  return (
    <section
      className="spatial-grouping-lesson"
      aria-label="Spatial grouping lesson"
    >
      <header className="linear-heading">
        <div>
          <code>
            {p.kind === "merge"
              ? "2 × 2 neighbors → 4C features"
              : p.kind === "restore"
                ? "Window sequences → spatial grid"
                : `${p.size} × ${p.size} grid cells → one window`}
          </code>
          <p>
            {p.kind === "merge"
              ? "Grouping keeps every value. A patch-merging layer can then normalize each 4C vector and project it to 2C."
              : `${count.toLocaleString()} windows per image, ${p.size * p.size} tokens per window. Windows are packed into the leading axis; images and windows remain independent.`}
          </p>
        </div>
        {onDetails && (
          <button className="secondary-button" onClick={onDetails}>
            Tensor details <ArrowRight size={13} />
          </button>
        )}
      </header>
      <div
        className="grouping-progression"
        aria-label="Recorded layout progression"
      >
        <div className="grouping-step">
          <span>Input</span>
          <code>{coordinate(input.shape)}</code>
        </div>
        {p.operations.map((op, i) => (
          <button
            key={op.id}
            className="grouping-step"
            onClick={() => onStep(op.id)}
            title={`Go to recorded ${kindName(op.kind)}, step ${op.index + 1}`}
          >
            <span>
              <ArrowRight size={12} />
              {kindName(op.kind)}
            </span>
            <code>{coordinate(p.tensors[i + 1].shape)}</code>
          </button>
        ))}
      </div>
      <div className="grouping-selection-bar">
        <div>
          <span>
            Batch <b>{neighborhood.batch}</b>
          </span>
          <span>
            {p.kind === "merge" ? "Neighborhood" : "Window"}{" "}
            <b>
              [{neighborhood.row}, {neighborhood.column}]
            </b>
          </span>
          <span>
            Feature <b>{neighborhood.feature}</b>
          </span>
        </div>
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={input.value_source === "shape"}
        />
      </div>
      <div className="spatial-group-tensors">
        <TensorCard
          tensor={input}
          label={labels[0]}
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected.indices[0]}
          highlights={neighborhood.cells.map((c) => c.indices[0])}
          onSelect={(index) =>
            setOutput(groupingSelection(p, index, "input").indices[4])
          }
        />
        <TensorCard
          tensor={result}
          label={labels[1]}
          tone="output"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={output}
          highlights={neighborhood.cells.map((c) => c.indices[4])}
          onSelect={setOutput}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select any cell, including in 3D. Highlights follow one feature across
          the selected neighborhood
          {p.size > 8 ? " (at most 8 × 8 cells at once)" : ""}. The selected
          element keeps its exact value through all four layout operations.
        </p>
      </div>
      <section
        className="grouping-coordinate-path"
        aria-label="Selected grouping coordinates"
      >
        <div>
          <span>Original coordinate</span>
          <code>{coordinate(selected.coordinates[0])}</code>
        </div>
        <ArrowRight size={16} />
        <div>
          <span>Result coordinate</span>
          <code>{coordinate(selected.coordinates[4])}</code>
        </div>
        <p>
          {p.kind === "merge"
            ? `Neighbor order: top-left, bottom-left, top-right, bottom-right. Selected neighbor ${neighborhood.localColumn * 2 + neighborhood.localRow} × ${features} features + feature ${neighborhood.feature} = feature slot ${(neighborhood.localColumn * 2 + neighborhood.localRow) * features + neighborhood.feature}.`
            : `Window-batch index = ${neighborhood.batch} × ${count} + ${neighborhood.row * (width / p.size) + neighborhood.column}. Local token = ${neighborhood.localRow} × ${p.size} + ${neighborhood.localColumn}.`}
        </p>
        <small>
          {input.storage_id === result.storage_id
            ? "The recorded endpoints share storage."
            : "The recorded endpoints use different storage; the layout path preserves values."}
        </small>
      </section>
    </section>
  );
}
