import { tensorSelection } from "./selection";
import { useState } from "react";
import { ArrowRight, ArrowUpRight, GitBranch } from "lucide-react";
import type { Operation, Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { tensorProducer } from "../tensors/provenance";
import { changedCells } from "./mutation";

export function MutationView({
  operation,
  run,
  showValues,
  onShowValues,
  onJump,
  onDetails,
  initialTensorId,
  initialCell,
}: {
  operation: Operation;
  initialTensorId?: string;
  initialCell?: number;
  run: Run;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onJump: (index: number) => void;
  onDetails: () => void;
}) {
  const effects = [...(operation.mutations ?? [])].sort(
    (a, b) => Number(a.kind === "alias") - Number(b.kind === "alias"),
  );
  const initial = tensorSelection(
    effects.map((effect) => run.trace.tensors[effect.after]),
    initialTensorId,
    initialCell,
  );
  const [choice, setChoice] = useState(initial.choice);
  const [selected, setSelected] = useState(initial.index);
  const effect = effects[choice] ?? effects[0];
  const before = run.trace.tensors[effect.before],
    after = run.trace.tensors[effect.after];
  const producer = tensorProducer(
    run.trace.operations,
    before.id,
    operation.index,
  );
  const aliasCount = effects.filter((e) => e.kind === "alias").length;
  const aligned =
    before.dtype === after.dtype &&
    JSON.stringify(before.shape) === JSON.stringify(after.shape);
  // With both snapshots recorded, the cells that differ are the write itself.
  const changed =
    aligned && effect.kind !== "metadata" ? changedCells(before, after) : null;
  const marked = changed?.count ? changed.indices : [selected];
  const frame = {
    rows: Math.max(
      1,
      ...[before, after].map((t) => Math.min(8, t.shape.at(-2) ?? 1)),
    ),
    columns: Math.max(
      1,
      ...[before, after].map((t) => Math.min(8, t.shape.at(-1) ?? 1)),
    ),
  };
  return (
    <section className="mutation-view" aria-label="In-place tensor changes">
      <div className="mutation-toolbar">
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={run.project.capture_mode === "shapes"}
        />
        <button className="text-button" onClick={onDetails}>
          Operation operands <ArrowUpRight size={13} />
        </button>
      </div>
      <div className="mutation-selection">
        <div className="mutation-summary">
          <GitBranch size={16} />
          <div>
            <h3>
              {effect.kind === "metadata"
                ? "Layout update"
                : "One write, shared storage"}
            </h3>
            <p>
              {aliasCount
                ? `${aliasCount} recorded ${aliasCount === 1 ? "view shares" : "views share"} the written storage.`
                : "Inspect the tensor before and after this operation."}
            </p>
          </div>
        </div>
        <label>
          Inspect affected tensor
          <select
            value={choice}
            onChange={(e) => {
              setChoice(Number(e.target.value));
              setSelected(0);
            }}
          >
            {effects.map((e, i) => (
              <option key={e.after} value={i}>
                {run.trace.tensors[e.before].name} ·{" "}
                {e.kind === "alias"
                  ? "shared view"
                  : e.kind === "metadata"
                    ? "layout changed"
                    : "written tensor"}{" "}
                [{run.trace.tensors[e.after].shape.join(", ")}]
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="mutation-context">
        {effect.kind === "alias"
          ? "This view was not another operation result. It still refers to the same storage, so it sees the recorded write. Individual cells may be unchanged when slices do not overlap."
          : effect.kind === "metadata"
            ? "Only this tensor’s layout or storage binding changes. Other views keep their own shape and strides."
            : "The operation updates this tensor in place. The earlier snapshot stays unchanged in the journey."}
      </p>
      <div className="mutation-comparison">
        <TensorCard
          tensor={before}
          runId={run.id}
          label="Before"
          showValues={showValues}
          gridFrame={frame}
          focusIndex={aligned ? selected : undefined}
          highlights={aligned ? marked : []}
          onSelect={aligned ? setSelected : undefined}
        />
        <div className="mutation-arrow">
          <span>{effect.kind === "metadata" ? "layout" : "write"}</span>
          <ArrowRight size={22} />
        </div>
        <TensorCard
          tensor={after}
          runId={run.id}
          label="After"
          tone="output"
          showValues={showValues}
          gridFrame={frame}
          focusIndex={aligned ? selected : undefined}
          highlights={aligned ? marked : []}
          onSelect={aligned ? setSelected : undefined}
        />
      </div>
      {changed && (
        <p className="mutation-context" role="status">
          {changed.count
            ? `${changed.count.toLocaleString()} of ${after.numel.toLocaleString()} cells changed value; they are highlighted in both snapshots${changed.count > changed.indices.length ? ` (the first ${changed.indices.length})` : ""}.`
            : "No cell of this tensor changed value: it shares the written storage, but the write touched other elements or stored the same values."}
        </p>
      )}
      <div className="mutation-footer">
        <span>
          {aligned
            ? "Select a cell to compare the same coordinate."
            : "Inspect each layout with its own coordinates."}{" "}
          Earlier snapshots are preserved.
        </span>
        {producer ? (
          <button
            className="text-button"
            onClick={() => onJump(producer.index)}
          >
            Before: step {producer.index + 1} · {producer.kind}
            <ArrowUpRight size={12} />
          </button>
        ) : (
          <span>
            Before: {before.name} · {before.role}
          </span>
        )}
      </div>
      {!operation.outputs.length && (
        <p className="mutation-context">
          This operation returned no tensor. These are its recorded side
          effects.
        </p>
      )}
      {operation.error && (
        <p className="field-error" role="alert">
          The operation failed after these observed changes: {operation.error}
        </p>
      )}
    </section>
  );
}
