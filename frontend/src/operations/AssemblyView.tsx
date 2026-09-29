import { Fragment, useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { IndexControl } from "../tensors/TensorNavigation";
import { formatValue, unravel } from "../tensors/coordinates";
import { useTensorValues } from "../tensors/useTensorValues";
import {
  locatePart,
  locateWhole,
  visibleParts,
  type TensorAssembly,
} from "./assembly";

const coord = (values: number[]) => `[${values.join(", ")}]`;
export function AssemblyView({
  assembly: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  assembly: TensorAssembly;
  run: Run;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onDetails: () => void;
}) {
  const [selection, setSelection] = useState({
    part: Math.max(
      0,
      p.parts.findIndex((t) => t.numel > 0),
    ),
    index: 0,
  });
  const part = p.parts[selection.part],
    empty = !part.numel;
  const selectedWhole = empty
    ? null
    : locateWhole(p, selection.part, selection.index);
  const selectedPart = empty ? null : unravel(selection.index, part.shape);
  const shapeOnly = p.whole.value_source === "shape",
    numeric = showValues && !shapeOnly;
  const partData = useTensorValues(
    part,
    run.id,
    numeric && !empty ? [selection.index] : [],
  );
  const wholeData = useTensorValues(
    p.whole,
    run.id,
    numeric && selectedWhole ? [selectedWhole.index] : [],
  );
  const error = partData.error || wholeData.error;
  const partLabel = `${p.joining ? "Input" : "Output"} ${selection.part}`;
  const wholeLabel = p.joining ? "Joined tensor" : "Original tensor";
  const choosePart = (part: number) => setSelection({ part, index: 0 });
  const visible = visibleParts(p.parts.length, selection.part);
  const gridFrame = {
    rows: Math.max(
      1,
      ...[part, p.whole].map((t) => Math.min(8, t.shape.at(-2) ?? 1)),
    ),
    columns: Math.max(
      1,
      ...[part, p.whole].map((t) => Math.min(8, t.shape.at(-1) ?? 1)),
    ),
  };
  const partCard = (
    <TensorCard
      key={`part-${selection.part}`}
      tensor={part}
      label={partLabel}
      runId={run.id}
      gridFrame={gridFrame}
      showValues={showValues}
      focusIndex={empty ? undefined : selection.index}
      highlights={empty ? [] : [selection.index]}
      onSelect={(index) => setSelection({ ...selection, index })}
      tone={p.joining ? undefined : "output"}
    />
  );
  const wholeCard = (
    <TensorCard
      tensor={p.whole}
      label={wholeLabel}
      runId={run.id}
      gridFrame={gridFrame}
      showValues={showValues}
      focusIndex={selectedWhole?.index}
      highlights={selectedWhole ? [selectedWhole.index] : []}
      onSelect={(index) => setSelection(locatePart(p, index))}
      tone={p.joining ? "output" : undefined}
    />
  );
  const partRange = p.inserted
    ? `Axis ${p.axis} index ${selection.part}`
    : empty
      ? `Empty interval at ${p.offsets[selection.part]}`
      : `Axis ${p.axis} indices ${p.offsets[selection.part]}–${p.ends[selection.part] - 1}`;
  return (
    <section
      className="assembly-lesson"
      aria-label="Tensor join and split lesson"
    >
      <header className="assembly-heading">
        <p>
          {p.inserted
            ? `Axis ${p.axis} chooses one of ${p.parts.length} tensors. The other coordinates stay the same.`
            : `The ${p.parts.length} parts occupy consecutive ranges on axis ${p.axis}. Other coordinates stay the same.`}{" "}
          No values are added together.
        </p>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div className="assembly-controls">
        <IndexControl
          label={`${p.parts.length} recorded ${p.joining ? "inputs" : "outputs"}`}
          name="Inspect tensor part"
          value={selection.part}
          size={p.parts.length}
          onChange={choosePart}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div
        className="assembly-parts"
        role="group"
        aria-label="Ordered tensor parts"
      >
        {visible.map((i, n) => (
          <Fragment key={i}>
            {n > 0 && i > visible[n - 1] + 1 && (
              <button
                className="assembly-gap"
                aria-label={`Inspect omitted part ${visible[n - 1] + 1} through ${i - 1}`}
                onClick={() => choosePart(visible[n - 1] + 1)}
              >
                <b>…</b>
                <small>
                  {visible[n - 1] + 1}–{i - 1}
                </small>
              </button>
            )}
            <button
              className="assembly-part"
              aria-pressed={selection.part === i}
              onClick={() => choosePart(i)}
            >
              <span>
                {p.joining ? "Input" : "Output"} {i}
              </span>
              <code>{coord(p.parts[i].shape)}</code>
              <small>
                {p.inserted
                  ? `axis ${p.axis} = ${i}`
                  : !p.parts[i].numel
                    ? "empty"
                    : `${p.offsets[i]}–${p.ends[i] - 1} on axis ${p.axis}`}
              </small>
            </button>
          </Fragment>
        ))}
      </div>
      <div className="assembly-coordinate" aria-live="polite">
        <span>{partRange}</span>
        {selectedWhole && selectedPart && (
          <code>
            {p.joining
              ? `${partLabel}${coord(selectedPart)} → result${coord(selectedWhole.coordinates)}`
              : `input${coord(selectedWhole.coordinates)} → ${partLabel}${coord(selectedPart)}`}
          </code>
        )}
      </div>
      <div className="assembly-tensors">
        {p.joining ? partCard : wholeCard}
        <ArrowRight className="assembly-direction" size={20} />
        {p.joining ? wholeCard : partCard}
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select a cell on either side, including in 3D. Selecting the whole
          tensor opens the corresponding part automatically. Indices are
          zero-based.
        </p>
      </div>
      <div
        className="assembly-readout"
        aria-label="Recorded join or split values"
      >
        {empty ? (
          <p>
            This part has no cells. The whole tensor is unchanged by an empty
            interval.
          </p>
        ) : (
          <>
            {numeric && (
              <p>
                <span>
                  {p.joining ? "Source" : "Original"} value{" "}
                  <b
                    title={String(
                      p.joining
                        ? partData.valueAt(selection.index)
                        : wholeData.valueAt(selectedWhole!.index),
                    )}
                  >
                    {formatValue(
                      p.joining
                        ? partData.valueAt(selection.index)
                        : wholeData.valueAt(selectedWhole!.index),
                    )}
                  </b>
                </span>
                <ArrowRight size={15} />
                <span>
                  Recorded result{" "}
                  <b
                    title={String(
                      p.joining
                        ? wholeData.valueAt(selectedWhole!.index)
                        : partData.valueAt(selection.index),
                    )}
                  >
                    {formatValue(
                      p.joining
                        ? wholeData.valueAt(selectedWhole!.index)
                        : partData.valueAt(selection.index),
                    )}
                  </b>
                </span>
              </p>
            )}
            <small>
              {part.storage_id === p.whole.storage_id
                ? "The selected snapshots share recorded storage. Their logical indices can differ."
                : "The selected snapshots use separate recorded storage."}{" "}
              {shapeOnly && "This run records coordinates and shapes only."}
            </small>
          </>
        )}
        {numeric && error && (
          <p className="field-error" role="alert">
            {error}
            <button
              className="text-button"
              onClick={() => {
                partData.retry();
                wholeData.retry();
              }}
            >
              Retry values
            </button>
          </p>
        )}
      </div>
    </section>
  );
}
