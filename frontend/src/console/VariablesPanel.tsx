import { useContext, useMemo } from "react";
import { Link2 } from "lucide-react";
import type { Run } from "../api/client";
import { ShapeGlyph } from "../editor/ShapeGlyph";
import {
  describeAxis,
  explainedLineage,
  lineageOf,
  originSummary,
} from "../tensors/axisLineage";
import { AxisInkContext, InkShape } from "../tensors/InkShape";
import { variables, type Variable } from "./variables";
import "./tensorShelf.css";

type Props = {
  run: Run | null;
  selected: string | null;
  onSelect: (nodeId: string) => void;
  /** Hovering or focusing a card follows its name: every node that wrote it. */
  onThread?: (nodeIds: string[] | null) => void;
};

const formatValue = (value: number) =>
  Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01)
    ? value.toExponential(1)
    : value.toFixed(2);

/**
 * Every named tensor of the displayed run, like a debugger's locals: as it
 * stands at the selected step (or at the end of the run), with its range,
 * type and shared storage. Names assigned later wait, unlit.
 */
export function VariablesPanel({ run, selected, onSelect, onThread }: Props) {
  const through = run?.trace.operations.find((op) => op.id === selected)?.index;
  const items = useMemo(
    () => (run ? variables(run.trace, through ?? Infinity) : []),
    [run, through],
  );
  const lineage = useMemo(() => (run ? lineageOf(run.trace) : null), [run]);
  const inkFor = useContext(AxisInkContext);
  if (!items.length)
    return (
      <p className="tensor-shelf-empty">Tensors appear here after a run.</p>
    );
  return (
    <ul className="tensor-shelf-grid" aria-label="Recorded tensors">
      {items.map((item) => {
        const shape = item.tensor.shape.length
          ? `[${item.tensor.shape.join(", ")}]`
          : "scalar";
        const shared = item.sharedWith.join(", ");
        const stories = explainedLineage(
          item.tensor,
          lineage?.(item.tensor.id),
        );
        // Axes that only restate their own names are left out of the line.
        const origin = stories && originSummary(stories, item.tensor.axes);
        const originFull = stories?.map(describeAxis).join(" · ");
        const originDetail = stories
          ?.map(
            (story, axis) =>
              `${item.tensor.axes[axis] ?? `axis ${axis}`}: ${describeAxis(story)}`,
          )
          .join("\n");
        const { minimum, maximum } = item.tensor;
        const range =
          !item.pending &&
          typeof minimum === "number" &&
          typeof maximum === "number"
            ? `${formatValue(minimum)} … ${formatValue(maximum)}`
            : null;
        const step = run?.trace.operations.find((op) => op.id === item.nodeId);
        return (
          <li key={item.name}>
            <button
              type="button"
              className={`tensor-shelf-card${item.anonymous ? " tensor-shelf-anonymous" : ""}${item.pending ? " tensor-shelf-pending" : ""}${item.fresh ? " tensor-shelf-fresh" : ""}`}
              aria-pressed={item.nodeId === selected}
              aria-label={`Inspect ${item.name}, shape ${shape}, ${item.tensor.dtype}${item.pending ? `, not computed yet at this step` : ""}${item.fresh ? ", just written" : ""}${range ? `, values from ${range}` : ""}${originFull ? `, axes from ${originFull}` : ""}${shared ? `, shares storage with ${shared}` : ""}`}
              title={`${item.name} · ${shape}${item.line ? ` · line ${item.line}` : ""}`}
              onClick={() => onSelect(item.nodeId)}
              onMouseEnter={() =>
                onThread?.(item.history.map((state) => state.nodeId))
              }
              onMouseLeave={() => onThread?.(null)}
              onFocus={() =>
                onThread?.(item.history.map((state) => state.nodeId))
              }
              onBlur={() => onThread?.(null)}
            >
              <span className="tensor-shelf-glyph" aria-hidden="true">
                <ShapeGlyph shape={item.tensor.shape} />
              </span>
              <span className="tensor-shelf-identity">
                <span className="tensor-shelf-name">{item.name}</span>
                <code className="tensor-shelf-shape">
                  <InkShape
                    shape={item.tensor.shape}
                    ink={item.pending ? null : inkFor?.(item.tensor)}
                  />
                </code>
                <span className="tensor-shelf-type">
                  {item.tensor.dtype}
                  {item.history.length > 1 && (
                    <span
                      title={`${item.history.length} recorded states with this name${item.pending ? "" : `; showing state ${item.shown + 1}`}`}
                    >
                      {item.pending
                        ? `${item.history.length} states`
                        : `state ${item.shown + 1} of ${item.history.length}`}
                    </span>
                  )}
                  {item.history.length > 1 && <StateStrip item={item} />}
                </span>
                {origin && (
                  <span
                    className="tensor-shelf-lineage"
                    title={`Where each axis comes from\n${originDetail}`}
                  >
                    ← {origin}
                  </span>
                )}
              </span>
              {item.pending ? (
                <span className="tensor-shelf-storage">
                  <span>
                    Not computed yet{step ? ` · step ${step.index + 1}` : ""}
                  </span>
                </span>
              ) : (
                <span
                  className={`tensor-shelf-storage${shared ? " tensor-shelf-shared" : ""}`}
                  title={
                    shared
                      ? `Shares recorded storage ${item.tensor.storage_id} with ${shared}. Shared storage does not necessarily mean the views overlap.`
                      : `No other tensor in this shelf shares recorded storage ${item.tensor.storage_id}.`
                  }
                >
                  {shared ? (
                    <>
                      <Link2 size={12} aria-hidden="true" />
                      <span>Shared with {shared}</span>
                    </>
                  ) : (
                    <span>
                      {item.fresh
                        ? "Just written"
                        : item.tensor.contiguous
                          ? "Contiguous"
                          : "Strided"}
                      {range && (
                        <code
                          className="tensor-shelf-range"
                          title="Smallest and largest value"
                        >
                          {range}
                        </code>
                      )}
                    </span>
                  )}
                </span>
              )}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * A watch on one name across the run: a bar per recorded state, spanning its
 * value range on a shared scale. The state at the playback position is lit,
 * earlier ones are quiet, and later ones are not computed yet.
 */
function StateStrip({ item }: { item: Variable }) {
  const ranges = item.history.map(({ tensor }) =>
    typeof tensor.minimum === "number" &&
    typeof tensor.maximum === "number" &&
    Number.isFinite(tensor.minimum) &&
    Number.isFinite(tensor.maximum)
      ? { min: tensor.minimum, max: tensor.maximum }
      : null,
  );
  if (ranges.length > 48 || ranges.some((range) => !range)) return null;
  const known = ranges as { min: number; max: number }[];
  const low = Math.min(...known.map((range) => range.min));
  const high = Math.max(...known.map((range) => range.max));
  const span = high - low || 1;
  const step = 6,
    height = 16;
  const y = (value: number) => 1.5 + (1 - (value - low) / span) * (height - 3);
  return (
    <svg
      className="tensor-shelf-history"
      width={known.length * step + 2}
      height={height}
      role="img"
      aria-label={`${item.name} across the run: ${known
        .map(
          (range, i) =>
            `state ${i + 1} from ${formatValue(range.min)} to ${formatValue(range.max)}${i === item.shown ? " (shown)" : i > item.shown ? " (not computed yet)" : ""}`,
        )
        .join("; ")}`}
    >
      {known.map((range, i) => (
        <line
          key={i}
          className={
            i === item.shown ? "shown" : i > item.shown ? "later" : undefined
          }
          x1={3 + i * step}
          x2={3 + i * step}
          y1={y(range.max)}
          y2={Math.max(y(range.min), y(range.max) + 1.5)}
        />
      ))}
    </svg>
  );
}
