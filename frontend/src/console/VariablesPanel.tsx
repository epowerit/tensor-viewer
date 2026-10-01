import { Link2 } from "lucide-react";
import type { Run } from "../api/client";
import { ShapeGlyph } from "../editor/ShapeGlyph";
import { variables } from "./variables";
import "./tensorShelf.css";

type Props = {
  run: Run | null;
  selected: string | null;
  onSelect: (nodeId: string) => void;
};

/** Every named tensor of the displayed run: shape, type, and shared storage. */
export function VariablesPanel({ run, selected, onSelect }: Props) {
  const items = run ? variables(run.trace) : [];
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
        return (
          <li key={item.name}>
            <button
              type="button"
              className={`tensor-shelf-card${item.anonymous ? " tensor-shelf-anonymous" : ""}`}
              aria-pressed={item.nodeId === selected}
              aria-label={`Inspect ${item.name}, shape ${shape}, ${item.tensor.dtype}${shared ? `, shares storage with ${shared}` : ""}`}
              title={`${item.name} · ${shape}${item.line ? ` · line ${item.line}` : ""}`}
              onClick={() => onSelect(item.nodeId)}
            >
              <span className="tensor-shelf-glyph" aria-hidden="true">
                <ShapeGlyph shape={item.tensor.shape} />
              </span>
              <span className="tensor-shelf-identity">
                <span className="tensor-shelf-name">{item.name}</span>
                <code className="tensor-shelf-shape">{shape}</code>
                <span className="tensor-shelf-type">
                  {item.tensor.dtype}
                  {item.states > 1 && (
                    <span
                      title={`${item.states} recorded states with this name`}
                    >
                      {item.states} states
                    </span>
                  )}
                </span>
              </span>
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
                    {item.tensor.contiguous ? "Contiguous" : "Strided"}
                  </span>
                )}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
