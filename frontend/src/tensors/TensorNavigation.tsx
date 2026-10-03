import { useEffect, useId, useState } from "react";
import {
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  LocateFixed,
} from "lucide-react";
import { parseCoordinate } from "./plane";

/** Constant-size control, even when an axis contains millions of indices. */
export function IndexControl({
  label,
  name,
  value,
  size,
  step = 1,
  onChange,
}: {
  label: string;
  name: string;
  value: number;
  size: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  const [draft, setDraft] = useState(String(value));
  const [error, setError] = useState(false);
  useEffect(() => {
    setDraft(String(value));
    setError(false);
  }, [value]);
  function commit() {
    const next = Number(draft);
    if (!/^\d+$/.test(draft) || !Number.isSafeInteger(next) || next >= size) {
      setError(true);
      return;
    }
    setDraft(String(next));
    setError(false);
    onChange(next);
  }
  function advance(direction: number) {
    const next = Math.max(
      0,
      Math.min(size - 1, Math.floor(value / step) * step + direction * step),
    );
    setDraft(String(next));
    setError(false);
    onChange(next);
  }
  return (
    <div className="index-control">
      <label htmlFor={id} title={label}>
        {label}
      </label>
      <div className="index-stepper">
        <button
          type="button"
          aria-label={`${name} previous${step > 1 ? " page" : ""}`}
          title={step > 1 ? `Previous ${step} indices` : "Previous index"}
          disabled={value < step}
          onClick={() => advance(-1)}
        >
          {step > 1 ? <ChevronsLeft size={14} /> : <ChevronLeft size={14} />}
        </button>
        <input
          id={id}
          aria-label={name}
          type="text"
          inputMode="numeric"
          value={draft}
          aria-invalid={error}
          aria-describedby={`${id}-range`}
          onChange={(event) => {
            setDraft(event.target.value);
            setError(false);
          }}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit();
            }
            if (event.key === "Escape") {
              event.stopPropagation();
              setDraft(String(value));
              setError(false);
            }
          }}
        />
        <button
          type="button"
          aria-label={`${name} next${step > 1 ? " page" : ""}`}
          title={step > 1 ? `Next ${step} indices` : "Next index"}
          disabled={Math.floor(value / step) * step + step >= size}
          onClick={() => advance(1)}
        >
          {step > 1 ? <ChevronsRight size={14} /> : <ChevronRight size={14} />}
        </button>
      </div>
      <small
        id={`${id}-range`}
        className={error ? "index-error" : ""}
        role={error ? "alert" : undefined}
      >
        {error ? "Use " : ""}0–{(size - 1).toLocaleString()}
      </small>
    </div>
  );
}

export function CoordinateJump({
  label,
  summaryLabel = "Go to coordinate",
  shape,
  coords,
  onSelect,
}: {
  label: string;
  summaryLabel?: string;
  shape: number[];
  coords: number[];
  onSelect: (index: number) => void;
}) {
  const id = useId();
  const [draft, setDraft] = useState(coords.join(", "));
  const [error, setError] = useState("");
  const position = coords.join(", ");
  useEffect(() => {
    setDraft(position);
    setError("");
  }, [position]);
  return (
    <details className="coordinate-jump">
      <summary>
        <LocateFixed size={13} /> {summaryLabel}
      </summary>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const result = parseCoordinate(draft, shape);
          setError(result.error);
          if (result.index !== null) onSelect(result.index);
        }}
      >
        <label htmlFor={id}>Indices in the original axis order</label>
        <div className="coordinate-entry">
          <input
            id={id}
            aria-label={`${label} coordinate`}
            value={draft}
            placeholder={shape.map(() => "0").join(", ")}
            aria-invalid={!!error}
            aria-describedby={`${id}-hint`}
            onChange={(event) => {
              setDraft(event.target.value);
              setError("");
            }}
          />
          <button
            className="secondary-button small"
            type="submit"
            aria-label={`${label} go to coordinate`}
          >
            Go <ArrowRight size={13} />
          </button>
        </div>
        <p
          id={`${id}-hint`}
          className={error ? "index-error" : ""}
          role={error ? "alert" : undefined}
        >
          {error ||
            "Use 0-based indices; -1 is the last. The view follows the selected element."}
        </p>
      </form>
    </details>
  );
}
