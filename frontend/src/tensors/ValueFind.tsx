import { useEffect, useId, useMemo, useState } from "react";
import { ChevronDown, ChevronUp, Search } from "lucide-react";
import type { SearchResult, Tensor } from "../api/client";
import { findValues, nextBroken, parseQuery, previousOf } from "./find";
import { useTensorQuery } from "./useTensorQuery";

/**
 * Find values in a tensor: every recorded value matching a comparison, with
 * a count, and Enter or the arrows stepping through them. Matches are ringed
 * in the grid through `onMatches`. A tensor whose values stay in a snapshot
 * is searched by the backend.
 */
export function ValueFind({
  label,
  tensor,
  runId,
  index,
  onSelect,
  onMatches,
}: {
  label: string;
  tensor: Tensor;
  /** The run holding the tensor's snapshot, for searching large tensors. */
  runId?: string;
  /** The selected cell, where stepping starts. */
  index: number;
  onSelect: (index: number) => void;
  onMatches: (matches: Set<number> | null) => void;
}) {
  const id = useId();
  const [text, setText] = useState("");
  const query = useMemo(() => parseQuery(text), [text]);
  const inline = tensor.values.length === tensor.numel;
  const remote = !inline && tensor.value_source === "paged" && !!runId;
  const local = useMemo(
    () => ("test" in query && inline ? findValues(tensor, query) : null),
    [query, tensor, inline],
  );
  const asked = useTensorQuery<SearchResult>(
    runId,
    tensor.id,
    "search",
    remote && "spec" in query ? query.spec : null,
    250,
  );
  const found = remote ? (asked.data?.indices ?? null) : local;
  // The backend counts every match but sends the first few thousand.
  const total = remote ? (asked.data?.count ?? 0) : (found?.length ?? 0);
  useEffect(() => {
    onMatches(found?.length ? new Set(found) : null);
  }, [found, onMatches]);
  useEffect(() => () => onMatches(null), [onMatches]);
  const position = found?.indexOf(index) ?? -1;
  const step = (forward: boolean) => {
    if (!found?.length) return;
    const next = forward ? nextBroken(found, index) : previousOf(found, index);
    if (next !== null) onSelect(next);
  };
  const error = "error" in query ? query.error : "";
  return (
    <details className="coordinate-jump value-find">
      <summary>
        <Search size={13} /> Find values
      </summary>
      {!inline && !remote ? (
        <p className="value-find-note">
          This tensor's values load by window, so it cannot be searched as a
          whole. Its histogram above still counts them.
        </p>
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            step(true);
          }}
        >
          <label htmlFor={id}>
            A comparison: &gt; 0.5, == 0, abs &gt; 2, nan, inf
          </label>
          <div className="coordinate-entry">
            <input
              id={id}
              aria-label={`Find values in ${label}`}
              value={text}
              placeholder="> 0.5"
              spellCheck={false}
              aria-invalid={!!error}
              aria-describedby={`${id}-result`}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && event.shiftKey) {
                  event.preventDefault();
                  step(false);
                } else if (event.key === "Escape" && text) {
                  event.preventDefault();
                  event.stopPropagation();
                  setText("");
                }
              }}
            />
            <button
              className="secondary-button small"
              type="button"
              aria-label="Previous match"
              title="Previous match (Shift+Enter)"
              disabled={!found?.length}
              onClick={() => step(false)}
            >
              <ChevronUp size={13} />
            </button>
            <button
              className="secondary-button small"
              type="submit"
              aria-label="Next match"
              title="Next match (Enter)"
              disabled={!found?.length}
            >
              <ChevronDown size={13} />
            </button>
          </div>
          <p
            id={`${id}-result`}
            className={error ? "index-error" : ""}
            role="status"
          >
            {error
              ? error
              : asked.error
                ? asked.error
                : remote && asked.loading
                  ? "Searching every value…"
                  : found
                    ? total
                      ? `${position >= 0 ? `${position + 1} of ` : ""}${total.toLocaleString()} ${total === 1 ? "value" : "values"} where ${"label" in query ? query.label : ""} · ${share(total, tensor.numel)}${asked.data?.truncated ? ` · stepping through the first ${found.length.toLocaleString()}` : ""}`
                      : `No value where ${"label" in query ? query.label : ""}.`
                    : "Matches are ringed in the grid; Enter steps through them."}
          </p>
        </form>
      )}
    </details>
  );
}

function share(count: number, total: number) {
  const percent = (count / total) * 100;
  return percent > 0 && percent < 1 ? "<1%" : `${Math.round(percent)}%`;
}
