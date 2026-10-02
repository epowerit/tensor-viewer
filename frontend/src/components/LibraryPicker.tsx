import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { api, type LibraryEntry } from "../api/client";

/**
 * The project library, from tensor shapes to multimodal transformers. Each
 * entry is an ordinary PyTorch file; choosing one creates its project the way
 * pasted code does.
 */
export function LibraryPicker({
  value,
  onChange,
  disabled,
  owned,
}: {
  value: LibraryEntry | null;
  onChange: (entry: LibraryEntry) => void;
  disabled: boolean;
  /** Whether the workspace already holds this entry's project. */
  owned?: (entry: LibraryEntry) => boolean;
}) {
  const [entries, setEntries] = useState<LibraryEntry[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let current = true;
    api
      .library()
      .then((found) => {
        if (!current) return;
        setEntries(found);
        if (!value && found[0]) onChange(found[0]);
      })
      .catch((e: Error) => current && setError(e.message));
    return () => {
      current = false;
    };
  }, []);
  if (error)
    return (
      <p className="field-error" role="alert">
        {error}
      </p>
    );
  if (!entries)
    return (
      <p className="library-loading" role="status">
        <LoaderCircle size={14} className="spin" /> Loading the library…
      </p>
    );
  const tracks = [...new Set(entries.map((entry) => entry.track))];
  return (
    <div
      className="project-library"
      role="radiogroup"
      aria-label="Library projects"
    >
      {tracks.map((track) => (
        <section key={track} aria-label={track}>
          <h3>{track}</h3>
          {entries
            .filter((entry) => entry.track === track)
            .map((entry) => (
              <button
                key={entry.id}
                type="button"
                role="radio"
                aria-checked={value?.id === entry.id}
                disabled={disabled}
                title={entry.summary}
                onClick={() => onChange(entry)}
              >
                <b>{String(entry.number).padStart(2, "0")}</b>
                <span>
                  {entry.title}
                  {owned?.(entry) && (
                    <em className="library-owned">In your workspace</em>
                  )}
                  <small>{entry.summary}</small>
                </span>
              </button>
            ))}
        </section>
      ))}
    </div>
  );
}

export const libraryProjectName = (entry: LibraryEntry) =>
  `${String(entry.number).padStart(2, "0")} · ${entry.title}`;
