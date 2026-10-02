import { useEffect, useState } from "react";
import { LoaderCircle, Search } from "lucide-react";
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
  const [query, setQuery] = useState("");
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
  // Every word must appear in the title, summary, track, or number.
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = entries.filter((entry) => {
    const text =
      `${entry.number} ${entry.title} ${entry.summary} ${entry.track}`.toLowerCase();
    return words.every((word) => text.includes(word));
  });
  const tracks = [...new Set(shown.map((entry) => entry.track))];
  return (
    <>
      <label className="library-search">
        <Search size={13} aria-hidden="true" />
        <input
          type="search"
          value={query}
          placeholder="Find a model: attention, vision, rotary…"
          aria-label="Find a library model"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            // Enter picks the first match instead of submitting the dialog.
            if (event.key === "Enter" && shown[0]) {
              event.preventDefault();
              onChange(shown[0]);
            }
          }}
        />
      </label>
      <div
        className="project-library"
        role="radiogroup"
        aria-label="Library projects"
      >
        {!shown.length && (
          <p className="library-empty">
            Nothing in the library matches “{query}”.
          </p>
        )}
        {tracks.map((track) => (
          <section key={track} aria-label={track}>
            <h3>{track}</h3>
            {shown
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
    </>
  );
}

export const libraryProjectName = (entry: LibraryEntry) =>
  `${String(entry.number).padStart(2, "0")} · ${entry.title}`;
