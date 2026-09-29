import { useState } from "react";
import { Check, GitBranch, LoaderCircle } from "lucide-react";
import { api, type SourceImport } from "../api/client";

export type SourceSelection = {
  snapshot: SourceImport;
  entry: string;
  className: string;
  root: string;
};
export function GitImport({
  disabled,
  onSelect,
  onBusy,
}: {
  disabled: boolean;
  onSelect: (source: SourceSelection | null) => void;
  onBusy: (busy: boolean) => void;
}) {
  const [repository, setRepository] = useState("");
  const [revision, setRevision] = useState("HEAD");
  const [subdirectory, setSubdirectory] = useState(".");
  const [selection, setSelection] = useState<SourceSelection | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  function choose(next: SourceSelection | null) {
    setSelection(next);
    onSelect(next);
  }
  async function read() {
    setLoading(true);
    onBusy(true);
    setError("");
    choose(null);
    try {
      const snapshot = await api.importGit(
        repository.trim(),
        revision.trim() || "HEAD",
        subdirectory.trim() || ".",
      );
      const paths = Object.keys(snapshot.files).filter((p) =>
        p.endsWith(".py"),
      );
      const entry =
        paths.find((p) => /(^|\/)model\.py$/.test(p)) ??
        paths.find((p) => !p.endsWith("__init__.py")) ??
        paths[0];
      const className =
        snapshot.files[entry].match(/class\s+(\w+)\s*\(/)?.[1] ?? "Model";
      choose({
        snapshot,
        entry,
        className,
        root: entry.startsWith("src/") ? "src" : ".",
      });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
      onBusy(false);
    }
  }
  const locked = disabled || loading;
  return (
    <section className="git-import" aria-label="Git source import">
      <label>
        Repository
        <input
          value={repository}
          placeholder="https://github.com/owner/project.git or /absolute/path"
          disabled={locked}
          onChange={(e) => {
            setRepository(e.target.value);
            choose(null);
          }}
        />
      </label>
      <div className="source-field-pair">
        <label>
          Revision
          <input
            value={revision}
            placeholder="HEAD, branch, tag, or commit"
            disabled={locked}
            onChange={(e) => {
              setRevision(e.target.value);
              choose(null);
            }}
          />
        </label>
        <label>
          Subdirectory
          <input
            value={subdirectory}
            placeholder="."
            disabled={locked}
            onChange={(e) => {
              setSubdirectory(e.target.value);
              choose(null);
            }}
          />
        </label>
      </div>
      <button
        type="button"
        className="secondary-button"
        disabled={locked || !repository.trim()}
        onClick={() => void read()}
      >
        {loading ? (
          <LoaderCircle size={15} className="spin" />
        ) : (
          <GitBranch size={15} />
        )}{" "}
        {loading ? "Reading committed files…" : "Read source files"}
      </button>
      <small className="source-help">
        Reads up to 128 source and configuration files (2 MB). Setup scripts and
        model code run only when you explicitly request execution.
      </small>
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      {selection && (
        <div className="import-selection">
          <p className="import-revision">
            <Check size={14} />
            {Object.keys(selection.snapshot.files).length} files · commit{" "}
            <code title={selection.snapshot.repository.revision}>
              {selection.snapshot.repository.revision.slice(0, 12)}
            </code>
          </p>
          {!!selection.snapshot.skipped && (
            <small>
              {selection.snapshot.skipped} unsupported files or links omitted.
            </small>
          )}
          <label>
            Entry file
            <select
              value={selection.entry}
              disabled={locked}
              onChange={(e) => choose({ ...selection, entry: e.target.value })}
            >
              {Object.keys(selection.snapshot.files)
                .filter((p) => p.endsWith(".py"))
                .map((p) => (
                  <option key={p}>{p}</option>
                ))}
            </select>
          </label>
          <div className="source-field-pair">
            <label>
              Module class
              <input
                value={selection.className}
                disabled={locked}
                onChange={(e) =>
                  choose({ ...selection, className: e.target.value })
                }
              />
            </label>
            <label>
              Import root
              <input
                value={selection.root}
                disabled={locked}
                onChange={(e) => choose({ ...selection, root: e.target.value })}
              />
            </label>
          </div>
          <small className="source-help">
            Choose the nn.Module to instantiate. Configure its arguments,
            inputs, and dependencies after import.
          </small>
        </div>
      )}
    </section>
  );
}
