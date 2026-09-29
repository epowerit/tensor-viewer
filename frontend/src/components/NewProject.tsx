import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  GitBranch,
  LayoutGrid,
  LoaderCircle,
  X,
} from "lucide-react";
import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";
import { GitImport, type SourceSelection } from "../sources/GitImport";
import { importedProject } from "../sources/files";

export function NewProject({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (draft: Draft) => Promise<void>;
}) {
  const [name, setName] = useState("Untitled experiment");
  const [mode, setMode] = useState<"canvas" | "git">("canvas");
  const [source, setSource] = useState<SourceSelection | null>(null);
  const [reading, setReading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const busy = creating || reading;
  const ready =
    !!name.trim() &&
    (mode === "canvas" ||
      (!!source &&
        /^[A-Za-z_]\w*$/.test(source.className) &&
        !!source.root.trim()));
  useEffect(() => {
    dialog.current?.showModal();
    nameInput.current?.select();
  }, []);
  return (
    <dialog
      ref={dialog}
      className={`new-project-dialog blank-project-dialog ${mode === "git" ? "git-project-dialog" : ""}`}
      aria-labelledby="new-project-title"
      onCancel={(e) => {
        if (busy) e.preventDefault();
        else onClose();
      }}
    >
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (!ready || busy) return;
          setCreating(true);
          setError("");
          try {
            await onCreate(
              mode === "git" && source
                ? importedProject(
                    name.trim(),
                    source.snapshot,
                    source.entry,
                    source.className,
                    source.root.trim(),
                  )
                : blankProject(name.trim()),
            );
          } catch (e) {
            setError((e as Error).message);
            setCreating(false);
          }
        }}
      >
        <div className="dialog-top">
          <span className="eyebrow">Your next experiment</span>
          <button
            type="button"
            className="icon-button"
            aria-label="Close new project"
            onClick={onClose}
            disabled={busy}
          >
            <X size={18} />
          </button>
        </div>
        <h2 id="new-project-title">New project</h2>
        <p>
          {mode === "canvas"
            ? "Build a model from reusable components."
            : "Bring a committed Python model into your workspace."}
        </p>
        <div className="project-origin-options" aria-label="Project source">
          <button
            type="button"
            aria-pressed={mode === "canvas"}
            disabled={busy}
            onClick={() => {
              setMode("canvas");
              setSource(null);
            }}
          >
            <LayoutGrid size={16} />
            Blank canvas
          </button>
          <button
            type="button"
            aria-pressed={mode === "git"}
            disabled={busy}
            onClick={() => {
              setMode("git");
              setSource(null);
            }}
          >
            <GitBranch size={16} />
            Git repository
          </button>
        </div>
        <label>
          Project name
          <input
            ref={nameInput}
            autoFocus
            required
            maxLength={100}
            disabled={busy}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        {mode === "git" && (
          <GitImport
            disabled={creating}
            onSelect={setSource}
            onBusy={setReading}
          />
        )}
        {error && (
          <p role="alert" className="field-error">
            {error}
          </p>
        )}
        <button
          type="submit"
          className="primary-button create-project-button"
          disabled={!ready || busy}
        >
          {creating ? (
            <LoaderCircle size={15} className="spin" />
          ) : (
            <ArrowRight size={15} />
          )}{" "}
          {creating
            ? "Creating…"
            : mode === "canvas"
              ? "Open canvas"
              : "Import and configure"}
        </button>
      </form>
    </dialog>
  );
}
