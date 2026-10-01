import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  GitBranch,
  LayoutGrid,
  LoaderCircle,
  Code2,
  X,
} from "lucide-react";
import { consoleProject } from "../console/script";
import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";
import { GitImport, type SourceSelection } from "../sources/GitImport";
import { importedProject } from "../sources/files";
import { CodeImport, type CodeSource } from "../sources/CodeImportPanel";
import { codeImportIssue, projectFromCode } from "../sources/codeImport";
import "./projectDialogs.css";

export function NewProject({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (draft: Draft) => Promise<void>;
}) {
  const [name, setName] = useState("Untitled experiment");
  const [mode, setMode] = useState<"console" | "canvas" | "git">("console");
  const [source, setSource] = useState<SourceSelection | null>(null);
  const [codeSource, setCodeSource] = useState<CodeSource>({
    code: consoleProject("Starter").script!,
    entry: "statements",
    className: "",
  });
  const [reading, setReading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const busy = creating || reading;
  const codeIssue =
    mode === "console"
      ? codeImportIssue(codeSource.code, codeSource.entry, codeSource.className)
      : null;
  const ready =
    !!name.trim() &&
    !codeIssue &&
    (mode !== "git" ||
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
      className={`new-project-dialog blank-project-dialog project-creation ${mode === "git" ? "git-project-dialog" : ""}`}
      aria-labelledby="new-project-title"
      aria-describedby="new-project-description"
      aria-busy={busy}
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
                : mode === "console"
                  ? projectFromCode(
                      name.trim(),
                      codeSource.code,
                      codeSource.entry,
                      codeSource.className,
                    )
                  : blankProject(name.trim()),
            );
          } catch (e) {
            setError((e as Error).message);
            setCreating(false);
          }
        }}
      >
        <header className="project-creation-heading">
          <div>
            <h2 id="new-project-title">New project</h2>
            <p id="new-project-description">
              Start with code or build a diagram. Explore every tensor in one
              workspace.
            </p>
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Close new project"
            onClick={onClose}
            disabled={busy}
          >
            <X size={18} />
          </button>
        </header>
        <div className="project-creation-body">
          <div
            className="project-origin-options"
            role="group"
            aria-label="Project source"
          >
            <button
              type="button"
              aria-pressed={mode === "console"}
              disabled={busy}
              onClick={() => {
                setMode("console");
                setSource(null);
                setError("");
              }}
            >
              <Code2 size={16} />
              <span>
                Code<small aria-hidden="true">Paste or upload</small>
              </span>
            </button>
            <button
              type="button"
              aria-pressed={mode === "canvas"}
              disabled={busy}
              onClick={() => {
                setMode("canvas");
                setSource(null);
                setError("");
              }}
            >
              <LayoutGrid size={16} />
              <span>
                Build a diagram<small aria-hidden="true">Add components</small>
              </span>
            </button>
            <button
              type="button"
              aria-pressed={mode === "git"}
              disabled={busy}
              onClick={() => {
                if (mode !== "git") {
                  setMode("git");
                  setSource(null);
                  setError("");
                }
              }}
            >
              <GitBranch size={16} />
              <span>
                Git repository<small aria-hidden="true">Import a model</small>
              </span>
            </button>
          </div>
          <label className="project-name-field">
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
          {mode === "console" && (
            <CodeImport
              value={codeSource}
              onChange={setCodeSource}
              disabled={busy}
              onReading={setReading}
            />
          )}
          {codeIssue && (
            <p className="field-error" role="status">
              {codeIssue}
            </p>
          )}
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
        </div>
        <footer className="project-creation-footer">
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
              : mode === "console"
                ? codeSource.entry === "module"
                  ? "Configure model"
                  : "Create experiment"
                : mode === "canvas"
                  ? "Open canvas"
                  : "Import and configure"}
          </button>
        </footer>
      </form>
    </dialog>
  );
}
