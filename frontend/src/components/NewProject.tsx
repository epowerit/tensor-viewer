import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Code2,
  GitBranch,
  LayoutGrid,
  LoaderCircle,
  Upload,
  X,
} from "lucide-react";
import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";
import { GitImport, type SourceSelection } from "../sources/GitImport";
import { importedProject } from "../sources/files";
import {
  pythonProject,
  readPythonFile,
  suggestModuleClass,
} from "../sources/pythonProject";

type Mode = "canvas" | "paste" | "upload" | "git";
const origins = [
  {
    id: "canvas",
    icon: LayoutGrid,
    title: "Build visually",
    detail: "Connect components",
  },
  {
    id: "paste",
    icon: Code2,
    title: "Paste code",
    detail: "Bring a PyTorch module",
  },
  {
    id: "upload",
    icon: Upload,
    title: "Upload code",
    detail: "Open a Python file",
  },
] as const;

export function NewProject({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (draft: Draft) => Promise<void>;
}) {
  const [name, setName] = useState("Untitled project");
  const [mode, setMode] = useState<Mode>("canvas");
  const [code, setCode] = useState("");
  const [classOverride, setClassOverride] = useState<string | null>(null);
  const [constructorText, setConstructorText] = useState("{}");
  const [filename, setFilename] = useState("");
  const [source, setSource] = useState<SourceSelection | null>(null);
  const [reading, setReading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const upload = useRef<HTMLInputElement>(null);
  const busy = creating || reading;
  const className = classOverride ?? suggestModuleClass(code);
  let constructor: Record<string, unknown> = {};
  let constructorIssue = "";
  try {
    const parsed = JSON.parse(constructorText);
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object")
      throw new Error();
    constructor = parsed;
  } catch {
    constructorIssue = 'Use a JSON object, such as {"dim": 8}.';
  }
  const fromCode = mode === "paste" || mode === "upload";
  const ready =
    !!name.trim() &&
    (mode === "canvas" ||
      (fromCode
        ? !!code.trim() &&
          /^[A-Za-z_]\w*$/.test(className.trim()) &&
          !constructorIssue
        : !!source &&
          /^[A-Za-z_]\w*$/.test(source.className) &&
          !!source.root.trim()));
  useEffect(() => {
    dialog.current?.showModal();
    nameInput.current?.select();
  }, []);
  function chooseMode(next: Mode) {
    setMode(next);
    setError("");
  }
  return (
    <dialog
      ref={dialog}
      className="new-project-dialog project-start-dialog"
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
            const draft =
              mode === "git" && source
                ? importedProject(
                    name.trim(),
                    source.snapshot,
                    source.entry,
                    source.className,
                    source.root.trim(),
                  )
                : fromCode
                  ? pythonProject(name, code, className, constructor)
                  : blankProject(name.trim());
            await onCreate(draft);
          } catch (e) {
            setError((e as Error).message);
            setCreating(false);
          }
        }}
      >
        <div className="dialog-top">
          <span className="eyebrow">ONE WORKSPACE. THREE WAYS IN.</span>
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
        <h2 id="new-project-title">Start with your model</h2>
        <p>
          Build it or bring your code. Explore every tensor in the same diagram.
        </p>
        <div className="project-origin-options" aria-label="Project source">
          {origins.map(({ id, icon: Icon, title, detail }) => (
            <button
              key={id}
              type="button"
              aria-pressed={mode === id}
              disabled={busy}
              onClick={() => chooseMode(id)}
            >
              <Icon size={21} />
              <b>{title}</b>
              <span>{detail}</span>
            </button>
          ))}
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
        {mode === "canvas" && (
          <div className="start-path-note">
            <LayoutGrid size={18} />
            <p>
              An open canvas and a toolbar of components. Add a layer, set your
              input, then generate the diagram.
            </p>
          </div>
        )}
        {fromCode && (
          <div className="source-start-fields">
            {mode === "upload" && (
              <>
                <button
                  type="button"
                  className="python-upload-target"
                  disabled={busy}
                  onClick={() => upload.current?.click()}
                >
                  {reading ? (
                    <LoaderCircle size={21} className="spin" />
                  ) : (
                    <Upload size={21} />
                  )}
                  <b>{filename || "Choose a Python file"}</b>
                  <span>
                    {filename
                      ? "Choose another .py file"
                      : ".py · up to 500 KB"}
                  </span>
                </button>
                <input
                  ref={upload}
                  type="file"
                  hidden
                  accept=".py,text/x-python"
                  aria-label="Upload Python model"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    e.target.value = "";
                    if (!file) return;
                    setReading(true);
                    setError("");
                    try {
                      const text = await readPythonFile(file);
                      setCode(text);
                      setFilename(file.name);
                      setClassOverride(null);
                    } catch (e) {
                      setError((e as Error).message);
                    } finally {
                      setReading(false);
                    }
                  }}
                />
              </>
            )}
            {(mode === "paste" || !!code) && (
              <>
                <label>
                  Python code
                  <textarea
                    className="source-start-code"
                    aria-label="Project Python code"
                    spellCheck={false}
                    disabled={busy}
                    value={code}
                    maxLength={500000}
                    placeholder={
                      "from torch import nn\n\nclass MyModel(nn.Module):\n    def forward(self, x):\n        return x.reshape(x.shape[0], -1)"
                    }
                    onChange={(e) => setCode(e.target.value)}
                  />
                </label>
                <label>
                  Class to run
                  <input
                    value={className}
                    maxLength={100}
                    placeholder="MyModel"
                    disabled={busy}
                    onChange={(e) => setClassOverride(e.target.value)}
                    required
                  />
                </label>
                <details className="disclosure-settings">
                  <summary>
                    Constructor arguments <span>optional</span>
                  </summary>
                  <label>
                    Arguments as JSON
                    <textarea
                      className="json-input"
                      aria-label="New model constructor arguments"
                      disabled={busy}
                      value={constructorText}
                      onChange={(e) => setConstructorText(e.target.value)}
                      rows={3}
                    />
                  </label>
                </details>
                {constructorIssue && (
                  <p className="field-error" role="alert">
                    {constructorIssue}
                  </p>
                )}
              </>
            )}
            <p className="field-hint">
              Next, choose the input shape. Your code runs locally when you
              generate the diagram.
            </p>
          </div>
        )}
        <details
          className="more-project-sources"
          open={mode === "git" ? true : undefined}
        >
          <summary>More sources</summary>
          <button
            type="button"
            className="secondary-button"
            aria-pressed={mode === "git"}
            disabled={busy}
            onClick={() => chooseMode("git")}
          >
            <GitBranch size={14} /> Git repository
          </button>
          {mode === "git" && (
            <GitImport
              disabled={creating}
              onSelect={setSource}
              onBusy={setReading}
            />
          )}
        </details>
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
          )}
          {creating
            ? "Creating…"
            : mode === "canvas"
              ? "Open canvas"
              : "Set up inputs"}
        </button>
      </form>
    </dialog>
  );
}
