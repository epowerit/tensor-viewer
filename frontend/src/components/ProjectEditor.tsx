import { useEffect, useId, useRef, useState } from "react";
import { Braces, FileCode2, Info, Upload, Plus, Trash2 } from "lucide-react";
import type { Draft } from "../api/client";
import { ForwardInputs } from "../inputs/ForwardInputs";
import { EnvironmentSetup } from "../sources/EnvironmentSetup";
import {
  chooseEntry,
  entryPath,
  pathIssue,
  projectFiles,
  sourceCode,
  updateFile,
} from "../sources/files";
import { WeightLibrary } from "../weights/WeightLibrary";
import { validClassName } from "../workflow/projectAction";
import { revealField } from "./revealField";

type Props = {
  draft: Draft;
  onChange: (draft: Draft) => void;
  onValidity: (valid: boolean) => void;
  busy: boolean;
  active: boolean;
  /** Canvas projects generate their module: only name, inputs, and weights apply. */
  readOnly?: boolean;
  /** Reveal the first unfinished setting; a new number repeats the request. */
  reviewRequest?: number;
  /** Reveal the first input that needs attention. */
  inputReviewRequest?: number;
  /** False while a canvas model still has connections to resolve. */
  builderReady?: boolean;
};

export function ProjectEditor({
  draft,
  onChange,
  onValidity,
  busy: externalBusy,
  active,
  readOnly = false,
  reviewRequest = 0,
  inputReviewRequest = 0,
  builderReady = true,
}: Props) {
  const fieldId = useId();
  const [settingUp, setSettingUp] = useState(false);
  const busy = externalBusy || settingUp;
  const [selectedFile, setSelectedFile] = useState(entryPath(draft));
  const [newPath, setNewPath] = useState("");
  const [adding, setAdding] = useState(false);
  const files = projectFiles(draft);
  const currentFile =
    files[selectedFile] === undefined ? entryPath(draft) : selectedFile;
  const currentCode = sourceCode(draft, currentFile);
  const newFileIssue = newPath
    ? (pathIssue(newPath) ??
      (files[newPath] !== undefined ? "This file already exists." : null))
    : null;
  const [constructor, setConstructor] = useState(
    JSON.stringify(draft.constructor, null, 2),
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const upload = useRef<HTMLInputElement>(null);
  const codeRef = useRef<HTMLTextAreaElement>(null);
  const editorRoot = useRef<HTMLDivElement>(null);
  const handledReview = useRef(0);
  const [inputsValid, setInputsValid] = useState(true);
  const scripted = draft.script != null;
  // A console derives its module from the statements; a canvas from its components.
  const moduleSettings = !scripted && !readOnly;
  const editorValid =
    !Object.values(errors).some(Boolean) &&
    !!draft.name.trim() &&
    (!moduleSettings ||
      (validClassName(draft.class_name) && !!draft.code.trim()));
  useEffect(
    () => onValidity(inputsValid && editorValid && !settingUp),
    [inputsValid, editorValid, settingUp, onValidity],
  );
  useEffect(() => {
    if (
      !active ||
      busy ||
      !reviewRequest ||
      reviewRequest === handledReview.current
    )
      return;
    handledReview.current = reviewRequest;
    const root = editorRoot.current;
    const invalid = !draft.name.trim()
      ? root?.querySelector<HTMLElement>('[data-setting="name"]')
      : root?.querySelector<HTMLElement>(
          '.configuration [aria-invalid="true"]',
        );
    revealField(
      invalid ??
        root?.querySelector<HTMLElement>(
          ".configuration input:not(:disabled)",
        ) ??
        null,
    );
  }, [reviewRequest, active, busy]);
  function validation(field: string, message: string) {
    setErrors((current) => ({ ...current, [field]: message }));
  }
  function changeConstructor(value: string) {
    setConstructor(value);
    try {
      const parsed = JSON.parse(value);
      if (!parsed || Array.isArray(parsed) || typeof parsed !== "object")
        throw new Error();
      validation("kwargs", "");
      onChange({ ...draft, constructor: parsed });
    } catch {
      validation(
        "kwargs",
        'Enter a JSON object, for example {"embed_dim": 8, "num_heads": 2}.',
      );
    }
  }
  return (
    <div className="editor-layout" ref={editorRoot}>
      <section className="code-editor" hidden>
        <div className="editor-toolbar">
          <span>
            <FileCode2 size={16} />
            <select
              aria-label="Source file"
              value={currentFile}
              disabled={busy}
              onChange={(e) => setSelectedFile(e.target.value)}
            >
              {Object.keys(files)
                .sort()
                .map((path) => (
                  <option key={path} value={path}>
                    {path}
                    {path === entryPath(draft) ? " · entry" : ""}
                  </option>
                ))}
            </select>
          </span>
          <button
            className="secondary-button small"
            disabled={busy}
            onClick={() => upload.current?.click()}
          >
            <Upload size={13} />
            Replace .py
          </button>
          <button
            className="icon-button"
            aria-label="Add source file"
            aria-expanded={adding}
            title="Add source file"
            disabled={busy || Object.keys(files).length >= 128}
            onClick={() => setAdding(!adding)}
          >
            <Plus size={15} />
          </button>
          {currentFile !== entryPath(draft) && (
            <button
              className="icon-button"
              aria-label="Remove selected file"
              title="Remove selected file"
              disabled={busy}
              onClick={() => {
                const next = { ...draft.files };
                delete next[currentFile];
                onChange({ ...draft, files: next });
                setSelectedFile(entryPath(draft));
              }}
            >
              <Trash2 size={14} />
            </button>
          )}
          <input
            hidden
            ref={upload}
            type="file"
            accept=".py,text/x-python"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              if (file.size > 500000) {
                validation("upload", "Use a Python file smaller than 500 KB.");
                return;
              }
              const code = await file.text();
              validation("upload", "");
              const found = code.match(
                /class\s+(\w+)\s*\([^)]*(?:nn\.Module|Module)/,
              );
              onChange({
                ...updateFile(draft, currentFile, code),
                class_name:
                  currentFile === entryPath(draft)
                    ? (found?.[1] ?? draft.class_name)
                    : draft.class_name,
              });
              e.target.value = "";
            }}
          />
        </div>
        {adding && (
          <div className="add-source-file">
            <label>
              New file path
              <input
                value={newPath}
                placeholder="layers/attention.py"
                disabled={busy}
                onChange={(e) => setNewPath(e.target.value)}
              />
            </label>
            <button
              className="secondary-button small"
              disabled={busy || !newPath || !!newFileIssue}
              onClick={() => {
                onChange(updateFile(draft, newPath, ""));
                setSelectedFile(newPath);
                setNewPath("");
                setAdding(false);
              }}
            >
              Add file
            </button>
            {newFileIssue && <p className="field-error">{newFileIssue}</p>}
          </div>
        )}
        <div className="editor-body">
          <div className="editor-line-numbers" aria-hidden="true">
            {currentCode.split("\n").map((_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </div>
          <textarea
            ref={codeRef}
            aria-label="Python module code"
            className="code-textarea"
            spellCheck={false}
            value={currentCode}
            disabled={busy}
            onChange={(e) =>
              onChange(updateFile(draft, currentFile, e.target.value))
            }
            onKeyDown={(e) => {
              if (e.key === "Tab") {
                e.preventDefault();
                const start = e.currentTarget.selectionStart;
                const end = e.currentTarget.selectionEnd;
                onChange(
                  updateFile(
                    draft,
                    currentFile,
                    currentCode.slice(0, start) +
                      "    " +
                      currentCode.slice(end),
                  ),
                );
                requestAnimationFrame(() => {
                  codeRef.current?.setSelectionRange(start + 4, start + 4);
                });
              }
            }}
          />
        </div>
        <div className="editor-note">
          {Object.keys(files).length} source{" "}
          {Object.keys(files).length === 1 ? "file" : "files"} · edits saved
          with each run
          <span>Python / PyTorch</span>
        </div>
        {errors.upload && <p className="field-error">{errors.upload}</p>}
      </section>
      <div className="configuration">
        <section className="config-card">
          <div className="section-label">
            <Braces size={16} /> Module configuration
          </div>
          <label>
            Project name
            <input
              value={draft.name}
              data-setting="name"
              aria-invalid={!draft.name.trim()}
              disabled={busy}
              maxLength={100}
              onChange={(e) => onChange({ ...draft, name: e.target.value })}
            />
          </label>
          {moduleSettings && (
            <>
              <label>
                Entry file
                <select
                  value={entryPath(draft)}
                  disabled={busy}
                  onChange={(e) => {
                    onChange(chooseEntry(draft, e.target.value));
                    setSelectedFile(e.target.value);
                  }}
                >
                  {Object.keys(files)
                    .filter((path) => path.endsWith(".py"))
                    .sort()
                    .map((path) => (
                      <option key={path}>{path}</option>
                    ))}
                </select>
              </label>
              {(Object.keys(files).length > 1 ||
                entryPath(draft) !== "model.py") && (
                <label>
                  Python import root
                  <input
                    value={draft.import_root ?? "."}
                    disabled={busy}
                    onChange={(e) =>
                      onChange({ ...draft, import_root: e.target.value })
                    }
                  />
                  <small>
                    Relative directory containing your packages, usually . or
                    src.
                  </small>
                </label>
              )}
              {draft.repository && (
                <div className="source-provenance">
                  <span>
                    Imported from commit{" "}
                    <code>{draft.repository.revision.slice(0, 12)}</code>
                  </span>
                  <small title={draft.repository.url}>
                    {draft.repository.url}
                  </small>
                  <small>
                    Project edits are stored independently of the repository.
                  </small>
                </div>
              )}
              <label>
                Class name
                <input
                  value={draft.class_name}
                  aria-invalid={!validClassName(draft.class_name)}
                  disabled={busy}
                  onChange={(e) =>
                    onChange({ ...draft, class_name: e.target.value })
                  }
                />
              </label>
              <label>
                Constructor arguments
                <textarea
                  aria-label="Constructor arguments"
                  className="json-input"
                  aria-invalid={!!errors.kwargs}
                  aria-describedby={
                    errors.kwargs ? `${fieldId}-kwargs` : undefined
                  }
                  value={constructor}
                  disabled={busy}
                  onChange={(e) => changeConstructor(e.target.value)}
                  rows={5}
                />
              </label>
              {errors.kwargs && (
                <p
                  id={`${fieldId}-kwargs`}
                  className="field-error"
                  role="alert"
                >
                  {errors.kwargs}
                </p>
              )}
            </>
          )}
        </section>
        {!draft.blueprint && (
          <details className="disclosure-settings">
            <summary>
              Python environment <span>optional</span>
            </summary>
            <EnvironmentSetup
              draft={draft}
              onChange={onChange}
              busy={busy}
              onBusy={setSettingUp}
            />
          </details>
        )}
        <details className="disclosure-settings weights-disclosure">
          <summary>
            Model weights <span>optional</span>
          </summary>
          <WeightLibrary
            draft={draft}
            onChange={onChange}
            busy={busy}
            active={active}
            invalid={!editorValid || !inputsValid || !builderReady}
          />
        </details>
        <ForwardInputs
          draft={draft}
          onChange={onChange}
          onValidity={setInputsValid}
          reviewRequest={inputReviewRequest}
          busy={busy}
          active={active}
        />
        <div className="local-note">
          <Info size={17} />
          <p>
            Code runs locally on your computer. Use code you trust. Value runs
            use CPU tensors; shape runs use PyTorch metadata. Both use
            evaluation mode and a 20-second execution limit.
          </p>
        </div>
        <div className="annotation-note">
          <b>Give dimensions meaning</b>
          <p>
            Add <code># axes: batch, tokens, features</code> to an assignment to
            label its output. Unlabeled dimensions remain neutral.
          </p>
        </div>
      </div>
    </div>
  );
}
