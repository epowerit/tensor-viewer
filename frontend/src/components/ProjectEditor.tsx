import { useEffect, useId, useRef, useState } from "react";
import { Braces, FileCode2, Info, Upload, Plus, Trash2 } from "lucide-react";
import type { Draft } from "../api/client";
import { EnvironmentSetup } from "../sources/EnvironmentSetup";
import {
  chooseEntry,
  entryPath,
  pathIssue,
  projectFiles,
  sourceCode,
  updateFile,
} from "../sources/files";
import { readPythonFile, suggestModuleClass } from "../sources/pythonProject";
import { revealField } from "./revealField";
import { validClassName } from "../workflow/projectAction";
import {
  lineSelection,
  type EditorNavigation,
} from "../sources/editorNavigation";

type Props = {
  draft: Draft;
  onChange: (draft: Draft) => void;
  onValidity: (valid: boolean) => void;
  busy: boolean;
  readOnly: boolean;
  active: boolean;
  navigation: EditorNavigation | null;
  reviewRequest: number;
};

export function ProjectEditor({
  draft,
  onChange,
  onValidity,
  busy: externalBusy,
  readOnly,
  active,
  navigation,
  reviewRequest,
}: Props) {
  const fieldId = useId();
  const [settingUp, setSettingUp] = useState(false);
  const [reading, setReading] = useState(false);
  const busy = externalBusy || settingUp || reading;
  const latestDraft = useRef(draft);
  latestDraft.current = draft;
  const uploadVersion = useRef(0);
  useEffect(
    () => () => {
      uploadVersion.current++;
    },
    [],
  );
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
  const handledNavigation = useRef(0);
  const [highlight, setHighlight] = useState<{
    file: string;
    code: string;
    line: number | null;
  } | null>(null);
  const highlightedLine =
    highlight?.file === currentFile && highlight.code === currentCode
      ? highlight.line
      : null;
  useEffect(() => {
    if (
      active &&
      navigation &&
      navigation.request !== handledNavigation.current &&
      files[navigation.file] !== undefined
    )
      setSelectedFile(navigation.file);
  }, [navigation?.request, active]);
  useEffect(() => {
    if (
      !active ||
      busy ||
      !navigation ||
      navigation.request === handledNavigation.current ||
      currentFile !== navigation.file
    )
      return;
    const editor = codeRef.current;
    if (!editor) return;
    handledNavigation.current = navigation.request;
    const selection = lineSelection(currentCode, navigation.line);
    setHighlight(
      selection
        ? { file: currentFile, code: currentCode, line: navigation.line }
        : null,
    );
    editor.focus({ preventScroll: true });
    if (selection) {
      editor.setSelectionRange(selection.start, selection.end);
      const row = editor.parentElement?.querySelector<HTMLElement>(
        `[data-line="${navigation.line}"]`,
      );
      row?.scrollIntoView({ block: "center" });
      editor.scrollTop = Math.max(0, (navigation.line! - 4) * 22);
    }
  }, [navigation?.request, currentFile, active, busy]);
  useEffect(() => {
    if (
      !active ||
      busy ||
      !reviewRequest ||
      reviewRequest === handledReview.current
    )
      return;
    if (!draft.code.trim() && currentFile !== entryPath(draft)) {
      setSelectedFile(entryPath(draft));
      return;
    }
    handledReview.current = reviewRequest;
    const invalid = !draft.name.trim()
      ? editorRoot.current?.querySelector<HTMLElement>('[data-setting="name"]')
      : editorRoot.current?.querySelector<HTMLElement>('[aria-invalid="true"]');
    revealField(invalid ?? codeRef.current);
  }, [reviewRequest, active, busy, currentFile]);
  const editorValid =
    !Object.values(errors).some(Boolean) &&
    !!draft.name.trim() &&
    validClassName(draft.class_name) &&
    !!draft.code.trim();
  const valid = readOnly
    ? !!draft.name.trim()
    : editorValid && !settingUp && !reading;
  useEffect(() => onValidity(valid), [valid, onValidity]);
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
      <section className="code-editor">
        <div className="editor-toolbar">
          <span>
            <FileCode2 size={16} />
            <select
              aria-label="Source file"
              value={currentFile}
              disabled={busy}
              onChange={(e) => {
                setSelectedFile(e.target.value);
                setHighlight(null);
              }}
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
          {!readOnly && (
            <>
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
                  e.target.value = "";
                  const version = ++uploadVersion.current;
                  setReading(true);
                  try {
                    const code = await readPythonFile(file);
                    if (version !== uploadVersion.current) return;
                    const currentDraft = latestDraft.current;
                    validation("upload", "");
                    onChange({
                      ...updateFile(currentDraft, currentFile, code),
                      class_name:
                        currentFile === entryPath(currentDraft)
                          ? suggestModuleClass(code) || currentDraft.class_name
                          : currentDraft.class_name,
                    });
                  } catch (e) {
                    if (version === uploadVersion.current)
                      validation("upload", (e as Error).message);
                  } finally {
                    if (version === uploadVersion.current) setReading(false);
                  }
                }}
              />
            </>
          )}
        </div>
        {!readOnly && adding && (
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
              <div
                key={i}
                data-line={i + 1}
                className={
                  highlightedLine === i + 1 ? "editor-error-line" : undefined
                }
              >
                {i + 1}
              </div>
            ))}
          </div>
          <textarea
            ref={codeRef}
            aria-label="Python module code"
            className="code-textarea"
            spellCheck={false}
            value={currentCode}
            disabled={busy}
            readOnly={readOnly}
            onChange={(e) => {
              if (readOnly) return;
              setHighlight(null);
              onChange(updateFile(draft, currentFile, e.target.value));
            }}
            onKeyDown={(e) => {
              if (e.key === "Tab" && !readOnly) {
                e.preventDefault();
                setHighlight(null);
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
          {Object.keys(files).length === 1 ? "file" : "files"} ·{" "}
          {readOnly ? "generated from your model" : "edits saved with each run"}
          <span>Python / PyTorch</span>
        </div>
        {!readOnly && errors.upload && (
          <p className="field-error">{errors.upload}</p>
        )}
      </section>
      {!readOnly && (
        <div className="configuration">
          <details className="config-card disclosure-settings">
            <summary>
              <Braces size={16} /> Model settings
            </summary>
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
                  Relative directory containing your packages, usually . or src.
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
              <p id={`${fieldId}-kwargs`} className="field-error" role="alert">
                {errors.kwargs}
              </p>
            )}
          </details>
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
          <div className="local-note">
            <Info size={17} />
            <p>
              Runs use CPU tensors in evaluation mode. Set input shapes and
              values in Inputs, then generate the diagram.
            </p>
          </div>
          <details className="annotation-note disclosure-settings">
            <summary>Label tensor dimensions</summary>
            <p>
              Add <code># axes: batch, tokens, features</code> to an assignment
              to label its output. Unlabeled dimensions remain neutral.
            </p>
          </details>
        </div>
      )}
    </div>
  );
}
