import { useId, useRef, useState } from "react";
import { LoaderCircle, Upload } from "lucide-react";
import { sourceClasses, type CodeEntry } from "./codeImport";
import "./codeImport.css";

export type CodeSource = { code: string; entry: CodeEntry; className: string };
type Props = {
  value: CodeSource;
  onChange: (value: CodeSource) => void;
  disabled: boolean;
  onReading: (reading: boolean) => void;
};

export function CodeImport({ value, onChange, disabled, onReading }: Props) {
  const upload = useRef<HTMLInputElement>(null);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [reading, setReading] = useState(false);
  const [entryChosen, setEntryChosen] = useState(false);
  const id = useId();
  const codeId = `${id}-code`;
  const helpId = `${id}-help`;
  const errorId = `${id}-error`;
  const classesId = `${id}-classes`;
  const candidates = sourceClasses(value.code);
  function replace(code: string, uploaded = false) {
    const names = sourceClasses(code);
    const entry =
      uploaded || !entryChosen
        ? names.length
          ? "module"
          : "statements"
        : value.entry;
    onChange({
      code,
      entry,
      className: names.includes(value.className)
        ? value.className
        : (names.at(-1) ?? value.className),
    });
    setError("");
  }
  return (
    <section
      className="code-import"
      aria-label="Python code"
      aria-busy={reading}
    >
      <div className="code-import-heading">
        <label htmlFor={codeId}>Paste code</label>
        <button
          className="secondary-button"
          type="button"
          disabled={disabled}
          onClick={() => upload.current?.click()}
        >
          {reading ? (
            <LoaderCircle size={14} className="spin" />
          ) : (
            <Upload size={14} />
          )}
          {reading ? "Reading…" : "Upload .py"}
        </button>
      </div>
      <input
        ref={upload}
        hidden
        aria-label="Upload Python file"
        type="file"
        disabled={disabled}
        accept=".py,text/x-python,text/plain"
        onChange={async (event) => {
          const file = event.currentTarget.files?.[0];
          event.currentTarget.value = "";
          if (!file) return;
          if (!file.name.toLowerCase().endsWith(".py") || file.size > 500_000) {
            setError("Choose a Python (.py) file up to 500 KB.");
            return;
          }
          setReading(true);
          onReading(true);
          try {
            replace(await file.text(), true);
            setFileName(file.name);
            setEntryChosen(false);
          } catch {
            setError("This file could not be read. Try pasting its code.");
          } finally {
            setReading(false);
            onReading(false);
          }
        }}
      />
      <textarea
        id={codeId}
        aria-describedby={`${helpId}${error ? ` ${errorId}` : ""}`}
        value={value.code}
        disabled={disabled}
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        rows={8}
        placeholder="Paste tensor statements or an nn.Module class…"
        onChange={(event) => {
          replace(event.target.value);
          setFileName("");
        }}
      />
      {fileName && (
        <small className="code-import-file" role="status">
          Loaded {fileName}
        </small>
      )}
      <div className="code-entry-fields">
        <label>
          Run as
          <select
            value={value.entry}
            disabled={disabled}
            onChange={(event) => {
              setEntryChosen(true);
              onChange({ ...value, entry: event.target.value as CodeEntry });
            }}
          >
            <option value="statements">Tensor statements</option>
            <option value="module">Model class</option>
          </select>
        </label>
        {value.entry === "module" && (
          <label>
            Model class
            <input
              value={value.className}
              disabled={disabled}
              list={classesId}
              placeholder="Attention"
              onChange={(event) =>
                onChange({ ...value, className: event.target.value })
              }
            />
            <datalist id={classesId}>
              {candidates.map((name) => (
                <option value={name} key={name} />
              ))}
            </datalist>
          </label>
        )}
      </div>
      <p id={helpId}>
        {value.entry === "module"
          ? "Choose the model class. Configure its arguments and tensor inputs in the workspace."
          : "Use x as the input tensor. torch, nn, F and math are available. The last result becomes the output."}
      </p>
      {error && (
        <p id={errorId} className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
