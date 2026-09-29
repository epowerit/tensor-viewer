import { useEffect, useId, useRef, useState } from "react";
import { Braces, FileCode2, Info, Upload } from "lucide-react";
import type { Draft } from "../api/client";
import { ForwardInputs } from "../inputs/ForwardInputs";
import { WeightLibrary } from "../weights/WeightLibrary";

type Props = {
  draft: Draft;
  onChange: (draft: Draft) => void;
  onValidity: (valid: boolean) => void;
  busy: boolean;
  active: boolean;
};

export function ProjectEditor({
  draft,
  onChange,
  onValidity,
  busy,
  active,
}: Props) {
  const fieldId = useId();
  const [constructor, setConstructor] = useState(
    JSON.stringify(draft.constructor, null, 2),
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const upload = useRef<HTMLInputElement>(null);
  const codeRef = useRef<HTMLTextAreaElement>(null);
  const [inputsValid, setInputsValid] = useState(true);
  const editorValid = !Object.values(errors).some(Boolean);
  useEffect(
    () => onValidity(inputsValid && editorValid),
    [inputsValid, editorValid, onValidity],
  );
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
    <div className="editor-layout">
      <section className="code-editor">
        <div className="editor-toolbar">
          <span>
            <FileCode2 size={16} />
            model.py
          </span>
          <button
            className="secondary-button small"
            disabled={busy}
            onClick={() => upload.current?.click()}
          >
            <Upload size={13} />
            Upload .py
          </button>
          <input
            hidden
            ref={upload}
            type="file"
            accept=".py,text/x-python"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              if (file.size > 50000) {
                validation("upload", "Use a Python file smaller than 50 KB.");
                return;
              }
              const code = await file.text();
              validation("upload", "");
              const found = code.match(
                /class\s+(\w+)\s*\([^)]*(?:nn\.Module|Module)/,
              );
              onChange({
                ...draft,
                code,
                class_name: found?.[1] ?? draft.class_name,
              });
              e.target.value = "";
            }}
          />
        </div>
        <div className="editor-body">
          <div className="editor-line-numbers" aria-hidden="true">
            {draft.code.split("\n").map((_, i) => (
              <div key={i}>{i + 1}</div>
            ))}
          </div>
          <textarea
            ref={codeRef}
            aria-label="Python module code"
            className="code-textarea"
            spellCheck={false}
            value={draft.code}
            disabled={busy}
            onChange={(e) => onChange({ ...draft, code: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Tab") {
                e.preventDefault();
                const start = e.currentTarget.selectionStart;
                const end = e.currentTarget.selectionEnd;
                onChange({
                  ...draft,
                  code:
                    draft.code.slice(0, start) + "    " + draft.code.slice(end),
                });
                requestAnimationFrame(() => {
                  codeRef.current?.setSelectionRange(start + 4, start + 4);
                });
              }
            }}
          />
        </div>
        <div className="editor-note">
          One file · one nn.Module · configurable inputs
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
              disabled={busy}
              maxLength={100}
              onChange={(e) => onChange({ ...draft, name: e.target.value })}
            />
          </label>
          <label>
            Class name
            <input
              value={draft.class_name}
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
              aria-describedby={errors.kwargs ? `${fieldId}-kwargs` : undefined}
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
        </section>
        <WeightLibrary
          draft={draft}
          onChange={onChange}
          busy={busy}
          active={active}
          invalid={!editorValid || !inputsValid}
        />
        <ForwardInputs
          draft={draft}
          onChange={onChange}
          onValidity={setInputsValid}
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
