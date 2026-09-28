import { useId, useRef, useState } from "react";
import { Braces, FileCode2, Info, Upload } from "lucide-react";
import type { Draft } from "../api/client";

type Props = {
  draft: Draft;
  onChange: (draft: Draft) => void;
  onValidity: (valid: boolean) => void;
  busy: boolean;
};

export function ProjectEditor({ draft, onChange, onValidity, busy }: Props) {
  const fieldId = useId();
  const [shape, setShape] = useState(draft.input.shape.join(", "));
  const [constructor, setConstructor] = useState(
    JSON.stringify(draft.constructor, null, 2),
  );
  const [axes, setAxes] = useState(draft.input.axis_names.join(", "));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const upload = useRef<HTMLInputElement>(null);
  const codeRef = useRef<HTMLTextAreaElement>(null);
  function validation(field: string, message: string) {
    const next = { ...errors, [field]: message };
    setErrors(next);
    onValidity(!Object.values(next).some(Boolean));
  }
  function changeShape(value: string) {
    setShape(value);
    const dimensions = value
      .split(/[,x×\s]+/)
      .filter(Boolean)
      .map(Number);
    const valid =
      dimensions.length > 0 &&
      dimensions.length <= 6 &&
      dimensions.every((n) => Number.isInteger(n) && n > 0) &&
      dimensions.reduce((a, b) => a * b, 1) <= 2 ** 40;
    const nextErrors = {
      ...errors,
      shape: !valid
        ? "Use 1–6 positive dimensions, up to 2^40 logical elements."
        : draft.capture_mode !== "shapes" &&
            dimensions.reduce((a, b) => a * b, 1) > 8_388_608
          ? "Switch to Shapes for this size, or use at most 8,388,608 elements for a value run."
          : "",
      axes: valid ? "" : (errors.axes ?? ""),
    };
    setErrors(nextErrors);
    onValidity(!Object.values(nextErrors).some(Boolean));
    if (valid) {
      const labels = axes
        .split(",")
        .map((a) => a.trim())
        .filter(Boolean);
      if (labels.length !== dimensions.length) setAxes("");
      onChange({
        ...draft,
        input: {
          ...draft.input,
          shape: dimensions,
          axis_names: labels.length === dimensions.length ? labels : [],
        },
      });
    }
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
          One file · one nn.Module · forward(x)<span>Python / PyTorch</span>
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
        <section className="config-card">
          <div className="section-label">Input tensor</div>
          <div className="capture-mode" aria-label="Recording mode">
            {(["values", "shapes"] as const).map((mode) => (
              <button
                type="button"
                key={mode}
                disabled={busy}
                aria-pressed={(draft.capture_mode ?? "values") === mode}
                onClick={() => {
                  onChange({ ...draft, capture_mode: mode });
                  const count = draft.input.shape.reduce((a, b) => a * b, 1);
                  const dimensions = shape
                    .split(/[,x×\s]+/)
                    .filter(Boolean)
                    .map(Number);
                  const valid =
                    dimensions.length > 0 &&
                    dimensions.length <= 6 &&
                    dimensions.every((n) => Number.isInteger(n) && n > 0) &&
                    dimensions.reduce((a, b) => a * b, 1) <= 2 ** 40;
                  validation(
                    "shape",
                    !valid
                      ? "Use 1–6 positive dimensions, up to 2^40 logical elements."
                      : mode === "values" && count > 8_388_608
                        ? "Use Shapes for this size, or at most 8,388,608 elements for a value run."
                        : "",
                  );
                }}
              >
                {mode === "values" ? "Values & shapes" : "Shapes only"}
              </button>
            ))}
          </div>
          <p className="capture-description">
            {draft.capture_mode === "shapes"
              ? "Trace large shapes and layout without allocating tensor values. Data-dependent code may require a value run."
              : "Compute real values. Only the visible window is loaded into the diagram."}
          </p>
          <label>
            Shape
            <input
              aria-label="Input shape"
              aria-invalid={!!errors.shape}
              aria-describedby={`${fieldId}-shape`}
              value={shape}
              disabled={busy}
              onChange={(e) => changeShape(e.target.value)}
            />
          </label>
          <p
            id={`${fieldId}-shape`}
            className={errors.shape ? "field-error" : "field-hint"}
            role={errors.shape ? "alert" : undefined}
          >
            {errors.shape ||
              `${draft.input.shape.reduce((a, b) => a * b, 1).toLocaleString()} elements · ${draft.capture_mode === "shapes" ? "shape preview" : "values loaded by window"}`}
          </p>
          <label>
            Values
            <select
              value={draft.input.generator}
              disabled={busy}
              onChange={(e) =>
                onChange({
                  ...draft,
                  input: {
                    ...draft.input,
                    generator: e.target.value as Draft["input"]["generator"],
                    dtype:
                      e.target.value === "random" &&
                      draft.input.dtype === "int64"
                        ? "float32"
                        : draft.input.dtype,
                  },
                })
              }
            >
              <option value="arange">Sequential · 0, 1, 2, 3…</option>
              <option value="random">Seeded random · normal</option>
              <option value="ones">Ones</option>
              <option value="zeros">Zeros</option>
            </select>
          </label>
          <div className="field-pair">
            <label>
              Data type
              <select
                value={draft.input.dtype}
                disabled={busy}
                onChange={(e) =>
                  onChange({
                    ...draft,
                    input: {
                      ...draft.input,
                      dtype: e.target.value as Draft["input"]["dtype"],
                    },
                  })
                }
              >
                <option>float32</option>
                <option>float64</option>
                <option disabled={draft.input.generator === "random"}>
                  int64
                </option>
              </select>
            </label>
            <label>
              Seed
              <input
                type="number"
                min="0"
                max="4294967295"
                value={draft.input.seed}
                disabled={busy}
                onChange={(e) =>
                  onChange({
                    ...draft,
                    input: { ...draft.input, seed: Number(e.target.value) },
                  })
                }
              />
            </label>
          </div>
          <label>
            Axis names <span className="optional">optional</span>
            <input
              value={axes}
              aria-invalid={!!errors.axes}
              aria-describedby={errors.axes ? `${fieldId}-axes` : undefined}
              disabled={busy}
              placeholder="batch, tokens, features"
              onChange={(e) => {
                setAxes(e.target.value);
                const labels = e.target.value
                  .split(",")
                  .map((a) => a.trim())
                  .filter(Boolean);
                validation(
                  "axes",
                  labels.length && labels.length !== draft.input.shape.length
                    ? "Supply one name per axis, or leave this blank."
                    : "",
                );
                onChange({
                  ...draft,
                  input: { ...draft.input, axis_names: labels },
                });
              }}
            />
          </label>
          {errors.axes && (
            <p id={`${fieldId}-axes`} className="field-error" role="alert">
              {errors.axes}
            </p>
          )}
        </section>
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
