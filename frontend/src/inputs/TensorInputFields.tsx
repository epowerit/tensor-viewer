import { useEffect, useId, useState } from "react";
import type { Draft } from "../api/client";
import { InputLibrary } from "./InputLibrary";
import { RandomStream } from "./RandomStream";
import { UploadedInputNotice } from "./UploadedInputNotice";
import { inputIssue } from "./fixtures";
import { withGenerator, withSentence } from "./samples";
import { TokenStrip } from "./TokenStrip";

export function TensorInputFields({
  input,
  captureMode,
  busy,
  active,
  primary,
  onChange,
  onValidity,
}: {
  input: Draft["input"];
  captureMode: "values" | "shapes";
  busy: boolean;
  active: boolean;
  primary: boolean;
  onChange: (input: Draft["input"], mode?: "values" | "shapes") => void;
  onValidity: (valid: boolean) => void;
}) {
  const id = useId();
  const [shape, setShape] = useState(input.shape.join(", "));
  const [axes, setAxes] = useState(input.axis_names.join(", "));
  useEffect(() => setShape(input.shape.join(", ")), [input.shape.join(",")]);
  useEffect(
    () => setAxes(input.axis_names.join(", ")),
    [input.axis_names.join(",")],
  );
  const dimensions = shape
    .split(/[,x×\s]+/)
    .filter(Boolean)
    .map(Number);
  const labels = axes
    .split(",")
    .map((axis) => axis.trim())
    .filter(Boolean);
  const issue = inputIssue(
    { ...input, shape: dimensions, axis_names: labels },
    captureMode,
  );
  useEffect(() => onValidity(!issue), [issue, onValidity]);
  return (
    <div className="tensor-input-fields">
      <InputLibrary
        active={active}
        input={input}
        captureMode={captureMode}
        busy={busy}
        invalid={!!issue}
        onApply={(settings) => {
          setShape(settings.input.shape.join(", "));
          setAxes(settings.input.axis_names.join(", "));
          onChange(settings.input, settings.capture_mode);
        }}
      />
      <UploadedInputNotice input={input} disabled={busy} onChange={onChange} />
      <label>
        Shape
        <input
          aria-label="Input shape"
          aria-invalid={
            !dimensions.length ||
            dimensions.length > 6 ||
            dimensions.some((n) => !Number.isInteger(n) || n < 1) ||
            dimensions.reduce((a, b) => a * b, 1) >
              (captureMode === "shapes" ? 2 ** 40 : 8_388_608)
          }
          aria-describedby={issue ? `${id}-issue` : undefined}
          value={shape}
          disabled={busy || !!input.uploaded || input.generator === "text"}
          onChange={(event) => {
            const value = event.target.value;
            setShape(value);
            const next = value
              .split(/[,x×\s]+/)
              .filter(Boolean)
              .map(Number);
            if (
              next.length > 0 &&
              next.length <= 6 &&
              next.every((n) => Number.isInteger(n) && n > 0) &&
              next.reduce((a, b) => a * b, 1) <= 2 ** 40
            ) {
              const nextAxes = labels.length === next.length ? labels : [];
              setAxes(nextAxes.join(", "));
              onChange({ ...input, shape: next, axis_names: nextAxes });
            }
          }}
        />
      </label>
      <p className="field-hint">
        {input.shape.reduce((a, b) => a * b, 1).toLocaleString()} elements ·{" "}
        {captureMode === "shapes" ? "shape preview" : "values loaded by window"}
      </p>
      <label>
        Values
        <select
          value={input.generator}
          disabled={busy}
          onChange={(event) =>
            onChange(
              withGenerator(
                input,
                event.target.value as Draft["input"]["generator"],
              ),
            )
          }
        >
          {input.uploaded && (
            <option value="uploaded">Uploaded .npy values</option>
          )}
          <option value="arange">Sequential · 0, 1, 2, 3…</option>
          <option value="random">Seeded random · normal</option>
          <option value="ones">Ones</option>
          <option value="zeros">Zeros</option>
          <option value="image">Sample image · disc on a gradient</option>
          <option value="text">Sentence · token ids</option>
        </select>
      </label>
      {input.generator === "text" && (
        <label>
          Sentence
          <input
            aria-label="Sentence"
            value={input.text ?? ""}
            maxLength={400}
            disabled={busy}
            onChange={(event) =>
              onChange(withSentence(input, event.target.value))
            }
          />
          <TokenStrip text={input.text ?? ""} />
        </label>
      )}
      {input.generator === "image" && (
        <p className="field-hint">
          The last two axes are height and width; the axis before them is the
          color channel. Tensors whose axes are named height and width can be
          viewed as pixels.
        </p>
      )}
      <div className="field-pair">
        <label>
          Data type
          <select
            value={input.dtype}
            disabled={busy || !!input.uploaded || input.generator === "text"}
            onChange={(event) =>
              onChange({
                ...input,
                dtype: event.target.value as Draft["input"]["dtype"],
              })
            }
          >
            <option>float32</option>
            <option>float64</option>
            <option
              disabled={
                input.generator === "random" || input.generator === "image"
              }
            >
              int64
            </option>
          </select>
        </label>
        <label>
          {primary ? "Model / input seed" : "Input seed"}
          <input
            type="number"
            min="0"
            max="4294967295"
            value={input.seed}
            disabled={busy}
            onChange={(event) =>
              onChange({ ...input, seed: Number(event.target.value) })
            }
          />
        </label>
      </div>
      {primary && (
        <p className="field-hint">
          The first input sets the model seed and its floating-point data type.
        </p>
      )}
      <RandomStream input={input} disabled={busy} onChange={onChange} />
      <label>
        Axis names <span className="optional">optional</span>
        <input
          value={axes}
          aria-invalid={
            labels.length > 0 && labels.length !== input.shape.length
          }
          aria-describedby={issue ? `${id}-issue` : undefined}
          disabled={busy}
          placeholder="batch, tokens, features"
          onChange={(event) => {
            setAxes(event.target.value);
            onChange({
              ...input,
              axis_names: event.target.value
                .split(",")
                .map((axis) => axis.trim())
                .filter(Boolean),
            });
          }}
        />
      </label>
      {issue && (
        <p id={`${id}-issue`} className="field-error" role="alert">
          {issue}
        </p>
      )}
    </div>
  );
}
