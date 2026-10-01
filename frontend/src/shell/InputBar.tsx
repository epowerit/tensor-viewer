import { useEffect, useState } from "react";
import { Settings2 } from "lucide-react";
import type { Draft } from "../api/client";
import { parseShape } from "../console/script";
import { withGenerator, withSentence } from "../inputs/samples";
import { TokenStrip } from "../inputs/TokenStrip";

type Props = {
  draft: Draft;
  busy: boolean;
  onChange: (draft: Draft) => void;
  onSettings: () => void;
  onValidity: (valid: boolean) => void;
};

const GENERATORS = [
  ["arange", "0, 1, 2…"],
  ["random", "random"],
  ["ones", "ones"],
  ["zeros", "zeros"],
  ["image", "sample image"],
  ["text", "sentence"],
] as const;

/** The experiment's starting tensor, independent of how its code was created. */
export function InputBar({
  draft,
  busy,
  onChange,
  onSettings,
  onValidity,
}: Props) {
  const [shapeText, setShapeText] = useState(draft.input.shape.join(", "));
  const [shapeError, setShapeError] = useState("");
  const uploaded = draft.input.generator === "uploaded";
  const sentence = draft.input.generator === "text";
  // Another project, example, or saved input may replace the shape.
  const shapeKey = draft.input.shape.join(", ");
  useEffect(() => {
    setShapeText(shapeKey);
    setShapeError("");
  }, [shapeKey]);
  useEffect(() => onValidity(!shapeError), [shapeError, onValidity]);

  function changeShape(text: string) {
    setShapeText(text);
    const shape = parseShape(text);
    if (typeof shape === "string") {
      setShapeError(shape);
      return;
    }
    setShapeError("");
    onChange({
      ...draft,
      input: {
        ...draft.input,
        shape,
        axis_names:
          draft.input.axis_names.length === shape.length
            ? draft.input.axis_names
            : [],
      },
    });
  }
  return (
    <div className="input-bar">
      <div className="input-declaration">
        <code className="input-name">{draft.input_name ?? "x"}</code>
        <span className="input-control-label">Values</span>
        {uploaded ? (
          <code>file</code>
        ) : (
          <select
            aria-label="Input values"
            value={draft.input.generator}
            disabled={busy}
            onChange={(event) =>
              onChange({
                ...draft,
                input: withGenerator(
                  draft.input,
                  event.target.value as Draft["input"]["generator"],
                ),
              })
            }
          >
            {GENERATORS.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        )}
        <span className="input-control-label">Shape</span>
        <input
          aria-label="Input shape"
          value={shapeText}
          disabled={busy || uploaded || sentence}
          aria-invalid={!!shapeError}
          spellCheck={false}
          size={Math.max(6, shapeText.length)}
          onChange={(event) => changeShape(event.target.value)}
        />
        <small>{draft.input.dtype}</small>
        {!!draft.additional_inputs?.length && (
          <small className="input-more">
            + {draft.additional_inputs.map((item) => item.name).join(", ")}
          </small>
        )}
        <button
          className="icon-button"
          aria-label="All input settings"
          title="All input settings"
          onClick={onSettings}
        >
          <Settings2 size={14} />
        </button>
      </div>
      {sentence && (
        <label className="input-sentence">
          <input
            aria-label="Sentence"
            value={draft.input.text ?? ""}
            maxLength={400}
            disabled={busy}
            onChange={(event) =>
              onChange({
                ...draft,
                input: withSentence(draft.input, event.target.value),
              })
            }
          />
          <TokenStrip text={draft.input.text ?? ""} />
        </label>
      )}
      {shapeError && (
        <p className="field-error" role="alert">
          {shapeError}
        </p>
      )}
    </div>
  );
}
