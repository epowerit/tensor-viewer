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
  ["arange", "sequential"],
  ["random", "random"],
  ["ones", "ones"],
  ["zeros", "zeros"],
  ["image", "sample image"],
  ["text", "sentence"],
] as const;

type Input = Draft["input"];

/** Every tensor the forward call receives: the main input, then the rest. */
export function forwardInputs(draft: Draft) {
  return [
    { name: draft.input_name ?? "x", input: draft.input },
    ...(draft.additional_inputs ?? []),
  ];
}

/** The draft with its index-th forward input replaced. */
export function withInputAt(draft: Draft, index: number, input: Input): Draft {
  if (index === 0) return { ...draft, input };
  return {
    ...draft,
    additional_inputs: (draft.additional_inputs ?? []).map((item, i) =>
      i === index - 1 ? { ...item, input } : item,
    ),
  };
}

/** The experiment's starting tensors, independent of how its code was created. */
export function InputBar({
  draft,
  busy,
  onChange,
  onSettings,
  onValidity,
}: Props) {
  const inputs = forwardInputs(draft);
  const [chosenIndex, setChosen] = useState(0);
  const chosen = Math.min(chosenIndex, inputs.length - 1);
  const { name, input } = inputs[chosen];
  const [shapeText, setShapeText] = useState(input.shape.join(", "));
  const [shapeError, setShapeError] = useState("");
  const uploaded = input.generator === "uploaded";
  const sentence = input.generator === "text";
  // Another project, example, saved input, or input tab may replace the shape.
  const shapeKey = `${chosen}:${input.shape.join(", ")}`;
  useEffect(() => {
    setShapeText(input.shape.join(", "));
    setShapeError("");
  }, [shapeKey]);
  useEffect(() => onValidity(!shapeError), [shapeError, onValidity]);
  const change = (next: Input) => onChange(withInputAt(draft, chosen, next));

  function changeShape(text: string) {
    setShapeText(text);
    const shape = parseShape(text);
    if (typeof shape === "string") {
      setShapeError(shape);
      return;
    }
    setShapeError("");
    change({
      ...input,
      shape,
      axis_names:
        input.axis_names.length === shape.length ? input.axis_names : [],
    });
  }
  return (
    <div className="input-bar">
      {inputs.length > 1 && (
        <div className="input-tabs" role="tablist" aria-label="Forward inputs">
          {inputs.map((item, index) => (
            <button
              key={item.name}
              role="tab"
              aria-selected={index === chosen}
              title={`Edit ${item.name}, input ${index + 1} of ${inputs.length}`}
              onClick={() => setChosen(index)}
            >
              <code>{item.name}</code>
              <small>
                {item.input.generator === "text"
                  ? "sentence"
                  : `[${item.input.shape.join(" × ")}]`}
              </small>
            </button>
          ))}
        </div>
      )}
      <div className="input-declaration">
        <code className="input-name">{name}</code>
        <span className="input-control-label">Values</span>
        {uploaded ? (
          <code>file</code>
        ) : (
          <select
            aria-label="Input values"
            value={input.generator}
            disabled={busy}
            onChange={(event) =>
              change(
                withGenerator(input, event.target.value as Input["generator"]),
              )
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
        <small>{input.dtype}</small>
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
            aria-label={`Sentence for ${name}`}
            value={input.text ?? ""}
            maxLength={400}
            disabled={busy}
            onChange={(event) =>
              change(withSentence(input, event.target.value))
            }
          />
          <TokenStrip text={input.text ?? ""} />
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
