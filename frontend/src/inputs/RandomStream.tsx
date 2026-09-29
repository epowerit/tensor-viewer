import type { Draft } from "../api/client";

export function RandomStream({
  input,
  disabled,
  onChange,
}: {
  input: Draft["input"];
  disabled: boolean;
  onChange: (input: Draft["input"]) => void;
}) {
  if (input.generator !== "random") return null;
  return (
    <div className="input-random-stream">
      <label>
        Random input seed
        <select
          disabled={disabled}
          value={input.random_stream ?? "model"}
          onChange={(event) =>
            onChange({
              ...input,
              random_stream: event.target.value as "input" | "model",
            })
          }
        >
          <option value="input">Independent of model</option>
          <option value="model">Shared with model · legacy</option>
        </select>
      </label>
      <p className="settings-note">
        {input.random_stream === "input"
          ? "The same shape, dtype, and seed reproduce the same input across models in this environment."
          : "Keeps the original behavior: model initialization affects the random input."}
      </p>
    </div>
  );
}
