import { useContext, useState } from "react";
import { api, type Sensitivity, type Tensor } from "../api/client";
import { formatValue, unravel } from "./coordinates";
import { TensorUseContext } from "./TensorUseContext";
import { TokenContext } from "./TokenContext";
import { GradientLensContext } from "../journey/GradientLens";

/** Input cells named in the summary, strongest first. */
const TOP = 4;
/** Above this many cells the input is not drawn, only summarized. */
const DRAWN = 4096;

/** A cell's shade: rose for a positive slope, blue for a negative one. */
function shade(value: number, largest: number) {
  if (!largest || !value) return undefined;
  const strength = Math.min(1, Math.abs(value) / largest);
  const color = value > 0 ? "#e98a9a" : "#7fb0e8";
  return `color-mix(in srgb, ${color} ${Math.round(12 + strength * 88)}%, transparent)`;
}

/** Whether a value can be traced back to the input cells that move it. */
export function asksSensitivity(
  flow: { runId?: string; trace?: { input_ids: string[] } } | null,
  tensor: Tensor,
) {
  const trace = flow?.trace;
  return (
    !!flow?.runId &&
    !!trace?.input_ids[0] &&
    !trace.input_ids.includes(tensor.id) &&
    // A weight is set before the input arrives: nothing in it moves it.
    tensor.role !== "parameter" &&
    tensor.dtype.startsWith("float") &&
    tensor.value_source !== "shape"
  );
}

/**
 * Which input cells this value depends on, and how strongly: the gradient of
 * the selected cell with respect to the model's input, from one backward pass
 * over the recorded code (nothing is saved). The input is drawn as a small
 * map, rose where raising a cell raises this value and blue where it lowers
 * it; a sentence input shows its words, by how much each one's embedding
 * moves it. The strongest cells open the input with that cell selected.
 */
export function SensitivityMap({
  tensor,
  index,
}: {
  tensor: Tensor;
  index: number;
}) {
  const flow = useContext(TensorUseContext);
  const tokens = useContext(TokenContext);
  const gradientLens = useContext(GradientLensContext);
  const key = `${flow?.runId}/${tensor.id}/${index}`;
  const [found, setFound] = useState<{
    key: string;
    result?: Sensitivity;
    error?: string;
    busy?: boolean;
  } | null>(null);
  const trace = flow?.trace;
  const inputId = trace?.input_ids[0];
  if (!flow?.runId || !trace || !inputId || !asksSensitivity(flow, tensor))
    return null;
  const input = trace.tensors[inputId];
  const current = found?.key === key ? found : null;
  const ask = () => {
    setFound({ key, busy: true });
    api
      .sensitivity(flow.runId!, tensor.id, index)
      .then((result) => setFound({ key, result }))
      .catch((error: Error) => setFound({ key, error: error.message }));
  };
  const coords = `${tensor.name}[${unravel(index, tensor.shape).join(", ")}]`;
  if (!current?.result)
    return (
      <div className="sensitivity">
        <button
          type="button"
          className="sensitivity-ask"
          disabled={current?.busy}
          onClick={ask}
          title="One backward pass through the recorded code: ∂ this value / ∂ each input cell. Nothing is saved."
        >
          {current?.busy
            ? "Finding what moves it…"
            : `Which ${input?.name ?? "input"} cells move ${coords}?`}
        </button>
        {gradientLens && (
          <button
            type="button"
            className="sensitivity-ask"
            onClick={() => gradientLens.show({ tensorId: tensor.id, index })}
            title="Colour every step on the canvas by the size of ∂ this value / ∂ its result"
          >
            Gradient on every step
          </button>
        )}
        {current?.error && (
          <small className="sensitivity-note">{current.error}</small>
        )}
      </div>
    );
  const { result } = current;
  const values = result.values.map((value) => value ?? 0);
  const largest = values.reduce((top, v) => Math.max(top, Math.abs(v)), 0);
  const moving = values.filter((value) => value !== 0).length;
  const order = values
    .map((value, at) => ({ value, at }))
    .filter(({ value }) => value !== 0)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, TOP);
  const words =
    input && result.kind === "embedding"
      ? (tokens?.(input, input.shape.length - 1)?.words ?? null)
      : null;
  const cellName = (at: number) =>
    words
      ? `"${words[at % words.length]}"`
      : `${input?.name ?? "x"}[${unravel(at, input?.shape ?? []).join(", ")}]`;
  const open = (at: number) =>
    input && flow.go(`input-${input.id}`, input.id, at);
  const columns = input?.shape.at(-1) ?? values.length;
  return (
    <div
      className="sensitivity"
      aria-label={`How much each ${input?.name ?? "input"} cell moves ${coords}`}
    >
      <span className="sensitivity-label">
        {result.kind === "embedding"
          ? `‖∂ ${coords} / ∂ embedding‖ per word`
          : `∂ ${coords} / ∂ ${input?.name ?? "x"}`}
        {" · "}
        {moving
          ? `${moving} of ${values.length} ${words ? "words" : "cells"} move it`
          : `no ${words ? "word" : "input cell"} moves it`}
      </span>
      {words ? (
        <div className="sensitivity-words">
          {values.map((value, at) => (
            <button
              type="button"
              key={at}
              style={{ background: shade(value, largest) }}
              onClick={() => open(at)}
              title={`${words[at % words.length]}: ${formatValue(value)}`}
            >
              {words[at % words.length]}
            </button>
          ))}
        </div>
      ) : values.length <= DRAWN ? (
        <div
          className="sensitivity-grid"
          style={{
            gridTemplateColumns: `repeat(${columns}, minmax(4px, 10px))`,
          }}
          role="img"
          aria-label={`${moving} cells of ${input?.name ?? "the input"} move this value`}
        >
          {values.map((value, at) => (
            <span
              key={at}
              style={{ background: shade(value, largest) }}
              title={`${cellName(at)}: ${formatValue(value)}`}
              onClick={() => open(at)}
            />
          ))}
        </div>
      ) : null}
      {order.length > 0 && (
        <span className="sensitivity-top">
          {order.map(({ value, at }) => (
            <button
              type="button"
              key={at}
              onClick={() => open(at)}
              title="Open the input with this cell selected"
            >
              {cellName(at)}{" "}
              <b>
                {value > 0 && !words ? "+" : ""}
                {formatValue(value)}
              </b>
            </button>
          ))}
        </span>
      )}
    </div>
  );
}
