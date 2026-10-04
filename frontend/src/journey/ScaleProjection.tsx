import { useMemo, useState } from "react";
import type { Run } from "../api/client";
import { bytesText, flopsText } from "./cost";
import { project, recordedSymbols } from "./projection";

const times = (ratio: number) =>
  ratio >= 100
    ? `×${Math.round(ratio).toLocaleString()}`
    : `×${Number(ratio.toPrecision(2))}`;

/**
 * The run at other sizes: one field per symbol of the input (B, T), and the
 * compute and live-memory peak every tensor's symbolic shape implies there,
 * next to the recorded run's. Weights keep their size.
 */
export function ScaleProjection({
  run,
  labelsOf,
}: {
  run: Run;
  labelsOf: (tensorId: string) => (string | null)[] | null;
}) {
  const input = run.trace.tensors[run.trace.input_ids[0]];
  const recorded = useMemo(
    () => recordedSymbols(input, labelsOf(input?.id ?? "")),
    [input, labelsOf],
  );
  const symbols = Object.keys(recorded);
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const values = Object.fromEntries(
    symbols.map((symbol) => {
      const typed = Number(chosen[symbol]);
      return [
        symbol,
        chosen[symbol] && Number.isInteger(typed) && typed > 0
          ? typed
          : recorded[symbol],
      ];
    }),
  );
  const key = symbols.map((symbol) => values[symbol]).join();
  const base = useMemo(
    () => project(run.trace, labelsOf, recorded),
    [run, labelsOf, recorded],
  );
  const scaled = useMemo(
    () => project(run.trace, labelsOf, values),
    // `key` names the chosen sizes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [run, labelsOf, key],
  );
  if (!symbols.length) return null;
  const changed = symbols.some((symbol) => values[symbol] !== recorded[symbol]);
  return (
    <span
      className="lens-weights scale-projection"
      title="Every tensor resized by its symbolic shape; weights keep their size. Compute and memory as the lenses estimate them, from shapes."
    >
      at
      {symbols.map((symbol) => (
        <label key={symbol}>
          {symbol} =
          <input
            value={chosen[symbol] ?? String(recorded[symbol])}
            inputMode="numeric"
            spellCheck={false}
            aria-label={`${symbol}, recorded as ${recorded[symbol]}`}
            onChange={(event) =>
              setChosen((current) => ({
                ...current,
                [symbol]: event.target.value,
              }))
            }
          />
        </label>
      ))}
      <span>
        Σ {flopsText(scaled.flops)}
        {changed && base.flops ? ` (${times(scaled.flops / base.flops)})` : ""}
        {" · "}peak {bytesText(scaled.peakBytes)}
        {changed && base.peakBytes
          ? ` (${times(scaled.peakBytes / base.peakBytes)})`
          : ""}
        {scaled.kept
          ? ` · ${scaled.kept} ${scaled.kept === 1 ? "tensor keeps its recorded size" : "tensors keep their recorded sizes"}`
          : ""}
      </span>
    </span>
  );
}
