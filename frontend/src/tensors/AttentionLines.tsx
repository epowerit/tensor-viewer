import { useMemo, useState } from "react";
import type { Tensor } from "../api/client";
import { formatValue, ravel } from "./coordinates";
import { asNumber } from "./margins";
import { useTokenAxes } from "./TokenContext";

/** Above this many words a side, the lines would be a blur. */
const MOST = 48;
const ROW = 14;
/** Lines weaker than this share of their row's strongest are left out. */
const FAINT = 0.04;
/** Words get about twelve characters of room beside the lines. */
const shortWord = (word: string) =>
  word.length > 12 ? `${word.slice(0, 11)}…` : word;

/**
 * Rows of weights that are already a distribution (each in 0…1, summing to
 * 1) are drawn as they are; anything else, like raw scores, as softmax would
 * weigh it, so a masked −∞ draws nothing.
 */
export function attentionRows(values: number[][]): {
  rows: number[][];
  normalized: boolean;
} {
  const distribution = values.every(
    (row) =>
      row.every((v) => v >= 0 && v <= 1) &&
      Math.abs(row.reduce((sum, v) => sum + v, 0) - 1) < 1e-3,
  );
  if (distribution) return { rows: values, normalized: false };
  return {
    normalized: true,
    rows: values.map((row) => {
      const top = Math.max(...row.filter(Number.isFinite), -Infinity);
      if (!Number.isFinite(top)) return row.map(() => 0);
      const exps = row.map((v) => (Number.isFinite(v) ? Math.exp(v - top) : 0));
      const total = exps.reduce((sum, v) => sum + v, 0);
      return exps.map((v) => v / total);
    }),
  };
}

/**
 * How each attention weight moved between two states of the same slice: now
 * minus before, per (query, key), with the largest move either way.
 */
export function attentionChange(now: number[][], before: number[][]) {
  const deltas = now.map((row, q) =>
    row.map((value, k) => value - before[q][k]),
  );
  let largest = 0;
  let gain: { q: number; k: number; delta: number } | null = null;
  deltas.forEach((row, q) =>
    row.forEach((delta, k) => {
      largest = Math.max(largest, Math.abs(delta));
      if (delta > 0 && (!gain || delta > gain.delta)) gain = { q, k, delta };
    }),
  );
  return {
    deltas,
    largest,
    gain: gain as { q: number; k: number; delta: number } | null,
  };
}

/**
 * Attention as lines between words: for a tensor whose last two axes are both
 * a sentence's positions (attention weights, or the scores before them), the
 * querying words down the left, the words they read down the right, and a
 * line for each pair as strong as its weight. It shows the slice the grid
 * shows (the batch and head of the selected cell). Pointing at a word keeps
 * its own lines; choosing a word or line selects that cell.
 */
export function AttentionLines({
  tensor,
  coords,
  onPick,
  before = null,
  beforeLabel = "the run before",
}: {
  tensor: Tensor;
  coords: number[];
  onPick: (index: number) => void;
  /** The same state elsewhere (the run before): offers the change view. */
  before?: Tensor | null;
  beforeLabel?: string;
}) {
  const [mode, setMode] = useState<"now" | "change">("now");
  const wordsOf = useTokenAxes(tensor);
  const rank = tensor.shape.length;
  const queries = rank >= 2 ? wordsOf(rank - 2) : null;
  const keys = rank >= 2 ? wordsOf(rank - 1) : null;
  const [focus, setFocus] = useState<{ side: "q" | "k"; at: number } | null>(
    null,
  );
  const lead = coords.slice(0, -2);
  const leadKey = lead.join();
  const inline = tensor.values?.length === tensor.numel;
  const slice = useMemo(() => {
    if (!queries || !keys || !inline) return null;
    const values = queries.map((_, q) =>
      keys.map((_, k) =>
        asNumber(tensor.values[ravel([...lead, q, k], tensor.shape)]),
      ),
    );
    return { values, ...attentionRows(values) };
    // `leadKey` names the slice.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tensor, leadKey, !!queries, !!keys, inline]);
  // The same slice in the state compared with, as weights.
  const change = useMemo(() => {
    if (
      !slice ||
      !before ||
      before.values?.length !== before.numel ||
      before.shape.join() !== tensor.shape.join()
    )
      return null;
    const values = slice.values.map((row, q) =>
      row.map((_, k) =>
        asNumber(before.values[ravel([...lead, q, k], tensor.shape)]),
      ),
    );
    return attentionChange(slice.rows, attentionRows(values).rows);
    // `leadKey` names the slice.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slice, before, leadKey]);
  const changing = mode === "change" && !!change;
  if (
    !queries ||
    !keys ||
    !slice ||
    queries.length > MOST ||
    keys.length > MOST
  )
    return null;
  const selectedQuery = coords[rank - 2];
  const shown = focus ?? { side: "q" as const, at: selectedQuery };
  const height = Math.max(queries.length, keys.length) * ROW + 4;
  const width = 260,
    left = 92,
    right = width - 92;
  const y = (at: number) => 2 + at * ROW + ROW / 2;
  const leading = lead
    .map((at, axis) => `${tensor.axes[axis] || `axis ${axis}`} ${at}`)
    .join(" · ");
  const pick = (q: number, k: number) =>
    onPick(ravel([...lead, q, k], tensor.shape));
  const lines = slice.rows.flatMap((row, q) => {
    const top = Math.max(...row);
    return row.flatMap((weight, k) => {
      // In the change view, a line is as strong as its weight moved.
      const delta = change ? change.deltas[q][k] : 0;
      const share = changing
        ? change!.largest > 0
          ? Math.abs(delta) / change!.largest
          : 0
        : top > 0
          ? weight / top
          : 0;
      if (share < FAINT) return [];
      const lit = shown.side === "q" ? shown.at === q : shown.at === k;
      return [{ q, k, weight, share, lit, delta }];
    });
  });
  return (
    <div className="attention-lines">
      <span className="attention-lines-label">
        {changing ? "Attention change" : "Attention"}
        {leading ? ` · ${leading}` : ""}
        {slice.normalized ? " · as softmax weighs these scores" : ""}
        {change && (
          <span className="attention-modes" role="group">
            <button
              type="button"
              aria-pressed={!changing}
              onClick={() => setMode("now")}
            >
              now
            </button>
            <button
              type="button"
              aria-pressed={changing}
              onClick={() => setMode("change")}
              title={`How each weight moved since ${beforeLabel}: warm where a word attends more, cool where it attends less`}
            >
              change
            </button>
          </span>
        )}
      </span>
      {changing && (
        <span className="attention-lines-summary">
          {change!.gain
            ? `Since ${beforeLabel}, the largest gain: ${queries[change!.gain.q]} → ${keys[change!.gain.k]} +${formatValue(change!.gain.delta)}`
            : `No weight grew since ${beforeLabel}`}
        </span>
      )}
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label={`Which words each word attends to${leading ? `, ${leading}` : ""}`}
        onPointerLeave={() => setFocus(null)}
      >
        {lines
          .sort((a, b) => Number(a.lit) - Number(b.lit))
          .map(({ q, k, weight, share, lit, delta }) => (
            <line
              key={`${q}/${k}`}
              className={
                changing
                  ? `${delta > 0 ? "attention-gained" : "attention-lost"}`
                  : lit
                    ? "attention-lit"
                    : undefined
              }
              x1={left + 4}
              y1={y(q)}
              x2={right - 4}
              y2={y(k)}
              strokeOpacity={(changing
                ? (0.15 + 0.85 * share) * (lit ? 1 : 0.55)
                : lit
                  ? 0.25 + 0.75 * share
                  : 0.12 * share
              ).toFixed(3)}
              strokeWidth={
                changing ? 0.8 + 2.2 * share : lit ? 1 + 2.5 * share : 1
              }
              onClick={() => pick(q, k)}
            >
              <title>{`${queries[q]} → ${keys[k]}: ${formatValue(weight)}${slice.normalized ? ` (score ${formatValue(slice.values[q][k])})` : ""}${change ? ` · was ${formatValue(weight - delta)} (${delta >= 0 ? "+" : ""}${formatValue(delta)})` : ""}`}</title>
            </line>
          ))}
        {queries.map((word, q) => (
          <text
            key={`q${q}`}
            x={left}
            y={y(q)}
            textAnchor="end"
            dominantBaseline="middle"
            className={
              shown.side === "q" && shown.at === q ? "attention-word-lit" : ""
            }
            onPointerEnter={() => setFocus({ side: "q", at: q })}
            onClick={() =>
              pick(q, slice.rows[q].indexOf(Math.max(...slice.rows[q])))
            }
          >
            {shortWord(word)}
          </text>
        ))}
        {keys.map((word, k) => (
          <text
            key={`k${k}`}
            x={right}
            y={y(k)}
            dominantBaseline="middle"
            className={
              shown.side === "k" && shown.at === k ? "attention-word-lit" : ""
            }
            onPointerEnter={() => setFocus({ side: "k", at: k })}
          >
            {shortWord(word)}
          </text>
        ))}
      </svg>
    </div>
  );
}
