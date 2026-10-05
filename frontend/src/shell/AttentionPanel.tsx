import { useEffect, useMemo, useRef, useState } from "react";
import type { Run } from "../api/client";
import {
  attentionMaps,
  headWeights,
  type AttentionMap,
} from "../tensors/attentionMaps";
import { valuesOf } from "../tensors/loadValues";
import { lensWords } from "./LogitLensPanel";

const THUMB = 76;
const percent = (p: number) => `${Math.round(p * 100)}%`;

/** A head's weights as a small picture: brighter where more weight goes. */
function HeadThumb({
  rows,
  active,
  label,
  onPick,
}: {
  rows: number[][];
  active: boolean;
  label: string;
  onPick: () => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    const cell = THUMB / Math.max(rows.length, rows[0]?.length ?? 1);
    context.clearRect(0, 0, THUMB, THUMB);
    rows.forEach((row, q) =>
      row.forEach((value, k) => {
        context.fillStyle = `rgba(127, 209, 199, ${Math.min(1, Math.max(0, value)).toFixed(3)})`;
        context.fillRect(k * cell, q * cell, Math.ceil(cell), Math.ceil(cell));
      }),
    );
  }, [rows]);
  return (
    <button
      type="button"
      className={`attention-thumb${active ? " is-active" : ""}`}
      aria-pressed={active}
      onClick={onPick}
      title={`${label}: enlarge`}
    >
      <canvas ref={canvas} width={THUMB} height={THUMB} />
      <small>{label}</small>
    </button>
  );
}

/**
 * Every attention map in the run at a glance: one row per attention layer,
 * one small picture per head, rows of each picture the words attending
 * (queries) and columns the words attended to (keys). Choosing a head draws
 * it large with its words, and its strongest links.
 */
export function AttentionPanel({
  run,
  onSelect,
}: {
  run: Run | null;
  onSelect?: (node: string) => void;
}) {
  const maps = useMemo(() => (run ? attentionMaps(run.trace) : []), [run]);
  const [values, setValues] = useState<{
    run: string;
    found: (Float64Array | null)[];
  } | null>(null);
  const [focus, setFocus] = useState<{ map: number; head: number }>({
    map: 0,
    head: 0,
  });
  useEffect(() => {
    if (!run || !maps.length) return;
    let live = true;
    Promise.all(maps.map((map) => valuesOf(run.id, map.tensor))).then(
      (found) => live && setValues({ run: run.id, found }),
    );
    return () => {
      live = false;
    };
  }, [run, maps]);
  if (!run)
    return (
      <p className="panel-empty">Run a model to see its attention maps.</p>
    );
  if (!maps.length)
    return (
      <p className="panel-empty">
        This run has no attention maps: softmax results named queries × keys, or
        shaped batch × heads × queries × keys.
      </p>
    );
  const loaded = values?.run === run.id ? values.found : null;
  if (!loaded)
    return <p className="panel-empty">Reading the attention maps…</p>;
  const rowsOf = (map: AttentionMap, at: number, head: number) =>
    loaded[at] ? headWeights(loaded[at]!, map, head) : null;
  const chosen = maps[focus.map] ? focus : { map: 0, head: 0 };
  const map = maps[chosen.map];
  const rows = rowsOf(map, chosen.map, chosen.head);
  const queryWords = lensWords(run, map.queries)?.tokens;
  const keyWords = lensWords(run, map.keys)?.tokens;
  const word = (words: string[] | undefined, at: number) =>
    words?.[at] ?? `${at}`;
  // The strongest link from each word, if it is not to itself.
  const links = rows
    ? rows
        .map((row, q) => {
          const k = row.indexOf(Math.max(...row));
          return { q, k, p: row[k] };
        })
        .filter(({ q, k }) => q !== k)
        .sort((a, b) => b.p - a.p)
        .slice(0, 4)
    : [];
  const cell = Math.max(
    8,
    Math.min(22, Math.floor(300 / Math.max(map.queries, map.keys))),
  );
  const labelled = map.keys <= 64;
  return (
    <div className="attention-panel">
      <div className="attention-grid">
        {maps.map((each, at) => (
          <div key={each.op.id} className="attention-row">
            <button
              type="button"
              className="attention-layer"
              onClick={() => onSelect?.(each.op.id)}
              title={`Open step ${each.op.index + 1}, which makes these weights`}
            >
              {each.label}
            </button>
            {Array.from({ length: each.heads }, (_, head) => {
              const thumb = rowsOf(each, at, head);
              return thumb ? (
                <HeadThumb
                  key={head}
                  rows={thumb}
                  label={`head ${head}`}
                  active={chosen.map === at && chosen.head === head}
                  onPick={() => setFocus({ map: at, head })}
                />
              ) : (
                <span key={head} className="attention-missing">
                  head {head}: no values
                </span>
              );
            })}
          </div>
        ))}
      </div>
      {rows && (
        <figure className="attention-focus">
          <figcaption>
            {map.label} · head {chosen.head}: rows attend to columns
            {links.length > 0 &&
              ` · strongest: ${links
                .map(
                  ({ q, k, p }) =>
                    `${word(queryWords, q)} → ${word(keyWords, k)} ${percent(p)}`,
                )
                .join(", ")}`}
          </figcaption>
          <svg
            // Room for the row labels, and for the last column's slanted one.
            width={(labelled ? 110 : 0) + map.keys * cell}
            height={(labelled ? 70 : 0) + map.queries * cell}
            role="img"
            aria-label={`${map.label} head ${chosen.head} attention`}
          >
            <g transform={labelled ? "translate(70 70)" : undefined}>
              {rows.map((row, q) =>
                row.map((value, k) => (
                  <rect
                    key={`${q}-${k}`}
                    x={k * cell}
                    y={q * cell}
                    width={cell - 1}
                    height={cell - 1}
                    fill={`rgba(127, 209, 199, ${Math.min(1, Math.max(0.04, value)).toFixed(3)})`}
                  >
                    <title>{`${word(queryWords, q)} → ${word(keyWords, k)}: ${percent(value)}`}</title>
                  </rect>
                )),
              )}
              {labelled &&
                Array.from({ length: map.queries }, (_, q) => (
                  <text
                    key={`q${q}`}
                    x={-6}
                    y={q * cell + cell / 2 + 3}
                    textAnchor="end"
                  >
                    {word(queryWords, q)}
                  </text>
                ))}
              {labelled &&
                Array.from({ length: map.keys }, (_, k) => (
                  <text
                    key={`k${k}`}
                    transform={`translate(${k * cell + cell / 2 + 3} -6) rotate(-60)`}
                  >
                    {word(keyWords, k)}
                  </text>
                ))}
            </g>
          </svg>
        </figure>
      )}
    </div>
  );
}
