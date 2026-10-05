import { useEffect, useMemo, useState } from "react";
import { api, isWhatIf, type LogitLens, type Run } from "../api/client";
import { tokenize } from "../inputs/samples";
import { hoverPosition, usePositionFocus } from "../tensors/positionFocus";
import { tensorUses } from "../tensors/TensorUseContext";

/** Readings already asked for, by run. */
const readings = new Map<string, Promise<LogitLens>>();

const percent = (p: number) =>
  p >= 0.995 ? "99%" : p >= 0.01 ? `${Math.round(p * 100)}%` : "<1%";

/**
 * The sentence the model's positions read: the text input whose length
 * matches them, with any what-if words set, and its vocabulary by id.
 */
export function lensWords(run: Run, positions: number) {
  const inputs = [
    { name: run.project.input_name ?? "x", input: run.project.input },
    ...(run.project.additional_inputs ?? []),
  ];
  for (const { input } of inputs) {
    if (!input.text) continue;
    const read = tokenize(input.text);
    if (read.tokens.length !== positions) continue;
    const tokens = [...read.tokens];
    for (const edit of input.edits ?? [])
      if (read.vocabulary[edit.value] !== undefined)
        tokens[edit.index % tokens.length] = read.vocabulary[edit.value];
    return { tokens, vocabulary: read.vocabulary };
  }
  return null;
}

/**
 * What each layer of a language model would predict: the state entering
 * the first block and each block's result, read by the model's own final
 * layers as if it were the last block's. A column is a position, headed by
 * the word the model has read there; a cell is the next word it would
 * predict from that layer, shaded by its probability. Cells that already
 * agree with the final prediction read brighter, so you can see the layer
 * where each prediction forms. A row's name opens the step that makes it.
 */
export function LogitLensPanel({
  run,
  onSelect,
}: {
  run: Run | null;
  onSelect?: (node: string) => void;
}) {
  const [reading, setReading] = useState<{
    run: string;
    found?: LogitLens;
    error?: string;
  } | null>(null);
  const runId = run?.id;
  const focused = usePositionFocus();
  useEffect(() => {
    if (!runId) return;
    let live = true;
    let asked = readings.get(runId);
    if (!asked) {
      asked = api.logitLens(runId);
      asked.catch(() => readings.delete(runId));
      readings.set(runId, asked);
    }
    setReading({ run: runId });
    asked
      .then((found) => live && setReading({ run: runId, found }))
      .catch(
        (error: Error) =>
          live && setReading({ run: runId, error: error.message }),
      );
    return () => {
      live = false;
    };
  }, [runId]);
  const states = reading?.run === runId ? reading?.found?.states : undefined;
  const positions = states?.[0]?.top?.length ?? 0;
  const words = useMemo(
    () => (run && positions ? lensWords(run, positions) : null),
    [run, positions],
  );
  if (!run)
    return (
      <p className="panel-empty">
        Run a language model to read what each of its layers would predict.
      </p>
    );
  const current = reading?.run === run.id ? reading : null;
  if (!current || (!current.found && !current.error))
    return (
      <p className="panel-empty">
        Reading every layer with the model's final layers
        {isWhatIf(run.id) ? " (a what-if run)" : ""}…
      </p>
    );
  if (current.error)
    return (
      <p className="panel-empty">
        {current.error} The logit lens reads models with a stack of repeated
        blocks (blocks.0, blocks.1, …) whose result the final layers turn into
        predictions.
      </p>
    );
  if (!states?.length)
    return <p className="panel-empty">No layer could be read in this run.</p>;
  const word = (id: number) => words?.vocabulary[id] ?? `#${id}`;
  const final = states[states.length - 1].top ?? [];
  const uses = tensorUses(run.trace);
  return (
    <div className="lens-panel">
      <p className="lens-summary">
        What each layer would predict next, read by the model's final layers as
        if it were the last block's result. Brighter cells already agree with
        the final prediction.
      </p>
      <table>
        <thead>
          <tr>
            <th>Layer</th>
            {Array.from({ length: positions }, (_, at) => (
              <th
                key={at}
                title={`Predicted after position ${at}`}
                className={focused === at ? "is-focused" : undefined}
                {...hoverPosition(at)}
              >
                {words ? `${words.tokens[at]} →` : `${at} →`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {states.map((state) => {
            const made = uses(state.tensor_id).made;
            return (
              <tr key={state.name}>
                <th>
                  <button
                    type="button"
                    disabled={!made}
                    onClick={() => made && onSelect?.(made.id)}
                    title={
                      made
                        ? `Open step ${made.step}, which makes this state`
                        : undefined
                    }
                  >
                    {state.name}
                  </button>
                </th>
                {(state.top ?? []).map((top, at) => {
                  const [id, p] = top[0];
                  const agrees = final[at]?.[0]?.[0] === id;
                  return (
                    <td
                      key={at}
                      className={
                        [
                          agrees ? "lens-agrees" : "",
                          focused === at ? "is-focused" : "",
                        ]
                          .filter(Boolean)
                          .join(" ") || undefined
                      }
                      {...hoverPosition(at)}
                      style={{
                        background: `color-mix(in srgb, #7fd1c7 ${Math.round(p * 42)}%, transparent)`,
                      }}
                      title={top
                        .map(([other, q]) => `${word(other)} ${percent(q)}`)
                        .join(" · ")}
                    >
                      {word(id)} <small>{percent(p)}</small>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
