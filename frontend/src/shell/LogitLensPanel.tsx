import { useEffect, useMemo, useState } from "react";
import { api, isWhatIf, type LogitLens, type Run } from "../api/client";
import { tokenize } from "../inputs/samples";
import {
  hoverPosition,
  useFocusedColumn,
  usePositionFocus,
} from "../tensors/positionFocus";
import { layerOfStep, layerStack } from "../tensors/layerStates";
import { tensorUses } from "../tensors/TensorUseContext";
import { ShadeScale } from "./ShadeScale";
import { useStepPreview } from "./stepPreview";
import { PanelLoading } from "./PanelLoading";
import { CopyTable } from "./CopyTable";

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
  selected = null,
  onSelect,
  onPreview,
}: {
  run: Run | null;
  /** The step playback is on, whose layer's row is marked. */
  selected?: string | null;
  onSelect?: (node: string) => void;
  /** Trace a layer's step on the canvas while its row is under the pointer. */
  onPreview?: (id: string | null) => void;
}) {
  const { rowPreview } = useStepPreview(onPreview);
  const [reading, setReading] = useState<{
    run: string;
    found?: LogitLens;
    error?: string;
  } | null>(null);
  // Only a model with a stack of repeated blocks has layers to read.
  const stacked = useMemo(
    () => (run ? layerStack(run.trace).length > 0 : false),
    [run],
  );
  const runId = stacked ? run?.id : undefined;
  const focused = usePositionFocus();
  const table = useFocusedColumn<HTMLTableElement>(focused);
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
  if (!stacked)
    return (
      <p className="panel-empty">
        This model has no stack of repeated blocks (blocks.0, blocks.1, …), so
        there are no layers to read one by one. The logit lens is for language
        models built that way, such as the library's GPT.
      </p>
    );
  const current = reading?.run === run.id ? reading : null;
  if (!current || (!current.found && !current.error))
    return (
      <PanelLoading>
        Reading every layer with the model's final layers
        {isWhatIf(run.id) ? " (a what-if run)" : ""}…
      </PanelLoading>
    );
  if (current.error) return <p className="panel-empty">{current.error}</p>;
  if (!states?.length)
    return <p className="panel-empty">No layer could be read in this run.</p>;
  const word = (id: number) => words?.vocabulary[id] ?? `#${id}`;
  // A predicted id with no word in the sentence, shown as #id, if any.
  const unknown = states
    .flatMap((state) => state.top ?? [])
    .map((top) => top[0]?.[0])
    .find((id) => id !== undefined && words?.vocabulary[id] === undefined);
  const final = states[states.length - 1].top ?? [];
  const uses = tensorUses(run.trace);
  const here = layerOfStep(run.trace, selected);
  // For each word, the first layer from which every later layer already
  // predicts what the model finally does, counted by layer.
  const settledAt = new Map<string, number>();
  final.forEach((top, at) => {
    const answer = top?.[0]?.[0];
    let from = states.length - 1;
    while (from > 0 && states[from - 1].top?.[at]?.[0]?.[0] === answer) from--;
    const name = states[from].name;
    settledAt.set(name, (settledAt.get(name) ?? 0) + 1);
  });
  const settled = states
    .filter((state) => settledAt.has(state.name))
    .map((state) => [state.name, settledAt.get(state.name)!] as const);
  return (
    <div className="lens-panel">
      {settled.length > 0 && (
        <p className="lens-summary">
          The final prediction is already there{" "}
          {settled
            .map(
              ([name, count], at) =>
                `${at === settled.length - 1 && at > 0 ? "and " : ""}from ${name} for ${count} ${count === 1 ? "word" : "words"}`,
            )
            .join(settled.length > 2 ? ", " : " ")}
          .
        </p>
      )}
      <p className="lens-key">
        <ShadeScale color="#7fd1c7" strength={0.42} />
        <span>how sure the layer is of its word</span>
        <CopyTable
          what="every layer's prediction at every word"
          rows={() => [
            ["layer", "position", "word", "predicted", "probability"],
            ...states.flatMap((state) =>
              (state.top ?? []).map((top, at) => [
                state.name,
                at,
                words?.tokens[at] ?? "",
                word(top[0]?.[0] ?? -1),
                top[0]?.[1] ?? null,
              ]),
            ),
          ]}
        />
        {unknown !== undefined && (
          <span>#{unknown}: a token id this sentence has no word for</span>
        )}
      </p>
      <table
        ref={table}
        aria-label="The word each layer would predict next, after each word of the input"
      >
        <thead>
          <tr>
            <th>Layer</th>
            {Array.from({ length: positions }, (_, at) => (
              <th
                key={at}
                title={`Predicted after position ${at}`}
                className={focused === at ? "is-focused" : undefined}
                data-position={at}
                {...hoverPosition(at, words?.tokens[at])}
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
              <tr
                key={state.name}
                className={state.name === here ? "is-current" : undefined}
                {...rowPreview(made?.id)}
                title={
                  state.name === here
                    ? "The layer of the step playback is on"
                    : undefined
                }
              >
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
                      {...hoverPosition(at, words?.tokens[at])}
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
