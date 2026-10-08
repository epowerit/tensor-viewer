import { useEffect, useState } from "react";
import { api, isWhatIf, type CausalTrace, type Run } from "../api/client";
import {
  hoverPosition,
  useFocusedColumn,
  usePositionFocus,
} from "../tensors/positionFocus";
import { layerOfStep, layerStack } from "../tensors/layerStates";
import { tensorUses } from "../tensors/TensorUseContext";
import { lensWords } from "../tensors/lensWords";
import { ShadeScale } from "./ShadeScale";
import { useStepPreview } from "./stepPreview";
import { wordChanges } from "./wordChanges";
import { PanelLoading } from "./PanelLoading";
import { CopyTable } from "./CopyTable";

/** Traces already asked for, by run and the run compared with. */
const traces = new Map<string, Promise<CausalTrace>>();
/** The runs compared with, by id, to say which words differ. */
const others = new Map<string, Promise<Run>>();

const percent = (share: number) => `${Math.round(share * 100)}%`;

/**
 * Causal tracing: where in the model, and at which position, the difference
 * between two runs lives. Each layer's state (the state entering the block
 * stack and each block's result) is patched in from the other run at one
 * position at a time, and each cell is how much of the other run's result
 * comes back: 100% when that one patch restores it whole. The columns are the
 * sentence's words; a changed one is marked.
 */
export function CausalTracePanel({
  run,
  previousRunId = null,
  selected = null,
  onSelect,
  onPreview,
}: {
  run: Run | null;
  previousRunId?: string | null;
  /** The step playback is on, whose layer's row is marked. */
  selected?: string | null;
  /** Trace a layer's step on the canvas while its row is under the pointer. */
  onPreview?: (id: string | null) => void;
  onSelect?: (node: string) => void;
}) {
  const { rowPreview } = useStepPreview(onPreview);
  const focused = usePositionFocus();
  const table = useFocusedColumn<HTMLTableElement>(focused);
  const [shown, setShown] = useState<{
    key: string;
    found?: CausalTrace;
    error?: string;
  } | null>(null);
  // The run compared with, to say which words differ before tracing.
  const [other, setOther] = useState<Run | null>(null);
  useEffect(() => {
    if (!previousRunId) return;
    let live = true;
    let asked = others.get(previousRunId);
    if (!asked) {
      asked = api.getRun(previousRunId);
      asked.catch(() => others.delete(previousRunId));
      others.set(previousRunId, asked);
    }
    asked.then((found) => live && setOther(found)).catch(() => undefined);
    return () => {
      live = false;
    };
  }, [previousRunId]);
  if (!run)
    return (
      <p className="panel-empty">
        Run a model to trace where a change in its input matters.
      </p>
    );
  const whatIf = isWhatIf(run.id);
  if (!layerStack(run.trace).length)
    return (
      <p className="panel-empty">
        This model has no stack of repeated blocks (blocks.0, blocks.1, …), so
        there are no layers to trace one by one. The causal trace is for
        language models built that way, such as the library's GPT.
      </p>
    );
  if (!previousRunId)
    return (
      <p className="panel-empty">
        Causal tracing compares two runs that differ in an input. Change a word
        of the input as a what-if, or run again with another, then trace here.
      </p>
    );
  const key = `${run.id}|${previousRunId}`;
  const current = shown?.key === key ? shown : null;
  const ask = () => {
    let asked = traces.get(key);
    if (!asked) {
      asked = api.causalTrace(run.id, previousRunId);
      asked.catch(() => traces.delete(key));
      traces.set(key, asked);
    }
    setShown({ key });
    asked
      .then((found) =>
        setShown((now) => (now?.key === key ? { key, found } : now)),
      )
      .catch((error: Error) =>
        setShown((now) =>
          now?.key === key ? { key, error: error.message } : now,
        ),
      );
  };
  const against = whatIf ? "the recorded run" : "the run before";
  const changes =
    other && other.id === previousRunId ? wordChanges(run, other) : null;
  if (!current)
    return (
      <div className="lens-panel">
        {changes && (
          <p className="trace-changes">
            {changes.length ? (
              <>
                Words that differ from {against}:{" "}
                {changes.map(({ at, from, to }) => (
                  <span key={at} className="trace-change">
                    word {at + 1}: “{from}” → “{to}”
                  </span>
                ))}
              </>
            ) : (
              <>
                This run read the same words as {against}, so a trace finds a
                difference only if something else changed. To follow a word,
                change it as a what-if first.
              </>
            )}
          </p>
        )}
        <p className="lens-summary">
          The trace makes one run per layer and word, each with a single state
          copied in from {against}.
        </p>
        <button type="button" className="trace-ask" onClick={ask}>
          Trace against {against}
        </button>
      </div>
    );
  if (!current.found && !current.error)
    return <PanelLoading>Tracing: one run per layer and word…</PanelLoading>;
  if (current.error) return <p className="panel-empty">{current.error}</p>;
  const found = current.found!;
  const states = found.states ?? [];
  const recovery = found.recovery ?? [];
  const positions = found.positions ?? 0;
  const words = lensWords(run, positions);
  // The words that differ from the other run, or a what-if's edited cells
  // by position along the input's last axis.
  const changed = new Set(
    changes
      ? changes.map((change) => change.at)
      : (run.project.input.edits ?? []).map((edit) => edit.index % positions),
  );
  let best: { state: string; at: number; value: number } | null = null;
  recovery.forEach((row, layer) =>
    row.forEach((value, at) => {
      if (value !== null && (!best || value > best.value))
        best = { state: states[layer], at, value };
    }),
  );
  const top = best as { state: string; at: number; value: number } | null;
  const uses = tensorUses(run.trace);
  const here = layerOfStep(run.trace, selected);
  const madeAt = (name: string) => {
    const call = name.startsWith("before ")
      ? (run.trace.module_calls ?? []).find(
          (each) => each.path === name.slice("before ".length),
        )?.inputs[0]
      : (run.trace.module_calls ?? []).find((each) => each.path === name)
          ?.outputs[0];
    return call ? uses(call).made : null;
  };
  return (
    <div className="lens-panel">
      <p className="lens-summary">
        Each cell: how much of {against}'s result comes back.
        {top &&
          ` The most, ${percent(top.value)}, from ${top.state} at ${words ? `“${words.tokens[top.at]}”` : `position ${top.at}`}.`}
      </p>
      <p className="lens-key">
        <ShadeScale color="#ffb347" strength={0.7} />
        {changed.size > 0 && <span>≠ a word that differs</span>}
        <CopyTable
          what="how much each layer and word brings back"
          rows={() => [
            ["layer", "position", "word", "changed", "brought_back"],
            ...states.flatMap((state, layer) =>
              (recovery[layer] ?? []).map((value, at) => [
                state,
                at,
                words?.tokens[at] ?? "",
                changed.has(at) ? 1 : 0,
                value,
              ]),
            ),
          ]}
        />
      </p>
      <table
        ref={table}
        aria-label={`How much of ${against}'s result each layer's state at each word brings back`}
      >
        <thead>
          <tr>
            <th>Layer</th>
            {Array.from({ length: positions }, (_, at) => (
              <th
                key={at}
                className={
                  [
                    changed.has(at) ? "trace-changed" : "",
                    focused === at ? "is-focused" : "",
                  ]
                    .filter(Boolean)
                    .join(" ") || undefined
                }
                title={changed.has(at) ? "Changed in this run" : undefined}
                data-position={at}
                {...hoverPosition(at, words?.tokens[at])}
              >
                {words ? words.tokens[at] : at}
                {changed.has(at) ? " ≠" : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {states.map((state, layer) => {
            const made = madeAt(state);
            return (
              <tr
                key={state}
                className={state === here ? "is-current" : undefined}
                {...rowPreview(made?.id)}
              >
                <th>
                  <button
                    type="button"
                    disabled={!made}
                    onClick={() => made && onSelect?.(made.id)}
                    title={
                      made
                        ? `Open step ${made.step}, which makes it`
                        : undefined
                    }
                  >
                    {state}
                  </button>
                </th>
                {(recovery[layer] ?? []).map((value, at) => {
                  const share = Math.max(0, Math.min(1, value ?? 0));
                  return (
                    <td
                      key={at}
                      className={
                        [
                          share >= 0.5 ? "lens-agrees" : "",
                          focused === at ? "is-focused" : "",
                        ]
                          .filter(Boolean)
                          .join(" ") || undefined
                      }
                      {...hoverPosition(at, words?.tokens[at])}
                      style={{
                        background: `color-mix(in srgb, #ffb347 ${Math.round(share * 70)}%, transparent)`,
                      }}
                      title={`Patching ${state} at position ${at} brings back ${value === null ? "an unknown share" : percent(value)} of ${against}'s result`}
                    >
                      {value === null ? "—" : percent(value)}
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
