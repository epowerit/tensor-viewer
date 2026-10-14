import { useEffect, useMemo } from "react";
import { CircleCheck, TriangleAlert } from "lucide-react";
import type { Dependencies, Run } from "../api/client";
import { lensWords } from "../tensors/lensWords";
import { CopyTable } from "./CopyTable";
import { PanelLoading } from "./PanelLoading";
import {
  allowed,
  askableAxes,
  askableResults,
  causalEvidence,
  defaultAsk,
  positionName,
  readList,
  violationsOf,
  type DependencyAsk,
} from "./dependencies";
import { askDependencies, useDependencyCheck } from "./dependencyStore";
import { useStepPreview } from "./stepPreview";

const percent = (share: number) =>
  share >= 0.995
    ? "100%"
    : share >= 0.01
      ? `${Math.round(share * 100)}%`
      : "<1%";
/** A share of the row's largest on a log scale from 10⁻⁶ to 1. */
const shade = (share: number) =>
  Math.max(0.12, Math.min(1, (Math.log10(share) + 6) / 6));
const PATTERN_NAME = {
  diagonal: "the diagonal",
  causal: "the lower triangle",
} as const;

/**
 * Who sees whom: for a result and one of its axes, which input positions
 * each of its positions reads, proved by one backward pass per position.
 * Rows are the result's positions, columns the input's; the cells the
 * expected pattern forbids are hatched, and any that are read anyway are
 * marked and become a run note on the first step that reads them.
 */
export function WhoSeesWhomPanel({
  run,
  selected = null,
  onSelect,
  onPreview,
}: {
  run: Run | null;
  /** The step playback is on, whose result can be asked about. */
  selected?: string | null;
  onSelect?: (node: string) => void;
  onPreview?: (id: string | null) => void;
}) {
  const { rowPreview } = useStepPreview(onPreview);
  const check = useDependencyCheck(run?.id);
  const trace = run?.trace ?? null;
  const evidence = useMemo(() => (trace ? causalEvidence(trace) : []), [trace]);
  const results = useMemo(
    () =>
      trace
        ? askableResults(trace).flatMap((tensor) => {
            const axes = askableAxes(trace, tensor, evidence);
            return axes.length ? [{ tensor, axes }] : [];
          })
        : [],
    [trace, evidence],
  );
  const first = useMemo(() => (trace ? defaultAsk(trace) : null), [trace]);
  // Opening the tab asks at once: about the latest result, on its axis with
  // the most to prove.
  useEffect(() => {
    if (run && !check && first) askDependencies(run.id, first);
  }, [run, check, first]);
  if (!run || !trace)
    return (
      <p className="panel-empty">
        Run a model to see which input positions each of its results reads.
      </p>
    );
  if (!results.length)
    return (
      <p className="panel-empty">
        No floating-point result here has an axis that lines up with an axis of
        the input, so there are no positions to compare.
      </p>
    );
  const ask = check?.ask ?? first!;
  const chosen = results.find(({ tensor }) => tensor.id === ask.tensor_id);
  const choose = (next: DependencyAsk | undefined) =>
    next && askDependencies(run.id, next);
  const step = selected
    ? trace.operations.find((op) => op.id === selected)
    : null;
  const here = step
    ? results.find(
        ({ tensor }) =>
          step.outputs.includes(tensor.id) && tensor.id !== ask.tensor_id,
      )
    : null;
  const input = trace.tensors[trace.input_ids[0]];
  return (
    <div className="lens-panel sees-panel">
      <div className="sees-controls">
        <label>
          Result
          <select
            value={ask.tensor_id}
            onChange={(event) => {
              const found = results.find(
                ({ tensor }) => tensor.id === event.target.value,
              );
              choose(
                found?.axes.find((each) => each.axis === ask.axis) ??
                  found?.axes[0],
              );
            }}
          >
            {results.map(({ tensor }) => {
              const made = trace.operations.find((op) =>
                op.outputs.includes(tensor.id),
              );
              return (
                <option key={tensor.id} value={tensor.id}>
                  {made ? `step ${made.index + 1} · ` : ""}
                  {tensor.name} [{tensor.shape.join(", ")}]
                </option>
              );
            })}
          </select>
        </label>
        <label>
          Axis
          <select
            value={ask.axis}
            onChange={(event) =>
              choose(
                chosen?.axes.find(
                  (each) => each.axis === Number(event.target.value),
                ),
              )
            }
          >
            {(chosen?.axes ?? []).map((each) => (
              <option key={each.axis} value={each.axis}>
                {chosen!.tensor.axes[each.axis]} (
                {chosen!.tensor.shape[each.axis]}) · reads input{" "}
                {input?.axes[each.input_axis ?? 0]}
              </option>
            ))}
          </select>
        </label>
        {here && (
          <button
            type="button"
            className="trace-ask"
            onClick={() => choose(here.axes[0])}
          >
            Ask about step {step!.index + 1}'s {here.tensor.name}
          </button>
        )}
      </div>
      <p className="lens-summary">
        {ask.expect
          ? `Expected: ${PATTERN_NAME[ask.expect]}, since ${ask.why}.`
          : "No pattern is expected along this axis: the map shows what each position reads."}
      </p>
      {check?.error ? (
        <p className="panel-empty">{check.error}</p>
      ) : !check?.answer ? (
        <PanelLoading>
          Differentiating each position of the result, one backward pass at a
          time…
        </PanelLoading>
      ) : (
        <Answer
          run={run}
          answer={check.answer}
          onSelect={onSelect}
          rowPreview={rowPreview}
        />
      )}
    </div>
  );
}

function Answer({
  run,
  answer,
  onSelect,
  rowPreview,
}: {
  run: Run;
  answer: Dependencies;
  onSelect?: (node: string) => void;
  rowPreview: ReturnType<typeof useStepPreview>["rowPreview"];
}) {
  const trace = run.trace;
  const tensor = trace.tensors[answer.tensor_id];
  const input = trace.tensors[answer.input_id ?? ""];
  const pattern = answer.expect ?? null;
  const reads = answer.reads ?? [];
  const strength = answer.strength ?? [];
  const zero = answer.zero ?? [];
  const columns = reads[0]?.length ?? 0;
  const violations = violationsOf(reads, pattern);
  const breaking = new Set(violations.map(([o, i]) => `${o},${i}`));
  const breach = answer.breach ?? null;
  const step = breach
    ? trace.operations.find((op) => op.id === breach.operation_id)
    : null;
  // A text input's words name its positions; a batch's examples have none.
  const words = pattern === "diagonal" ? null : lensWords(run, columns)?.tokens;
  const name = (at: number) => positionName(pattern, at, words);
  const cell = columns > 32 ? 7 : columns > 16 ? 11 : 16;
  const label = (at: number) =>
    columns <= 32 || at % 8 === 0 ? String(at) : "";
  const total = reads.reduce((sum, row) => sum + row.filter(Boolean).length, 0);
  return (
    <>
      <p className={`sees-verdict${violations.length ? " is-broken" : ""}`}>
        {pattern && !violations.length && (
          <>
            <CircleCheck size={13} aria-hidden="true" />
            Proved: each of the {reads.length}{" "}
            {pattern === "diagonal" ? "examples" : "positions"} reads only{" "}
            {pattern === "diagonal"
              ? "itself"
              : "itself and the positions before it"}
            .
          </>
        )}
        {violations.length > 0 && (
          <>
            <TriangleAlert size={13} aria-hidden="true" />
            {violations.length}{" "}
            {violations.length === 1 ? "cell breaks" : "cells break"} the
            pattern: {readList(violations)}.
            {step && (
              <button
                type="button"
                className="text-button"
                onClick={() => onSelect?.(step.id)}
                {...rowPreview(step.id)}
              >
                First at step {step.index + 1}, {step.kind}
                {step.source ? ` (line ${step.source.line})` : ""}
              </button>
            )}
          </>
        )}
        {!pattern &&
          `${total} of ${reads.length * columns} pairs read each other.`}
        {answer.size > reads.length &&
          ` The first ${reads.length} of ${answer.size} positions were checked.`}
      </p>
      <p className="lens-key">
        <span>
          rows: {tensor?.name}.{tensor?.axes[answer.axis]} · columns:{" "}
          {input?.name}.{input?.axes[answer.input_axis]}
        </span>
        <span>brighter reads more</span>
        {pattern && <span className="sees-key-outside">forbidden</span>}
        {violations.length > 0 && (
          <span className="sees-key-breach">read anyway</span>
        )}
        <CopyTable
          what="which input position each position reads"
          rows={() => [
            ["output", "input", "reads", "exact_zero", "strength", "allowed"],
            ...reads.flatMap((row, output) =>
              row.map((read, at) => [
                output,
                at,
                read ? 1 : 0,
                zero[output]?.[at] ? 1 : 0,
                strength[output]?.[at] ?? "",
                allowed(pattern, output, at) ? 1 : 0,
              ]),
            ),
          ]}
        />
      </p>
      <div
        className="sees-grid"
        role="grid"
        aria-label={`Which input position each position of ${tensor?.name} reads`}
        style={{
          gridTemplateColumns: `auto repeat(${columns}, ${cell}px)`,
        }}
      >
        <span className="sees-corner" aria-hidden="true">
          ↓ reads →
        </span>
        {Array.from({ length: columns }, (_, at) => (
          <span key={at} className="sees-head" title={name(at)}>
            {label(at)}
          </span>
        ))}
        {reads.map((row, output) => (
          <div key={output} role="row" className="sees-row">
            <span className="sees-head" role="rowheader" title={name(output)}>
              {words && columns <= 16 ? words[output] : output}
            </span>
            {row.map((read, at) => {
              const share = strength[output]?.[at];
              const exact = zero[output]?.[at];
              const outside = !allowed(pattern, output, at);
              const broken = breaking.has(`${output},${at}`);
              const first = breach?.output === output && breach?.input === at;
              return (
                <span
                  key={at}
                  role="gridcell"
                  className={
                    [
                      "sees-cell",
                      outside ? "is-outside" : "",
                      broken ? "is-broken" : "",
                      first ? "is-first" : "",
                      !read && !exact ? "is-faint" : "",
                    ]
                      .filter(Boolean)
                      .join(" ") || undefined
                  }
                  style={
                    read && share !== null && share !== undefined
                      ? { ["--share" as string]: shade(share) }
                      : undefined
                  }
                  title={`${name(output)} ${
                    exact
                      ? "does not read"
                      : read
                        ? share === null || share === undefined
                          ? "reads (a non-finite gradient)"
                          : `reads, ${percent(share)} of its largest`
                        : "barely reads (below the threshold)"
                  } ${name(at)}${broken ? ": against the pattern" : ""}`}
                />
              );
            })}
          </div>
        ))}
      </div>
    </>
  );
}
