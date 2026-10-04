import { useContext, useEffect, useState } from "react";
import { api, type Tensor } from "../api/client";
import {
  nonFiniteText,
  traceNonFinite,
  type NanCell,
  type NanTrace,
} from "../journey/nanTrace";
import { formatValue, unravel } from "./coordinates";
import { asNumber } from "./margins";
import { TensorUseContext } from "./TensorUseContext";

/** Cells shown at each end of a long trail; the middle folds to a count. */
const ENDS = 3;

/**
 * Where a NaN or an infinity came from: chosen on a cell that is not finite,
 * the trail of cells it was computed from, step by step back to the step
 * that made it out of finite values, and why (log(0), 0 / 0, an overflowing
 * exp). Each cell on the trail opens its step with the cell selected.
 */
export function NanTrail({
  tensor,
  index,
  value,
}: {
  tensor: Tensor;
  index: number;
  value: number | string | boolean | undefined;
}) {
  const flow = useContext(TensorUseContext);
  const number = asNumber(value);
  const broken = value !== undefined && !Number.isFinite(number);
  const key = `${flow?.runId}/${tensor.id}/${index}`;
  const [found, setFound] = useState<{ key: string; trail: NanTrace } | null>(
    null,
  );
  useEffect(() => {
    if (!broken || !flow?.trace) return;
    const trace = flow.trace;
    const controller = new AbortController();
    traceNonFinite(trace, tensor.id, index, async (id, indices) => {
      const state = trace.tensors[id];
      if (!state) return null;
      if (state.values?.length === state.numel)
        return indices.map((at) => state.values[at]);
      if (!flow.runId || state.value_source !== "paged") return null;
      return api
        .tensorValues(flow.runId, id, indices, controller.signal)
        .then((result) => result.values)
        .catch(() => null);
    }).then((trail) => {
      if (!controller.signal.aborted) setFound({ key, trail });
    });
    return () => controller.abort();
    // `key` names the run, the tensor and the cell.
  }, [key, broken]);
  if (!broken || !flow) return null;
  const trail = found?.key === key ? found.trail : null;
  const name = (cell: { tensorId: string; index: number }) => {
    const state = flow.trace?.tensors[cell.tensorId];
    return `${state?.name ?? "tensor"}[${unravel(cell.index, state?.shape ?? []).join(", ")}]`;
  };
  const valueText = (cell: { tensorId: string; value: number }) =>
    !Number.isFinite(cell.value)
      ? nonFiniteText(cell.value)
      : flow.trace?.tensors[cell.tensorId]?.dtype === "bool"
        ? cell.value
          ? "True"
          : "False"
        : formatValue(cell.value);
  const stepOf = (cell: NanCell) => flow.uses(cell.tensorId).made;
  const open = (cell: NanCell) =>
    cell.opId && flow.go(cell.opId, cell.tensorId, cell.index);
  const chip = (cell: NanCell) => {
    const step = stepOf(cell);
    return (
      <button
        type="button"
        className="nan-trail-cell"
        disabled={!cell.opId}
        onClick={() => open(cell)}
        title={
          step
            ? `Step ${step.step} · ${step.kind}: open it with this cell selected`
            : "An input or a parameter"
        }
      >
        {name(cell)} <b>{nonFiniteText(cell.value)}</b>
      </button>
    );
  };
  if (!trail)
    return (
      <div className="nan-trail" aria-live="polite">
        <span className="nan-trail-label">
          Tracing {nonFiniteText(number)} back…
        </span>
      </div>
    );
  const { cells } = trail;
  const origin = cells.at(-1);
  // The chosen cell is the first; the rest lead back to the origin.
  const shown =
    cells.length > ENDS * 2 + 1
      ? [
          ...cells.slice(0, ENDS),
          cells.length - ENDS * 2,
          ...cells.slice(-ENDS),
        ]
      : cells;
  const originStep = origin ? stepOf(origin) : null;
  return (
    <div
      className="nan-trail"
      aria-label={`Where this ${nonFiniteText(number)} came from`}
    >
      {cells.length > 1 && (
        <ol className="nan-trail-cells">
          {shown.map((cell, i) => (
            <li key={typeof cell === "number" ? "folded" : i}>
              {typeof cell === "number" ? (
                <span className="nan-trail-folded">{cell} more</span>
              ) : (
                chip(cell)
              )}
            </li>
          ))}
        </ol>
      )}
      {origin && trail.cause && (
        <p className="nan-trail-cause">
          <span className="nan-trail-label">
            {originStep
              ? `Born at step ${originStep.step} · ${originStep.kind}`
              : "Already in the input"}
          </span>{" "}
          {trail.cause}
          {trail.from.length > 0 &&
            trail.from.length <= 4 &&
            ` (${trail.from
              .map((each) => `${name(each)} = ${valueText(each)}`)
              .join(", ")})`}
          .
        </p>
      )}
      {trail.infinity && (
        <button
          type="button"
          className="nan-trail-more"
          disabled={!trail.infinity.opId}
          onClick={() => trail.infinity && open(trail.infinity)}
        >
          Trace the {nonFiniteText(trail.infinity.value)} it came from:{" "}
          {name(trail.infinity)}
        </button>
      )}
      {trail.approximate && (
        <small className="nan-trail-note">
          Part of the trail passes a step whose cell relation is not recorded;
          there it follows the first {nonFiniteText(number)} of the step's
          input.
        </small>
      )}
      {trail.stopped && (
        <small className="nan-trail-note">{trail.stopped}</small>
      )}
    </div>
  );
}
