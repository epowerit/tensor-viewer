import { useEffect, useMemo, useState } from "react";
import { api, type Run, type Tensor } from "../api/client";
import { formatValue } from "./coordinates";
import { asNumber } from "./margins";

/**
 * Every recorded state with this tensor's name and shape, in the order the
 * run wrote them: a residual stream updated layer by layer, say.
 */
export function namedStates(trace: Run["trace"], tensor: Tensor): Tensor[] {
  const shape = tensor.shape.join();
  const ids = [
    ...trace.input_ids,
    ...trace.operations.flatMap((op) => [
      ...op.outputs,
      ...(op.mutations ?? []).map((mutation) => mutation.after),
    ]),
  ];
  const seen = new Set<string>();
  return ids
    .filter((id) => !seen.has(id) && !!seen.add(id))
    .map((id) => trace.tensors[id] as Tensor | undefined)
    .filter(
      (state): state is Tensor =>
        !!state &&
        state.name === tensor.name &&
        state.shape.join() === shape &&
        state.value_source !== "shape",
    );
}

/** Up to this many states are traced. */
const STATES = 32;

/**
 * The chosen cell's value in each state of the same name, as a small line:
 * how one element of a residual stream moves through the layers. Choosing a
 * point opens the step that wrote that state.
 */
export function CellHistory({
  tensor,
  trace,
  runId,
  index,
  stepOf,
  onOpen,
}: {
  tensor: Tensor;
  trace: Run["trace"];
  runId?: string;
  index: number;
  stepOf: (tensorId: string) => { id: string; step: number } | null;
  onOpen: (stepId: string) => void;
}) {
  const states = useMemo(
    () => namedStates(trace, tensor).slice(0, STATES),
    [trace, tensor],
  );
  const [fetched, setFetched] = useState<{
    key: string;
    values: Record<string, number | string>;
  }>({ key: "", values: {} });
  const paged = states.filter((state) => state.value_source === "paged");
  const key = `${runId}/${index}/${paged.map((state) => state.id).join(",")}`;
  useEffect(() => {
    if (!runId || !paged.length) return;
    const controller = new AbortController();
    Promise.all(
      paged.map((state) =>
        api
          .tensorValues(runId, state.id, [index], controller.signal)
          .then((result) => [state.id, result.values[0]] as const),
      ),
    )
      .then((pairs) => {
        if (!controller.signal.aborted)
          setFetched({ key, values: Object.fromEntries(pairs) });
      })
      .catch(() => {});
    return () => controller.abort();
    // `key` covers the run, the cell and the paged states.
  }, [key]);
  if (states.length < 2) return null;
  const values = states.map((state) =>
    asNumber(
      state.value_source === "paged"
        ? fetched.key === key
          ? fetched.values[state.id]
          : undefined
        : state.values[index],
    ),
  );
  const finite = values.filter(Number.isFinite);
  const low = Math.min(...finite),
    high = Math.max(...finite);
  const width = 132,
    height = 18;
  const x = (i: number) => 4 + (i / (states.length - 1)) * (width - 8);
  const y = (value: number) =>
    high === low
      ? height / 2
      : 3 + (1 - (value - low) / (high - low)) * (height - 6);
  const shown = states.findIndex((state) => state.id === tensor.id);
  const path = values
    .map((value, i) =>
      Number.isFinite(value)
        ? `${x(i).toFixed(1)},${y(value).toFixed(1)}`
        : null,
    )
    .filter(Boolean)
    .join(" ");
  return (
    <div
      className="cell-history"
      aria-label={`This cell in each of the ${states.length} states of ${tensor.name}`}
    >
      <span className="cell-history-label">
        {tensor.name} · {states.length} states
      </span>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="group"
      >
        {finite.length > 1 && <polyline points={path} />}
        {values.map((value, i) => {
          const step = stepOf(states[i].id);
          return (
            <circle
              key={states[i].id}
              className={`${i === shown ? "cell-history-shown" : ""}${Number.isFinite(value) ? "" : " cell-history-broken"}`}
              cx={x(i)}
              cy={Number.isFinite(value) ? y(value) : height / 2}
              r={i === shown ? 3 : 2}
              role={step ? "button" : undefined}
              tabIndex={step ? 0 : undefined}
              aria-label={`State ${i + 1}${step ? `, step ${step.step}` : ", input"}: ${formatValue(value)}`}
              onClick={() => step && onOpen(step.id)}
              onKeyDown={(event) => {
                if (step && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onOpen(step.id);
                }
              }}
            >
              <title>{`State ${i + 1}${step ? ` · step ${step.step}` : " · input"}: ${formatValue(value)}`}</title>
            </circle>
          );
        })}
      </svg>
      {finite.length > 0 && (
        <span className="cell-history-range">
          {formatValue(low)} … {formatValue(high)}
        </span>
      )}
    </div>
  );
}
