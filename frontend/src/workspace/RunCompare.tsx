import { useEffect, useMemo, useRef } from "react";
import { X } from "lucide-react";
import type { Run } from "../api/client";
import { compareRuns, summarize, type StepInfo } from "./compare";

type Props = {
  current: Run;
  other: Run;
  onSelect: (operationId: string) => void;
  onClose: () => void;
};

const LABELS = {
  same: "same",
  values: "values differ",
  shape: "shape or type differs",
  outputs: "output count differs",
  operation: "different operation",
  status: "one failed",
  added: "only in the other run",
  removed: "only in this run",
} as const;

function Cell({ step }: { step: StepInfo | null }) {
  if (!step) return <td className="compare-missing">—</td>;
  const outputs = step.outputs.map((tensor) =>
    tensor
      ? `${tensor.name} [${tensor.shape.join(", ")}] · ${tensor.dtype}`
      : "tensor unavailable",
  );
  return (
    <td>
      <code>{step.kind}</code>{" "}
      <span title={outputs.join("; ")}>
        {step.failed
          ? "failed"
          : outputs.length
            ? `${outputs.slice(0, 3).join("; ")}${outputs.length > 3 ? `; +${outputs.length - 3} outputs` : ""}`
            : "no tensor"}
      </span>
    </td>
  );
}

/** Two saved runs of one project, aligned step by step. */
export function RunCompare({ current, other, onSelect, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const steps = useMemo(() => compareRuns(current, other), [current, other]);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  const when = (run: Run) => new Date(run.created_at).toLocaleString();
  return (
    <dialog
      ref={dialog}
      className="run-compare"
      aria-labelledby="run-compare-title"
      onCancel={onClose}
    >
      <header>
        <div>
          <span className="eyebrow">COMPARE RUNS</span>
          <h2 id="run-compare-title">{summarize(steps)}</h2>
        </div>
        <button
          className="icon-button"
          aria-label="Close comparison"
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </header>
      <div className="compare-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">Step</th>
              <th scope="col">Displayed run · {when(current)}</th>
              <th scope="col">Other run · {when(other)}</th>
              <th scope="col">Difference</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((step) => (
              <tr key={step.index} className={`compare-${step.change}`}>
                <th scope="row">
                  {step.left ? (
                    <button
                      title="Show this step of the displayed run"
                      onClick={() => {
                        onSelect(step.left!.id);
                        onClose();
                      }}
                    >
                      {step.index + 1}
                    </button>
                  ) : (
                    step.index + 1
                  )}
                </th>
                <Cell step={step.left} />
                <Cell step={step.right} />
                <td>
                  {step.change === "same" && step.valuesCompared === false
                    ? "values not compared"
                    : LABELS[step.change]}
                  {step.change === "values" &&
                    step.delta !== null &&
                    (Number.isFinite(step.delta)
                      ? ` · max |Δ| ${Number(step.delta.toPrecision(3))}`
                      : " · difference exceeds numeric range")}
                  {step.change === "values" &&
                    step.valuesCompared === false &&
                    " · some values unavailable"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="compare-note">
        Steps and all output tensors are aligned in execution order. Values are
        compared only when both runs recorded the full tensor inline. Paged and
        shape-only values are not compared.
      </p>
    </dialog>
  );
}
