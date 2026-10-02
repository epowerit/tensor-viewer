import { useEffect, useMemo, useRef, useState, Fragment } from "react";
import { X } from "lucide-react";
import type { Run } from "../api/client";
import { compareRuns, summarize, type StepInfo } from "./compare";
import { TensorShape } from "../tensors/InkShape";
import { kindName, ownName } from "../operations/kindName";
import { draftChanges } from "./runChanges";

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
      <code>{kindName(step.kind)}</code>{" "}
      <span title={outputs.join("; ")}>
        {step.failed
          ? "failed"
          : outputs.length
            ? step.outputs.slice(0, 3).map((tensor, i) => (
                <Fragment key={i}>
                  {i > 0 && "; "}
                  {tensor ? (
                    <>
                      {ownName(tensor.name, step.kind) && `${tensor.name} `}
                      <TensorShape tensor={tensor} /> · {tensor.dtype}
                    </>
                  ) : (
                    "tensor unavailable"
                  )}
                </Fragment>
              ))
            : "no tensor"}
        {!step.failed &&
          outputs.length > 3 &&
          `; +${outputs.length - 3} outputs`}
      </span>
    </td>
  );
}

/** Two saved runs of one project, aligned step by step. */
export function RunCompare({ current, other, onSelect, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const steps = useMemo(() => compareRuns(current, other), [current, other]);
  // What the later run changed, whichever of the two is displayed.
  const [earlier, later] =
    new Date(current.created_at) < new Date(other.created_at)
      ? [current, other]
      : [other, current];
  const changes = useMemo(
    () => draftChanges(earlier.project, later.project),
    [earlier, later],
  );
  const [showSame, setShowSame] = useState(false);
  const same = steps.filter((step) => step.change === "same").length;
  const shown = showSame
    ? steps
    : steps.filter((step) => step.change !== "same");
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
          <p className="compare-changes">
            {changes.length ? (
              <>
                The later run changed{" "}
                {changes.map((change, i) => (
                  <Fragment key={change}>
                    {i > 0 && ", "}
                    <code>{change}</code>
                  </Fragment>
                ))}
                .
              </>
            ) : (
              "Both runs used the same code and inputs."
            )}
          </p>
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
        {shown.length > 0 && (
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
              {shown.map((step) => (
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
                      step.shift &&
                      ` · distribution differs, mean ${step.shift.before.toPrecision(3)} → ${step.shift.after.toPrecision(3)}`}
                    {step.change === "values" &&
                      step.valuesCompared === false &&
                      !step.shift &&
                      " · some values unavailable"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {same > 0 && (
          <button
            className="compare-toggle"
            aria-expanded={showSame}
            onClick={() => setShowSame(!showSame)}
          >
            {showSame
              ? `Hide the ${same} matching steps`
              : `Show the ${same} matching ${same === 1 ? "step" : "steps"}`}
          </button>
        )}
      </div>
      <p className="compare-note">
        Steps and all output tensors are aligned in execution order. Values are
        compared only when both runs recorded the full tensor inline. Paged and
        shape-only values are not compared, but a large tensor whose recorded
        distribution differs is reported as changed, with its mean shift.
      </p>
    </dialog>
  );
}
