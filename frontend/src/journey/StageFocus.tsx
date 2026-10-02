import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import {
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  Repeat,
  UnfoldHorizontal,
} from "lucide-react";
import type { Run } from "../api/client";
import type { JourneyStage } from "./stages";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { spatialGrouping } from "../operations/spatialGrouping";
import { SpatialGroupingView } from "../operations/SpatialGroupingView";
import { windowScores } from "../operations/windowScores";
import { WindowScoresView } from "../operations/WindowScoresView";
import "./focusWorkspace.css";
import { TensorShape } from "../tensors/InkShape";
import { kindName } from "../operations/kindName";

type Props = {
  active: boolean;
  run: Run;
  stage: JourneyStage;
  initialTensorId?: string;
  connections: ReactNode;
  onClose: () => void;
  onExpand: () => void;
  onSelect: (id: string) => void;
  showValues: boolean;
  onShowValues: (value: boolean) => void;
  /** This call's pass in a folded loop, with stepping to the other passes. */
  pass?: {
    iteration: number;
    count: number;
    text: string;
    onStep: (delta: number) => void;
  };
};
export function StageFocus({
  active,
  run,
  stage,
  initialTensorId,
  connections,
  onClose,
  onExpand,
  onSelect,
  showValues,
  onShowValues,
  pass,
}: Props) {
  const [input, setInput] = useState(0),
    [output, setOutput] = useState(() =>
      Math.max(0, stage.outputs.indexOf(initialTensorId ?? "")),
    );
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    if (active) panel.current?.focus({ preventScroll: true });
  }, [active]);
  const operations = run.trace.operations.slice(
    stage.start_index,
    stage.end_index,
  );
  const grouping = spatialGrouping(run, stage);
  const scores = windowScores(run, stage);
  const tensors = [stage.inputs[input], stage.outputs[output]].map(
    (id) => run.trace.tensors[id],
  );
  const frame = {
    rows: Math.max(
      1,
      ...tensors.filter(Boolean).map((t) => Math.min(8, t.shape.at(-2) ?? 1)),
    ),
    columns: Math.max(
      1,
      ...tensors.filter(Boolean).map((t) => Math.min(8, t.shape.at(-1) ?? 1)),
    ),
  };
  return (
    <section
      className="transformation-focus tensor-focus stage-focus"
      hidden={!active}
      inert={!active}
      aria-label="Expanded stage"
      tabIndex={-1}
      ref={panel}
      onKeyDown={(e) => {
        if (active && e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
      }}
    >
      <header className="focus-heading">
        <div className="focus-toolbar">
          <button
            className="focus-back"
            aria-label="Back to journey"
            title="Back to journey"
            onClick={onClose}
          >
            <ArrowLeft size={15} />
          </button>
          <div className="focus-title">
            <h2>{stage.title}</h2>
          </div>
          {pass && (
            <span
              className="focus-loop focus-pass"
              title={`${pass.text} · pass ${pass.iteration} of ${pass.count}`}
            >
              <button
                aria-label="Same call, previous pass ([)"
                title="Same call, previous pass ([)"
                onClick={() => pass.onStep(-1)}
              >
                <ChevronLeft size={12} />
              </button>
              <Repeat size={12} aria-hidden="true" />
              {pass.iteration} of {pass.count}
              <button
                aria-label="Same call, next pass (])"
                title="Same call, next pass (])"
                onClick={() => pass.onStep(1)}
              >
                <ChevronRight size={12} />
              </button>
            </span>
          )}
          <span className="eyebrow">
            STEPS {stage.start_index + 1}–{stage.end_index}
          </span>
          <button className="secondary-button" onClick={onExpand}>
            <UnfoldHorizontal size={14} /> Expand stage
          </button>
        </div>
        <p className="focus-stage-path">
          <code>{stage.path}</code>
          {stage.failed && (
            <span className="focus-stage-failure">
              Execution stopped in this call
            </span>
          )}
        </p>
      </header>
      <div className="focus-content stage-focus-content">
        {connections}
        {scores ? (
          <WindowScoresView
            key={stage.id}
            lesson={scores}
            run={run}
            showValues={showValues}
            onShowValues={onShowValues}
            onStep={onSelect}
          />
        ) : grouping ? (
          <SpatialGroupingView
            key={stage.id}
            grouping={grouping}
            run={run}
            showValues={showValues}
            onShowValues={onShowValues}
            onStep={onSelect}
          />
        ) : (
          <>
            <div className="stage-boundary-tools">
              <ValuesToggle
                checked={showValues}
                onChange={onShowValues}
                shapeOnly={run.project.capture_mode === "shapes"}
              />
            </div>
            <div className="stage-boundaries">
              {(["Input", "Output"] as const).map((label, i) => {
                const ids = i === 0 ? stage.inputs : stage.outputs,
                  selection = i === 0 ? input : output,
                  setSelection = i === 0 ? setInput : setOutput;
                const tensor = run.trace.tensors[ids[selection]];
                return (
                  <div className="stage-boundary" key={label}>
                    {ids.length > 1 && (
                      <label className="stage-tensor-choice">
                        {label} tensor
                        <select
                          aria-label={`Stage ${label.toLowerCase()} tensor`}
                          value={selection}
                          onChange={(e) => setSelection(Number(e.target.value))}
                        >
                          {ids.map((id, index) => (
                            <option key={`${id}-${index}`} value={index}>
                              {index + 1} · {run.trace.tensors[id]?.name} [
                              {run.trace.tensors[id]?.shape.join(", ")}]
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {tensor ? (
                      <TensorCard
                        light="active"
                        key={tensor.id}
                        runId={run.id}
                        tensor={tensor}
                        label={`Stage ${label.toLowerCase()}`}
                        showValues={showValues}
                        gridFrame={frame}
                      />
                    ) : (
                      <div className="stage-no-tensor">
                        {stage.failed && i === 1
                          ? "This call did not return a tensor before execution stopped."
                          : `No tensor ${label.toLowerCase()} recorded.`}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
        <details className="stage-operation-details">
          <summary>
            <ChevronRight size={14} /> Inside this call{" "}
            <span>{operations.length} operations</span>
          </summary>
          <div className="stage-operation-list">
            {operations.map((op) => (
              <button key={op.id} onClick={() => onSelect(op.id)}>
                <span>{String(op.index + 1).padStart(2, "0")}</span>
                <b>{kindName(op.kind)}</b>
                <code>
                  {op.mutations?.length
                    ? `${op.mutations.length} in-place tensor ${op.mutations.length === 1 ? "state" : "states"}`
                    : op.outputs.length
                      ? op.outputs.map((id, i) => (
                          <Fragment key={`${id}-${i}`}>
                            {i > 0 && " · "}
                            <TensorShape tensor={run.trace.tensors[id]} />
                          </Fragment>
                        ))
                      : "No output"}
                </code>
                <ArrowRight size={14} />
              </button>
            ))}
          </div>
        </details>
      </div>
    </section>
  );
}
