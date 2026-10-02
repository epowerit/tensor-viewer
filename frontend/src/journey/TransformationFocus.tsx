import { sourceCode, entryPath } from "../sources/files";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Code2,
  Info,
  Repeat,
} from "lucide-react";
import type { Run } from "../api/client";
import { OperationView } from "../operations/OperationView";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import type { JourneyNode } from "./graph";
import { FocusSlotContext } from "./FocusSlot";
import "./focusWorkspace.css";
import "./inputFocus.css";
import { kindName } from "../operations/kindName";

type Props = {
  active: boolean;
  run: Run;
  node: JourneyNode;
  connections: ReactNode;
  codeOpen: boolean;
  inspectorOpen: boolean;
  onSelect: (id: string) => void;
  onClose: () => void;
  onCode: (open: boolean) => void;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  initialTensorId?: string;
  initialCell?: number;
  onCell?: (index: number) => void;
  /** Step to the same operation in another pass of the folded loop it is in. */
  pass?: { loopId: string; onStep: (delta: number) => void };
};

/** A readable, unscaled view of a node, layered over its place in the journey. */
export function TransformationFocus({
  active,
  run,
  node,
  connections,
  codeOpen,
  inspectorOpen,
  onSelect,
  onClose,
  onCode,
  showValues,
  onShowValues,
  initialTensorId,
  initialCell,
  onCell,
  pass,
}: Props) {
  const panel = useRef<HTMLElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const [explaining, setExplaining] = useState(false);
  // Step controls portal into the header row instead of taking stage rows.
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  const explanationId = useId();
  const operation = node.operation;
  const tensor = node.tensors[0];
  const source = operation?.source;

  useEffect(() => {
    body.current?.scrollTo(0, 0);
    setExplaining(false);
  }, [node.id]);
  useEffect(() => {
    // A connection can replace the focused button while the code drawer is
    // open. Recover focus without stealing it from a still-mounted code line.
    if (active && (!inspectorOpen || document.activeElement === document.body))
      panel.current?.focus({ preventScroll: true });
  }, [node.id, inspectorOpen, active]);

  return (
    <section
      ref={panel}
      hidden={!active}
      inert={!active}
      className={`transformation-focus tensor-focus ${operation ? "" : "focus-tensor-only"} category-node-${operation?.lesson.category ?? "input"}`}
      aria-label="Expanded transformation"
      tabIndex={-1}
      onKeyDown={(event) => {
        if (
          !active ||
          event.key !== "Escape" ||
          event.defaultPrevented ||
          document.querySelector("dialog[open]")
        )
          return;
        event.preventDefault();
        event.stopPropagation();
        onClose();
      }}
    >
      <header className="focus-heading">
        <div className="focus-toolbar">
          <button
            className="focus-back"
            onClick={onClose}
            aria-label="Back to journey"
            title="Back to journey"
          >
            <ArrowLeft size={15} />
          </button>
          <div className="focus-title">
            <h2 title={operation?.lesson.title ?? tensor?.name ?? "Tensor"}>
              {operation?.lesson.title ?? tensor?.name ?? "Tensor"}
            </h2>
            {operation && (
              <span className="focus-kind">{kindName(operation.kind)}</span>
            )}
          </div>
          <div className="focus-slot" ref={setSlot}>
            {connections}
          </div>
          {operation?.loops?.map((step) => {
            // Each enclosing loop says which pass this step belongs to.
            const passes = Math.max(
              ...run.trace.operations.flatMap((op) =>
                (op.loops ?? [])
                  .filter((item) => item.id === step.id)
                  .map((item) => item.iteration),
              ),
            );
            const title = `Line ${step.line}: ${step.text} · iteration ${step.iteration} of ${passes}`;
            // In a folded loop the lesson flips between passes in place.
            return pass?.loopId === step.id ? (
              <span
                key={step.id}
                className="focus-loop focus-pass"
                title={title}
              >
                <button
                  aria-label="Same step, previous pass ([)"
                  title="Same step, previous pass ([)"
                  onClick={() => pass.onStep(-1)}
                >
                  <ChevronLeft size={12} />
                </button>
                <Repeat size={12} aria-hidden="true" />
                {step.iteration} of {passes}
                <button
                  aria-label="Same step, next pass (])"
                  title="Same step, next pass (])"
                  onClick={() => pass.onStep(1)}
                >
                  <ChevronRight size={12} />
                </button>
              </span>
            ) : (
              <span key={step.id} className="focus-loop" title={title}>
                <Repeat size={12} aria-hidden="true" />
                {step.iteration} of {passes}
              </span>
            );
          })}
          <span className="eyebrow">
            {operation
              ? `STEP ${operation.index + 1} / ${run.trace.operations.length}`
              : tensor?.role === "input"
                ? "INPUT TENSOR"
                : "CAPTURED TENSOR"}
          </span>
          <button
            className="focus-explain"
            aria-label="Explain this transformation"
            title="Explain this transformation"
            aria-expanded={explaining}
            aria-controls={explanationId}
            onClick={() => setExplaining((open) => !open)}
          >
            <Info size={15} />
          </button>
          <button
            className="focus-code"
            onClick={() => onCode(!codeOpen)}
            aria-label={codeOpen ? "Close executed code" : "Open executed code"}
            aria-pressed={codeOpen}
            title={
              source
                ? `Source: ${source.file ?? entryPath(run.project)}:${source.line}`
                : "Recorded code"
            }
          >
            <Code2 size={15} />{" "}
            <span className="slot-label">Executed code</span>
          </button>
        </div>
        <div
          className="focus-explanation"
          id={explanationId}
          hidden={!explaining}
        >
          <p>
            {operation?.lesson.summary ??
              (tensor?.role === "input"
                ? "One of the input tensors passed to forward. Follow its connected operations to see what it becomes."
                : "This tensor was captured before its first recorded use.")}
          </p>
          {source && (
            <div
              className="focus-source"
              title="Open Code to inspect this recorded line"
            >
              <span title={source.file ?? entryPath(run.project)}>
                {source.file ? `${source.file}:` : "L"}
                {source.line}
              </span>
              <code>
                {sourceCode(run.project, source.file)
                  .split("\n")
                  [source.line - 1]?.trim() || operation.kind}
              </code>
            </div>
          )}
        </div>
      </header>
      <div className="focus-content" ref={body}>
        <FocusSlotContext value={slot}>
          {operation ? (
            <OperationView
              key={`${run.id}-${operation.id}-${initialTensorId ?? ""}-${initialCell ?? ""}`}
              run={run}
              operation={operation}
              showValues={showValues}
              onShowValues={onShowValues}
              initialTensorId={initialTensorId}
              initialCell={initialCell}
              onCell={onCell}
              onJump={(index) => {
                const next = run.trace.operations.find(
                  (op) => op.index === index,
                );
                if (next) onSelect(next.id);
              }}
              compact
              expanded
            />
          ) : tensor ? (
            <div className="focus-input">
              <div className="focus-input-tools">
                <ValuesToggle
                  checked={showValues}
                  onChange={onShowValues}
                  shapeOnly={tensor.value_source === "shape"}
                />
              </div>
              <TensorCard
                light="active"
                runId={run.id}
                tensor={tensor}
                label={tensor.role === "input" ? "Input" : "Captured tensor"}
                showValues={showValues}
                focusIndex={
                  !initialTensorId || initialTensorId === tensor.id
                    ? initialCell
                    : undefined
                }
                onSelect={onCell}
                gridFrame={{
                  rows: Math.max(1, Math.min(8, tensor.shape.at(-2) ?? 1)),
                  columns: Math.max(1, Math.min(8, tensor.shape.at(-1) ?? 1)),
                }}
              />
            </div>
          ) : null}
        </FocusSlotContext>
      </div>
    </section>
  );
}
