import { useEffect, useRef } from "react";
import { ArrowLeft, Code2 } from "lucide-react";
import type { Run } from "../api/client";
import { OperationView } from "../operations/OperationView";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import type { JourneyNode } from "./graph";

type Props = {
  run: Run;
  node: JourneyNode;
  codeOpen: boolean;
  inspectorOpen: boolean;
  onSelect: (id: string) => void;
  onClose: () => void;
  onCode: (open: boolean) => void;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
};

/** A readable, unscaled view of a node, layered over its place in the journey. */
export function TransformationFocus({
  run,
  node,
  codeOpen,
  inspectorOpen,
  onSelect,
  onClose,
  onCode,
  showValues,
  onShowValues,
}: Props) {
  const panel = useRef<HTMLElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const operation = node.operation;
  const tensor = node.tensors[0];
  const source = operation?.source;

  useEffect(() => {
    body.current?.scrollTo(0, 0);
  }, [node.id]);
  useEffect(() => {
    if (!inspectorOpen) panel.current?.focus({ preventScroll: true });
  }, [node.id, inspectorOpen]);

  return (
    <section
      ref={panel}
      className={`transformation-focus category-node-${operation?.lesson.category ?? "input"}`}
      aria-label="Expanded transformation"
      tabIndex={-1}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
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
            <h2>{operation?.lesson.title ?? tensor?.name ?? "Tensor"}</h2>
            {operation && <span className="focus-kind">{operation.kind}</span>}
          </div>
          <span className="eyebrow">
            {operation
              ? `STEP ${operation.index + 1} / ${run.trace.operations.length}`
              : tensor?.role === "input"
                ? "INPUT TENSOR"
                : "CAPTURED TENSOR"}
          </span>
          <button
            className="focus-code"
            onClick={() => onCode(!codeOpen)}
            aria-label={codeOpen ? "Close code panel" : "Open code panel"}
            aria-pressed={codeOpen}
          >
            <Code2 size={15} /> Code
          </button>
        </div>
        <p>
          {operation?.lesson.summary ??
            (tensor?.role === "input"
              ? "The starting tensor passed to forward(x). Follow the next step to see what it becomes."
              : "This tensor was captured before its first recorded use.")}
        </p>
        {source && (
          <button
            className="focus-source"
            onClick={() => onCode(true)}
            title="Open this line in the recorded code"
          >
            <span>L{source.line}</span>
            <code>
              {run.project.code.split("\n")[source.line - 1]?.trim() ||
                operation.kind}
            </code>
          </button>
        )}
      </header>
      <div className="focus-content" ref={body}>
        {operation ? (
          <OperationView
            key={`${run.id}-${operation.id}`}
            run={run}
            operation={operation}
            showValues={showValues}
            onShowValues={onShowValues}
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
            <ValuesToggle
              checked={showValues}
              onChange={onShowValues}
              shapeOnly={run.project.capture_mode === "shapes"}
            />
            <TensorCard
              runId={run.id}
              tensor={tensor}
              label={tensor.role === "input" ? "Input" : "Captured tensor"}
              showValues={showValues}
              gridFrame={{
                rows: Math.max(1, Math.min(8, tensor.shape.at(-2) ?? 1)),
                columns: Math.max(1, Math.min(8, tensor.shape.at(-1) ?? 1)),
              }}
            />
          </div>
        ) : null}
      </div>
    </section>
  );
}
