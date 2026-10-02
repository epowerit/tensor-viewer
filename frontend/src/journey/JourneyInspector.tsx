import { sourceCode, entryPath } from "../sources/files";
import { useEffect, useRef, useId } from "react";
import { Braces, ChevronLeft, ChevronRight, Code2, X } from "lucide-react";
import type { Run } from "../api/client";
import { OperationView } from "../operations/OperationView";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import type { JourneyNode } from "./graph";
import { TensorShape } from "../tensors/InkShape";
import { kindName } from "../operations/kindName";

function PythonLine({ text }: { text: string }) {
  const tokens = text.split(
    /(#.*$|"[^"\n]*"|'[^'\n]*'|\b(?:def|class|return|import|from|as|for|in|if|else|elif|with|True|False|None|self|super)\b|\b\d+(?:\.\d+)?\b)/g,
  );
  return (
    <>
      {tokens.map((token, index) => {
        const kind = token.startsWith("#")
          ? "comment"
          : /^["']/.test(token)
            ? "string"
            : /^\d/.test(token)
              ? "number"
              : /^(def|class|return|import|from|as|for|in|if|else|elif|with|True|False|None|self|super)$/.test(
                    token,
                  )
                ? "keyword"
                : "plain";
        return (
          <span className={`syntax-${kind}`} key={index}>
            {token}
          </span>
        );
      })}
    </>
  );
}

type Props = {
  /** False keeps the panel mounted but hidden while another view is open. */
  active?: boolean;
  run: Run;
  node: JourneyNode;
  initialTensorId?: string;
  initialCell?: number;
  onSelect: (id: string) => void;
  onClose: () => void;
  tab: "code" | "values";
  onTab: (tab: "code" | "values") => void;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
};
export function JourneyInspector({
  active = true,
  run,
  node,
  initialTensorId,
  initialCell,
  onSelect,
  onClose,
  tab,
  onTab,
  showValues,
  onShowValues,
}: Props) {
  const tabId = useId();
  const panel = useRef<HTMLElement>(null);
  const code = useRef<HTMLDivElement>(null);
  const operation = node.operation;
  const position = operation ? run.trace.operations.indexOf(operation) : -1;
  const tensor = node.tensors[0];
  const activeLine = operation?.source?.line;
  const sourceFile = operation?.source?.file ?? entryPath(run.project);
  const sourceMap = new Map<number, typeof run.trace.operations>();
  run.trace.operations.forEach((op) => {
    if (op.source && (op.source.file ?? entryPath(run.project)) === sourceFile)
      sourceMap.set(op.source.line, [
        ...(sourceMap.get(op.source.line) ?? []),
        op,
      ]);
  });
  // The debugger view of a line: the last tensor it produced.
  const lineShape = (items: typeof run.trace.operations) => {
    if (items.some((op) => op.status === "error")) return "error";
    const id = items.at(-1)?.outputs[0];
    const tensor = id ? run.trace.tensors[id] : null;
    return tensor ? <TensorShape tensor={tensor} /> : null;
  };
  const sameLine = activeLine ? (sourceMap.get(activeLine) ?? []) : [];
  useEffect(() => {
    if (active) panel.current?.focus({ preventScroll: true });
  }, [active]);
  useEffect(() => {
    const container = code.current;
    const selected = container?.querySelector<HTMLElement>(".code-line-active");
    if (selected && container)
      container.scrollTo({
        top: Math.max(0, selected.offsetTop - container.clientHeight / 3),
      });
  }, [activeLine, sourceFile, tab, run.id]);

  return (
    <aside
      hidden={!active}
      inert={!active}
      ref={panel}
      tabIndex={-1}
      className="journey-inspector"
      aria-label="Transformation inspector"
      onKeyDown={(event) => {
        if (
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
      <header className="drawer-heading">
        <div>
          <span className="eyebrow">
            {operation
              ? `STEP ${operation.index + 1} / ${run.trace.operations.length}`
              : tensor?.role === "input"
                ? "INPUT TENSOR"
                : "CAPTURED TENSOR"}
          </span>
          <h2>
            {operation?.lesson.title ??
              (tensor?.role === "input"
                ? "Your input tensor"
                : "Captured tensor")}
          </h2>
        </div>
        <div className="inspector-heading-actions">
          <div className="drawer-step-navigation">
            <button
              className="icon-button"
              aria-label="Inspect previous operation"
              disabled={position <= 0}
              onClick={() => onSelect(run.trace.operations[position - 1].id)}
            >
              <ChevronLeft size={17} />
            </button>
            <button
              className="icon-button"
              aria-label="Inspect next operation"
              disabled={position >= run.trace.operations.length - 1}
              onClick={() => onSelect(run.trace.operations[position + 1].id)}
            >
              <ChevronRight size={17} />
            </button>
          </div>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close code panel"
            title="Close code panel"
          >
            <X size={18} />
          </button>
        </div>
      </header>
      <div
        className="inspector-tabs"
        role="tablist"
        aria-label="Inspector views"
        onKeyDown={(event) => {
          if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
            event.preventDefault();
            const next =
              event.key === "Home"
                ? "code"
                : event.key === "End"
                  ? "values"
                  : tab === "code"
                    ? "values"
                    : "code";
            onTab(next);
            event.currentTarget
              .querySelector<HTMLButtonElement>(`[data-view="${next}"]`)
              ?.focus();
          }
        }}
      >
        <button
          role="tab"
          id={`${tabId}-code`}
          data-view="code"
          aria-controls={`${tabId}-panel`}
          aria-selected={tab === "code"}
          tabIndex={tab === "code" ? 0 : -1}
          onClick={() => onTab("code")}
        >
          <Code2 size={15} /> Code
        </button>
        <button
          role="tab"
          id={`${tabId}-values`}
          data-view="values"
          aria-controls={`${tabId}-panel`}
          aria-selected={tab === "values"}
          tabIndex={tab === "values" ? 0 : -1}
          onClick={() => onTab("values")}
        >
          <Braces size={15} /> Tensor values
        </button>
      </div>
      <div
        className="inspector-tab-panel"
        role="tabpanel"
        id={`${tabId}-panel`}
        aria-labelledby={`${tabId}-${tab}`}
      >
        {tab === "code" ? (
          <>
            <div className="source-file">
              <span title={sourceFile}>{sourceFile}</span>
              <span>
                Recorded run <span className="status-dot" />
              </span>
            </div>
            <div
              className="source-code"
              ref={code}
              aria-label="Recorded Python source"
            >
              {sourceCode(run.project, sourceFile)
                .split("\n")
                .map((line, index) => {
                  const operations = sourceMap.get(index + 1) ?? [];
                  return (
                    <button
                      className={`code-line ${index + 1 === activeLine ? "code-line-active" : ""}`}
                      key={index}
                      disabled={!operations.length}
                      onClick={() => onSelect(operations[0].id)}
                      aria-label={`Line ${index + 1}${operations.length ? `: ${operations.map((op) => op.kind).join(", ")}` : ""}`}
                      aria-current={
                        index + 1 === activeLine ? "true" : undefined
                      }
                    >
                      <span className="code-line-number">{index + 1}</span>
                      <code>
                        <PythonLine text={line || " "} />
                      </code>
                      {operations.length > 1 && (
                        <span className="line-operation-count">
                          {operations.length}
                        </span>
                      )}
                      {lineShape(operations) && (
                        <span className="line-shape">
                          {lineShape(operations)}
                        </span>
                      )}
                    </button>
                  );
                })}
            </div>
            <div className="inspector-explanation">
              {sameLine.length > 1 && (
                <div className="line-operations">
                  <span>Operations on this line</span>
                  {sameLine.map((op) => (
                    <button
                      key={op.id}
                      className={op.id === operation?.id ? "active" : ""}
                      aria-current={
                        op.id === operation?.id ? "step" : undefined
                      }
                      onClick={() => onSelect(op.id)}
                    >
                      {op.index + 1} · {kindName(op.kind)}
                    </button>
                  ))}
                </div>
              )}
              <h3>
                {operation?.kind ??
                  (tensor?.role === "input"
                    ? "Recorded input"
                    : "Captured tensor")}
              </h3>
              <p>
                {operation?.error ??
                  operation?.lesson.detail ??
                  (tensor?.role === "input"
                    ? `Recorded input with shape [${tensor.shape.join(", ")}]. Follow the arrows to see where it goes.`
                    : "This snapshot has no earlier recorded producer. Follow its outgoing arrows to see how it is used.")}
              </p>
              <small>Select an executable line to find it on the canvas.</small>
            </div>
          </>
        ) : (
          <div className="inspector-values">
            {operation ? (
              <OperationView
                key={`${run.id}-${operation.id}-${initialTensorId ?? ""}-${initialCell ?? ""}`}
                operation={operation}
                initialTensorId={initialTensorId}
                initialCell={initialCell}
                run={run}
                onJump={(index) => {
                  const op = run.trace.operations.find(
                    (item) => item.index === index,
                  );
                  if (op) onSelect(op.id);
                }}
                showValues={showValues}
                onShowValues={onShowValues}
                compact
              />
            ) : (
              tensor && (
                <>
                  <p className="input-description">
                    {tensor.role === "input" ? (
                      <>
                        This input tensor is passed to <code>forward</code>.
                      </>
                    ) : (
                      "This tensor was captured before its first recorded use."
                    )}
                  </p>
                  <ValuesToggle
                    checked={showValues}
                    onChange={onShowValues}
                    shapeOnly={tensor.value_source === "shape"}
                  />
                  <TensorCard
                    light="active"
                    runId={run.id}
                    tensor={tensor}
                    label={
                      tensor.role === "input" ? "Input" : "Captured tensor"
                    }
                    showValues={showValues}
                    focusIndex={
                      !initialTensorId || initialTensorId === tensor.id
                        ? initialCell
                        : undefined
                    }
                    gridFrame={{
                      rows: Math.max(1, Math.min(8, tensor.shape.at(-2) ?? 1)),
                      columns: Math.max(
                        1,
                        Math.min(8, tensor.shape.at(-1) ?? 1),
                      ),
                    }}
                  />
                </>
              )
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
