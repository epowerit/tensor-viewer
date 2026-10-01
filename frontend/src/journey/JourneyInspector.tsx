import { sourceCode, entryPath } from "../sources/files";
import { useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { Run } from "../api/client";
import type { JourneyNode } from "./graph";

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
  active: boolean;
  run: Run;
  node: JourneyNode;
  onSelect: (id: string) => void;
  onClose: () => void;
};
export function JourneyInspector({
  active,
  run,
  node,
  onSelect,
  onClose,
}: Props) {
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
  }, [activeLine, sourceFile, run.id]);

  return (
    <aside
      ref={panel}
      hidden={!active}
      inert={!active}
      tabIndex={-1}
      className="journey-inspector"
      aria-label="Executed code"
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          onClose();
        }
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
          <h2>Executed code</h2>
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
            aria-label="Close executed code"
            title="Close executed code"
          >
            <X size={18} />
          </button>
        </div>
      </header>
      <div className="inspector-tab-panel">
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
                  aria-current={index + 1 === activeLine ? "true" : undefined}
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
                  onClick={() => onSelect(op.id)}
                >
                  {op.index + 1} · {op.kind}
                </button>
              ))}
            </div>
          )}
          <h3>
            {operation?.kind ??
              (tensor?.role === "input" ? "Recorded input" : "Captured tensor")}
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
      </div>
    </aside>
  );
}
