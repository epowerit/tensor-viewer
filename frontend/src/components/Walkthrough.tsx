import { useEffect, useState } from "react";
import {
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  GitBranch,
  Layers3,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import type { Run } from "../api/client";
import { OperationView } from "../operations/OperationView";

export function Walkthrough({ run, busy }: { run: Run | null; busy: boolean }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [run?.id]);
  const operations = run?.trace.operations ?? [];
  const current = operations[step];
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () =>
        setStep((s) => {
          if (s >= operations.length - 1) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        }),
      2200,
    );
    return () => clearInterval(timer);
  }, [playing, operations.length]);
  useEffect(() => {
    document
      .querySelector(".step-button.selected")
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [step]);
  function jump(index: number) {
    setPlaying(false);
    setStep(index);
  }
  if (!run)
    return (
      <div className="empty-workspace">
        <div className="empty-cube">
          <Layers3 size={42} />
        </div>
        <span className="eyebrow">A closer look at tensors</span>
        <h2>
          {busy
            ? "Recording your forward pass…"
            : "See what happens between the lines."}
        </h2>
        <p>
          {busy
            ? "PyTorch is running in a separate local process. Your walkthrough will appear here."
            : "Configure your input, then run the example to reveal every tensor transformation."}
        </p>
      </div>
    );
  return (
    <>
      {run.trace.error && (
        <div className="run-error" role="alert">
          <CircleAlert size={19} />
          <div>
            <strong>
              {run.trace.error.type}
              {run.trace.error.line ? ` · line ${run.trace.error.line}` : ""}
            </strong>
            <p>{run.trace.error.message}</p>
            <small>
              {operations.length
                ? "Recorded steps remain available below."
                : "Edit the code or input settings, then run again."}
            </small>
          </div>
        </div>
      )}
      <div className="trace-summary">
        <span>
          <span className={`dot ${run.trace.error ? "dot-error" : ""}`} />
          {run.trace.error ? "Run stopped" : "Execution complete"}
        </span>
        <span>
          <Layers3 size={14} />
          {operations.length} operations
        </span>
        <span>
          <GitBranch size={14} />
          {Object.keys(run.trace.tensors).length} tensor states
        </span>
        <span className="trace-duration">
          {run.trace.duration_ms.toFixed(0)} ms
          <span className="muted"> · includes tracing</span>
        </span>
      </div>
      {current ? (
        <div className="walkthrough-layout">
          <aside className="steps-panel">
            <div className="steps-title">
              <span className="eyebrow">Forward pass</span>
              <span>{operations.length} steps</span>
            </div>
            <div className="steps-list">
              {operations.map((op, i) => (
                <button
                  key={op.id}
                  className={`step-button ${i === step ? "selected" : ""} ${op.status === "error" ? "step-error" : ""}`}
                  onClick={() => jump(i)}
                  aria-current={i === step ? "step" : undefined}
                >
                  <span className="step-number">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="step-description">
                    <b>{op.kind}</b>
                    <small>
                      {op.outputs[0]
                        ? `[${run.trace.tensors[op.outputs[0]].shape.join(", ")}]`
                        : "Execution error"}
                    </small>
                  </span>
                  {i === step && <ChevronRight size={14} />}
                </button>
              ))}
            </div>
            <div className="steps-end">
              <ArrowDown size={14} />
              {run.trace.output_ids[0]
                ? `Output [${run.trace.tensors[run.trace.output_ids[0]].shape.join(", ")}]`
                : "End of recorded path"}
            </div>
          </aside>
          <section className="step-workspace">
            <div className="playback">
              <span className="mono">
                STEP {String(step + 1).padStart(2, "0")}{" "}
                <span className="muted">
                  / {String(operations.length).padStart(2, "0")}
                </span>
              </span>
              <div className="playback-buttons">
                <button
                  title="Restart walkthrough"
                  aria-label="Restart walkthrough"
                  onClick={() => jump(0)}
                >
                  <RotateCcw size={15} />
                </button>
                <button
                  aria-label="Previous operation"
                  disabled={step === 0}
                  onClick={() => jump(step - 1)}
                >
                  <ChevronLeft size={17} />
                </button>
                <button
                  className="play-toggle"
                  aria-label={playing ? "Pause playback" : "Play walkthrough"}
                  onClick={() => {
                    if (step === operations.length - 1) setStep(0);
                    setPlaying(!playing);
                  }}
                >
                  {playing ? <Pause size={13} /> : <Play size={13} />}
                  <span>{playing ? "Pause" : "Play"}</span>
                </button>
                <button
                  aria-label="Next operation"
                  disabled={step === operations.length - 1}
                  onClick={() => jump(step + 1)}
                >
                  <ChevronRight size={17} />
                </button>
              </div>
            </div>
            <OperationView
              key={`${run.id}-${current.id}`}
              operation={current}
              run={run}
              onJump={jump}
            />
          </section>
        </div>
      ) : (
        !run.trace.error && (
          <div className="empty-workspace">
            <h2>No tensor operations were recorded.</h2>
            <p>
              The forward method may return its input unchanged. Try the
              Attention or Tensor basics template.
            </p>
          </div>
        )
      )}
      {run.trace.stdout && (
        <details className="stdout-panel">
          <summary>Program output</summary>
          <pre>{run.trace.stdout}</pre>
        </details>
      )}
    </>
  );
}
