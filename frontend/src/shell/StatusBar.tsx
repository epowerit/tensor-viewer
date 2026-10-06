import { CircleAlert, TriangleAlert } from "lucide-react";
import { useMemo } from "react";
import type { Draft, Run } from "../api/client";
import { loopFolds } from "../journey/loops";
import { kindName } from "../operations/kindName";

type Props = {
  draft: Draft | null;
  run: Run | null;
  busy: boolean;
  stale: boolean;
  errors: number;
  warnings: number;
  currentNode: string | null;
  /** A folded call or capsule playback is on, played as one step. */
  currentCard?: {
    title: string;
    operationIds: string[];
    lines: [number, number] | null;
  } | null;
  cell: number | null;
  cursor: { line: number; column: number } | null;
  /** Where a recorded line is in the code now, edited since the run. */
  placeLine?: (file: string | null, line: number) => number | null;
  /** Saving and updating after typing pauses, when turned on. */
  autoUpdate?: { onTurnOff: () => void } | null;
  onProblems: () => void;
  onCaptureMode: (mode: "values" | "shapes") => void;
  /** Shape checking for console and code projects; null where it does not apply. */
  check: {
    /** "recorded": the run on screen matches the code, so its shapes are current. */
    state: "none" | "recorded" | "checking" | "passed" | "partial" | "failed";
    live: boolean;
    onCheck: () => void;
    onLive: () => void;
  } | null;
};

/** One line that always says what is on screen: run state, selection, runtime. */
export function StatusBar({
  draft,
  run,
  busy,
  stale,
  errors,
  warnings,
  currentNode,
  currentCard,
  cursor,
  placeLine = (_, line) => line,
  autoUpdate = null,
  onProblems,
  onCaptureMode,
  check,
}: Props) {
  const mode = draft?.capture_mode ?? "values";
  const folds = useMemo(() => (run ? loopFolds(run.trace) : []), [run]);
  // The step on screen: what it does, what it wrote, and where.
  const step = currentNode
    ? run?.trace.operations.find((operation) => operation.id === currentNode)
    : undefined;
  const written = step && run!.trace.tensors[step.outputs[0]];
  // Lines name where the code is now; a step whose line was deleted has none.
  const stepLine =
    step?.source?.line != null
      ? placeLine(step.source.file ?? null, step.source.line)
      : null;
  const cardLines = currentCard?.lines
    ?.map((line) => placeLine(null, line))
    .filter((line): line is number => line !== null);
  return (
    <footer className="status-bar">
      {/* The run: how it went, and what needs a look. */}
      <div className="status-group">
        <span className={`status-run ${run?.trace.error ? "failed" : ""}`}>
          <span
            className={`status-dot ${run?.trace.error ? "error-dot" : ""}`}
          />
          {busy
            ? "Running…"
            : !run
              ? "Not run yet"
              : run.trace.error
                ? `Stopped · ${run.trace.error.type}`
                : `Recorded ${run.trace.operations.length} steps in ${run.trace.duration_ms.toFixed(0)} ms`}
        </span>
        {!busy && folds.length > 0 && (
          <span
            className="status-item status-loops"
            title={`${folds.map((fold) => `${fold.text} (line ${fold.line}): ${fold.iterations.length} identical passes`).join("; ")}. Each is drawn and played once, so playback has fewer steps than were recorded.`}
          >
            ↻ {folds.length === 1 ? "1 loop" : `${folds.length} loops`} drawn
            once
          </span>
        )}
        {stale && !busy && (
          <span className="status-stale">edited since run</span>
        )}
        {(errors > 0 || warnings > 0) && (
          <button
            className="status-item"
            onClick={onProblems}
            title="Show problems"
            aria-label={`${errors} errors, ${warnings} warnings`}
          >
            <CircleAlert size={12} /> {errors}
            <TriangleAlert size={12} /> {warnings}
          </button>
        )}
      </div>
      {/* The step playback is on, as the canvas and the panels show it. */}
      {(step || currentCard) && !busy && (
        <div className="status-group status-step">
          <span className="status-label">step</span>
          {step && (
            <span
              className="status-selection"
              title={step.source?.text ?? kindName(step.kind)}
            >
              {kindName(step.kind)}
              {written && written.name !== step.kind && ` → ${written.name}`}
              {written && ` [${written.shape.join(", ")}]`}
              {stepLine != null &&
                ` · ${step.source?.file ? `${step.source.file}:` : "line "}${stepLine}`}
            </span>
          )}
          {!step && currentCard && (
            <span
              className="status-selection"
              title={`${currentCard.title}, played as one step`}
            >
              {currentCard.title} · {currentCard.operationIds.length} steps
              {!!cardLines?.length &&
                ` · ${cardLines[0] === cardLines.at(-1) ? `line ${cardLines[0]}` : `lines ${cardLines[0]}–${cardLines.at(-1)}`}`}
            </span>
          )}
        </div>
      )}
      <span className="status-spacer" />
      {cursor && (
        <span>
          Ln {cursor.line}, Col {cursor.column}
        </span>
      )}
      {/* How the code is checked and run. */}
      <div className="status-group status-settings">
        {check && (
          <span className="status-check">
            <button
              className={`status-item check-${check.state}`}
              onClick={check.onCheck}
              disabled={
                busy || check.state === "checking" || check.state === "recorded"
              }
              title={
                check.state === "recorded"
                  ? "The code and inputs match the recorded run, so its shapes are current. Edit the code or inputs to check again without running."
                  : "Check shapes: dry-run the current code on shapes alone, without values and without saving a run (Ctrl/⌘ + Shift + Enter)"
              }
            >
              {check.state === "checking"
                ? "checking shapes…"
                : check.state === "recorded"
                  ? "shapes from run"
                  : check.state === "passed"
                    ? "shapes ✓"
                    : check.state === "partial"
                      ? "shapes ✓ until values are needed"
                      : check.state === "failed"
                        ? "shapes ✕"
                        : "check shapes"}
            </button>
            <button
              className="status-item"
              aria-pressed={check.live}
              onClick={check.onLive}
              title={
                check.live
                  ? "Live shape checking is on: your code is dry-run shortly after each edit. Click to turn off."
                  : "Turn on live shape checking. Your code will be executed on shapes alone shortly after each edit, so enable it only for code you trust to run as you type."
              }
            >
              live check {check.live ? "on" : "off"}
            </button>
          </span>
        )}
        {autoUpdate && (
          <button
            className="status-item status-auto-update"
            aria-pressed="true"
            onClick={autoUpdate.onTurnOff}
            title="The diagram updates shortly after you stop typing: each pause saves and runs your code. Click to turn off; Ctrl/⌘ + S still updates it."
          >
            updates as you type
          </button>
        )}
        {draft && (
          <button
            className="status-item"
            disabled={busy}
            title={
              mode === "values"
                ? "Recording values. Switch to shapes only for very large tensors."
                : "Recording shapes only: no numeric values are computed."
            }
            onClick={() =>
              onCaptureMode(mode === "values" ? "shapes" : "values")
            }
          >
            {mode === "values" ? "recording values" : "shapes only"}
          </button>
        )}
        <span title="Code runs locally on this computer">runs locally</span>
      </div>
    </footer>
  );
}
