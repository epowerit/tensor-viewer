import {
  Check,
  CircleAlert,
  Eraser,
  GitCompareArrows,
  History,
  Trash2,
} from "lucide-react";
import type { Draft, Run, RunSummary } from "../api/client";
import { draftChanges } from "../workspace/runChanges";
import { since } from "./since";
import "./collections.css";

type Props = {
  history: RunSummary[];
  run: Run | null;
  busy: boolean;
  onOpen: (id: string) => void;
  onCompare: (id: string) => void;
  /** Delete one run, with Undo. */
  onDelete?: (id: string) => void;
  /** Keep the newest run and clear the ones before it, with Undo. */
  onClearOlder?: () => void;
};

/** Saved runs of the open project: reopen one, or compare it with the one on screen. */
export function RunsView({
  history,
  run,
  busy,
  onOpen,
  onCompare,
  onDelete,
  onClearOlder,
}: Props) {
  if (!history.length)
    return (
      <div className="explorer collection-explorer runs-view">
        <div className="collection-empty">
          <History size={22} aria-hidden="true" />
          <b>A place for every run</b>
          <p>Run your model to save its tensor journey here.</p>
        </div>
      </div>
    );
  return (
    <div className="explorer collection-explorer runs-view">
      {onClearOlder && history.length > 1 && (
        <div className="runs-tools">
          <span>
            {history.length >= 20 ? "The latest 20" : history.length} runs
          </span>
          <button
            className="text-button"
            disabled={busy}
            title="Keep the newest run and delete the ones before it, with Undo"
            onClick={onClearOlder}
          >
            <Eraser size={12} aria-hidden="true" /> Keep latest only
          </button>
        </div>
      )}
      <ul className="explorer-list">
        {history.map((item, index) => {
          const current = item.id === run?.id;
          // History is newest first: the run before this one is the next row.
          const previous = history[index + 1]?.project;
          const changes =
            item.project && previous
              ? draftChanges(previous as Draft, item.project as Draft)
              : null;
          // History lists the latest 20 runs; with fewer, the last is the first.
          const first = index === history.length - 1 && history.length < 20;
          const created = new Date(item.created_at);
          const validDate = !Number.isNaN(created.getTime());
          const time = validDate
            ? created.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })
            : "Saved run";
          const date = validDate
            ? created.toLocaleDateString([], {
                month: "short",
                day: "numeric",
                year: "numeric",
              })
            : "Date unavailable";
          return (
            <li key={item.id}>
              <button
                className={`${current ? "current" : ""} ${item.failed ? "failed" : ""}`}
                aria-current={current ? "true" : undefined}
                disabled={busy}
                title={`${date}, ${time} · Show original code and inputs`}
                onClick={() => onOpen(item.id)}
              >
                {item.failed ? <CircleAlert size={14} /> : <Check size={14} />}
                <span className="run-date-label">
                  <span>
                    <time
                      dateTime={validDate ? created.toISOString() : undefined}
                    >
                      {time}
                    </time>
                    {index === 0 && <b className="run-latest">Latest</b>}
                  </span>
                  <small>{validDate ? since(created) : date}</small>
                  {changes && (
                    <small
                      className="run-changes"
                      title={
                        changes.length
                          ? `Changed since the run before: ${changes.join(", ")}`
                          : "Same code and inputs as the run before"
                      }
                    >
                      {changes.length ? changes.join(" · ") : "re-run"}
                    </small>
                  )}
                  {first && <small className="run-changes">first run</small>}
                </span>
                <span className="run-step-count">
                  {item.operation_count} <small>steps</small>
                  {item.failed && <small>Stopped</small>}
                </span>
              </button>
              {run && !current && (
                <button
                  className="icon-button explorer-remove"
                  aria-label="Compare with the displayed run"
                  title={`Compare ${date}, ${time} with the displayed run`}
                  disabled={busy}
                  onClick={() => onCompare(item.id)}
                >
                  <GitCompareArrows size={13} />
                </button>
              )}
              {onDelete && (
                <button
                  className="icon-button explorer-remove"
                  aria-label={`Delete the run of ${date}, ${time}`}
                  title="Delete this run, with Undo"
                  disabled={busy}
                  onClick={() => onDelete(item.id)}
                >
                  <Trash2 size={13} />
                </button>
              )}
            </li>
          );
        })}
      </ul>
      {run && (
        <details className="run-configuration">
          <summary>Run configuration</summary>
          <pre>
            {JSON.stringify(
              {
                class_name: run.project.class_name,
                constructor: run.project.constructor,
                input: run.project.input,
                input_name: run.project.input_name ?? "x",
                input_binding: run.project.input_binding ?? "positional",
                additional_inputs: run.project.additional_inputs ?? [],
                weights: run.project.weights ?? null,
                entry_path: run.project.entry_path ?? "model.py",
                import_root: run.project.import_root ?? ".",
                source_files: Object.keys(run.project.files ?? {}),
                repository: run.project.repository ?? null,
                environment: run.project.environment ?? "TensorViewer",
                runtime: run.trace.runtime ?? {},
              },
              null,
              2,
            )}
          </pre>
        </details>
      )}
    </div>
  );
}
