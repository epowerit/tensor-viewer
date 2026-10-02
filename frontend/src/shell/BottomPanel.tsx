import { CircleAlert, Info, TriangleAlert, X } from "lucide-react";
import type { Run } from "../api/client";
import { VariablesPanel } from "../console/VariablesPanel";
import { ShapeDiagnosis } from "../operations/ShapeDiagnosis";
import { problemCounts, type Problem } from "./problems";

export type PanelTab = "problems" | "variables" | "output";
type Props = {
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  onClose: () => void;
  run: Run | null;
  problems: Problem[];
  selected: string | null;
  onSelect: (node: string) => void;
  onProblem: (problem: Problem) => void;
  /** Follow a tensor name across the canvas. */
  onThread?: (nodeIds: string[] | null) => void;
};

const ICONS = { error: CircleAlert, warning: TriangleAlert, info: Info };

/** Problems, variables, and program output for the displayed run. */
export function BottomPanel({
  tab,
  onTab,
  onClose,
  run,
  problems,
  selected,
  onSelect,
  onProblem,
  onThread,
}: Props) {
  const counts = problemCounts(problems);
  const tabs: [PanelTab, string, number | null][] = [
    ["variables", "Tensors", null],
    ["problems", "Run notes", counts.errors + counts.warnings || null],
    ["output", "Printed output", null],
  ];
  return (
    <section
      className="bottom-panel"
      tabIndex={-1}
      aria-label="Tensor shelf"
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
      <header className="ide-tabs panel-tabs" role="tablist">
        {tabs.map(([id, label, count]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            onClick={() => onTab(id)}
          >
            {label}
            {count !== null && (
              <span className={counts.errors ? "badge failed" : "badge"}>
                {count}
              </span>
            )}
          </button>
        ))}
        <button
          className="icon-button tab-action"
          aria-label="Close panel"
          title="Close panel (Escape)"
          onClick={onClose}
        >
          <X size={14} />
        </button>
      </header>
      <div className="panel-body" role="tabpanel">
        {tab === "problems" &&
          (problems.length ? (
            <ul className="problem-list">
              {problems.map((problem) => {
                const Icon = ICONS[problem.severity];
                return (
                  <li
                    key={problem.id}
                    className={`problem-${problem.severity}`}
                  >
                    <button
                      disabled={!problem.node && problem.line === null}
                      onClick={() => onProblem(problem)}
                    >
                      <Icon size={14} />
                      <b>{problem.title}</b>
                      {problem.line !== null && (
                        <span>
                          {problem.file} · line {problem.line}
                        </span>
                      )}
                    </button>
                    {problem.diagnosis ? (
                      <ShapeDiagnosis diagnosis={problem.diagnosis} bare />
                    ) : (
                      <p>{problem.detail}</p>
                    )}
                    {problem.diagnosis &&
                      problem.diagnosis.explanation !== problem.detail && (
                        <p className="problem-raw">{problem.detail}</p>
                      )}
                  </li>
                );
              })}
            </ul>
          ) : (
            <div className="panel-empty">
              {run ? (
                <>
                  <p>
                    No problems in the {run.trace.operations.length} recorded
                    steps.
                  </p>
                  <p className="panel-empty-detail">
                    Checked for NaN and infinity, values zeroed by an
                    activation, reductions over the batch or a size-1 axis,
                    products that sum unrelated axes, broadcasts that stretch
                    both operands, squeeze() dropping the batch axis, float64
                    promotion, writes into an input, values computed but never
                    used, growth across loop passes, and <code># shape:</code>{" "}
                    contracts.
                    {run.project.capture_mode === "shapes" &&
                      " Shape-only runs have no values, so the value checks were skipped."}
                  </p>
                </>
              ) : (
                <p>Problems found while running appear here.</p>
              )}
            </div>
          ))}
        {tab === "variables" && (
          <VariablesPanel
            run={run}
            selected={selected}
            onSelect={onSelect}
            onThread={onThread}
          />
        )}
        {tab === "output" &&
          (run ? (
            <div className="panel-output">
              {run.trace.stdout ? (
                <pre>{run.trace.stdout}</pre>
              ) : (
                <p className="panel-empty">
                  Nothing was printed. Use print() in your code to write here.
                </p>
              )}
              <p>
                {run.trace.operations.length} operations ·{" "}
                {Object.keys(run.trace.tensors).length} tensor states ·{" "}
                {run.trace.duration_ms.toFixed(0)} ms including tracing ·{" "}
                {new Date(run.created_at).toLocaleString()}
              </p>
            </div>
          ) : (
            <p className="panel-empty">Run to see printed output.</p>
          ))}
      </div>
    </section>
  );
}
