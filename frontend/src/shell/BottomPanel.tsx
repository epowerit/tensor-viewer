import { useEffect, useRef, useState } from "react";
import {
  CircleAlert,
  Info,
  Maximize2,
  Minimize2,
  TriangleAlert,
  X,
} from "lucide-react";
import type { Run } from "../api/client";
import { VariablesPanel } from "../console/VariablesPanel";
import { ShapeDiagnosis } from "../operations/ShapeDiagnosis";
import { EdgeResizer } from "./EdgeResizer";
import { FlowTable } from "./FlowTable";
import { groupProblems, problemCounts, type Problem } from "./problems";

export type PanelTab = "problems" | "variables" | "flow" | "output";
type Props = {
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  onClose: () => void;
  run: Run | null;
  problems: Problem[];
  selected: string | null;
  /** A folded card's steps, played as one step. */
  selectedRange?: string[];
  /** Steps the last save added, edited, or reshaped, lit for a moment. */
  changed?: ReadonlySet<string> | null;
  onSelect: (node: string) => void;
  onProblem: (problem: Problem) => void;
  /** Follow a tensor name across the canvas. */
  onThread?: (nodeIds: string[] | null) => void;
  /** Trace a step on the canvas while its Flow row is under the pointer. */
  onPreview?: (id: string | null) => void;
  /** Steps under the pointer on the canvas. */
  hovered?: string[] | null;
  /** The run recorded before the displayed one. */
  previousRunId?: string | null;
  /** The height the person dragged the panel to, or null for the usual. */
  height?: number | null;
  onHeight?: (height: number | null) => void;
};

/** The shortest the panel gets, and the least of the canvas it leaves. */
const PANEL_MIN = 120;
const CANVAS_MIN = 160;

const ICONS = { error: CircleAlert, warning: TriangleAlert, info: Info };
const SEVERITY_NAMES = {
  error: ["error", "errors"],
  warning: ["warning", "warnings"],
  info: ["note", "notes"],
} as const;

/** Problems, variables, and program output for the displayed run. */
export function BottomPanel({
  tab,
  onTab,
  onClose,
  run,
  problems,
  selected,
  selectedRange,
  changed,
  onSelect,
  onProblem,
  onThread,
  onPreview,
  hovered = null,
  previousRunId = null,
  height = null,
  onHeight,
}: Props) {
  const counts = problemCounts(problems);
  // Kinds of run note the reader has hidden, and the kinds there are.
  const [hidden, setHidden] = useState<Set<Problem["severity"]>>(new Set());
  const severities = (["error", "warning", "info"] as const)
    .map(
      (severity) =>
        [
          severity,
          problems.filter((problem) => problem.severity === severity).length,
        ] as const,
    )
    .filter(([, count]) => count > 0);
  const section = useRef<HTMLElement>(null);
  // The panel's height as drawn, and the room its column has, so dragging
  // starts where the panel is and stops before the canvas gets too small.
  const [drawn, setDrawn] = useState({ height: 0, room: 0 });
  useEffect(() => {
    const element = section.current;
    const column = element?.parentElement;
    if (!element || !column) return;
    const measure = () =>
      setDrawn({ height: element.offsetHeight, room: column.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    observer.observe(column);
    return () => observer.disconnect();
  }, []);
  const tallest = Math.max(PANEL_MIN, drawn.room - CANVAS_MIN);
  const shown = Math.min(height ?? drawn.height, tallest);
  const maximized = drawn.room > 0 && shown >= tallest - 1;
  const before = useRef<number | null>(null);
  const tabs: [PanelTab, string, number | null][] = [
    ["variables", "Tensors", null],
    ["flow", "Flow", null],
    ["problems", "Run notes", counts.errors + counts.warnings || null],
    ["output", "Printed output", null],
  ];
  return (
    <>
      {onHeight && (
        <EdgeResizer
          className="panel-resizer"
          label="Resize the tensor shelf"
          grow="up"
          size={Math.round(shown)}
          min={PANEL_MIN}
          max={tallest}
          onSize={(next) =>
            onHeight(Math.max(PANEL_MIN, Math.min(tallest, next)))
          }
          onReset={() => onHeight(null)}
          onCollapse={onClose}
        />
      )}
      <section
        ref={section}
        className="bottom-panel"
        style={height ? { flexBasis: Math.min(height, tallest) } : undefined}
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
          {onHeight && (
            <button
              className="icon-button tab-action"
              aria-label={maximized ? "Restore panel size" : "Maximize panel"}
              aria-pressed={maximized}
              title={
                maximized
                  ? "Restore the panel's size"
                  : "Maximize the panel, leaving a strip of canvas"
              }
              onClick={() => {
                if (maximized) onHeight(before.current);
                else {
                  before.current = height;
                  onHeight(tallest);
                }
              }}
            >
              {maximized ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
            </button>
          )}
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
              <>
                {severities.length > 1 && (
                  <div
                    className="problem-filter"
                    role="group"
                    aria-label="Show run notes by kind"
                  >
                    {severities.map(([severity, count]) => {
                      const Icon = ICONS[severity];
                      return (
                        <button
                          key={severity}
                          type="button"
                          className={`problem-${severity}`}
                          aria-pressed={!hidden.has(severity)}
                          onClick={() => {
                            const next = new Set(hidden);
                            if (next.has(severity)) next.delete(severity);
                            else next.add(severity);
                            setHidden(next);
                          }}
                        >
                          <Icon size={12} /> {count}{" "}
                          {SEVERITY_NAMES[severity][count === 1 ? 0 : 1]}
                        </button>
                      );
                    })}
                  </div>
                )}
                <ul className="problem-list">
                  {groupProblems(
                    problems.filter((problem) => !hidden.has(problem.severity)),
                  ).map((item) => {
                    if ("members" in item) {
                      const Icon = ICONS[item.severity];
                      return (
                        <li
                          key={item.key}
                          className={`problem-${item.severity} problem-group`}
                        >
                          <div className="problem-group-heading">
                            <Icon size={14} />
                            <b>{item.title}</b>
                            <span>×{item.members.length}</span>
                          </div>
                          <div className="problem-group-members">
                            {item.members.map(({ problem, label }) => (
                              <button
                                key={problem.id}
                                type="button"
                                disabled={
                                  !problem.node && problem.line === null
                                }
                                title={`${problem.title}${problem.line !== null ? ` · ${problem.file} · line ${problem.line}` : ""}`}
                                onClick={() => onProblem(problem)}
                              >
                                <b>{label}</b>
                                {problem.line !== null && (
                                  <span>line {problem.line}</span>
                                )}
                              </button>
                            ))}
                          </div>
                          <p>{item.detail}</p>
                        </li>
                      );
                    }
                    const problem = item;
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
                        {problem.fix && (
                          <div className="problem-fix">
                            <button
                              type="button"
                              className="text-button"
                              onClick={problem.fix.apply}
                            >
                              {problem.fix.label}
                            </button>
                          </div>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </>
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
              range={selectedRange}
              changed={changed}
              onSelect={onSelect}
              onThread={onThread}
              previousRunId={previousRunId}
            />
          )}
          {tab === "flow" &&
            (run ? (
              <FlowTable
                run={run}
                selected={selected}
                range={selectedRange}
                changed={changed}
                onSelect={onSelect}
                onPreview={onPreview}
                hovered={hovered}
                previousRunId={previousRunId}
              />
            ) : (
              <p className="panel-empty">
                <span>The flow of tensors appears here after a run.</span>
              </p>
            ))}
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
    </>
  );
}
