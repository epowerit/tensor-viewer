import { Fragment, lazy, Suspense, useEffect, useRef, useState } from "react";
import { PanelLoading } from "./PanelLoading";
import { lockPosition, useLockedPosition } from "../tensors/positionFocus";
import { PANEL_GROUPS, PANEL_TABS, panelTab, type PanelTab } from "./panelTabs";
import {
  CircleAlert,
  CircleCheck,
  CircleHelp,
  LoaderCircle,
  Play,
  Columns2,
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
import { groupProblems, problemCounts, type Problem } from "./problems";

// The tabs beyond the tensor shelf and run notes load apart, so the workbench
// starts without them; once the browser is idle they are fetched ahead, so
// opening one does not wait.
const TAB_MODULES = {
  flow: () => import("./FlowTable"),
  watch: () => import("./WatchPanel"),
  weights: () => import("./WeightsPanel"),
  lens: () => import("./LogitLensPanel"),
  trace: () => import("./CausalTracePanel"),
  map: () => import("./MapPanel"),
  attention: () => import("./AttentionPanel"),
};
const FlowTable = lazy(() =>
  TAB_MODULES.flow().then((module) => ({ default: module.FlowTable })),
);
const WatchPanel = lazy(() =>
  TAB_MODULES.watch().then((module) => ({ default: module.WatchPanel })),
);
const WeightsPanel = lazy(() =>
  TAB_MODULES.weights().then((module) => ({ default: module.WeightsPanel })),
);
const LogitLensPanel = lazy(() =>
  TAB_MODULES.lens().then((module) => ({ default: module.LogitLensPanel })),
);
const CausalTracePanel = lazy(() =>
  TAB_MODULES.trace().then((module) => ({
    default: module.CausalTracePanel,
  })),
);
const MapPanel = lazy(() =>
  TAB_MODULES.map().then((module) => ({ default: module.MapPanel })),
);
const AttentionPanel = lazy(() =>
  TAB_MODULES.attention().then((module) => ({
    default: module.AttentionPanel,
  })),
);
let prefetched = false;
function prefetchTabs() {
  if (prefetched || typeof window === "undefined") return;
  prefetched = true;
  const load = () =>
    Object.values(TAB_MODULES).forEach((module) =>
      module().catch(() => undefined),
    );
  if ("requestIdleCallback" in window) window.requestIdleCallback(load);
  else setTimeout(load, 1500);
}

/** What the run notes check every run for, by kind. */
const RUN_CHECKS: [string, string[]][] = [
  [
    "Numbers",
    [
      "NaN and infinity",
      "exp near overflow",
      "log near 0",
      "cancelling subtractions",
      "tiny divisors",
      "saturated softmax",
      "float16 range",
      "float64 promotion",
    ],
  ],
  [
    "Activations",
    ["features an activation leaves dead", "values zeroed by an activation"],
  ],
  [
    "Shapes",
    [
      "reductions over the batch or a size-1 axis",
      "products that sum unrelated axes",
      "broadcasts that stretch both operands",
      "squeeze() dropping the batch axis",
      "# shape: contracts",
    ],
  ],
  [
    "Data flow",
    [
      "writes into an input",
      "values computed but never used",
      "growth across loop passes",
    ],
  ],
];

/** The title bar's next action, offered where a tab has nothing to show. */
type NextStep = {
  label: string;
  detail: string;
  /** The action runs the code, rather than asking for something first. */
  run: boolean;
  busy: boolean;
  disabled: boolean;
  onClick: () => void;
};

type Props = {
  nextStep?: NextStep | null;
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
  /** Whose watch expressions the Watch tab keeps. */
  projectId?: string | null;
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

/**
 * How a tab opens, as the canvas opens a step: a plain title, one sentence
 * on what it shows, and short hints on how to read it.
 */
function PanelIntro({ which }: { which: PanelTab }) {
  const { intro } = panelTab(which);
  return (
    <div className="panel-intro">
      <p>
        <strong>{intro.title}</strong> {intro.text}
      </p>
      {intro.read && (
        <p className="panel-intro-read">
          {intro.read.map((hint, at) => (
            <Fragment key={hint}>
              {at > 0 && <span aria-hidden="true"> · </span>}
              {hint}
            </Fragment>
          ))}
        </p>
      )}
    </div>
  );
}

/**
 * A tab before anything has run: one sentence, the same for every tab, and
 * the title bar's next action, so the way forward is right here.
 */
function PanelNoRun({ next }: { next: NextStep | null }) {
  return (
    <div className="panel-no-run">
      <p>
        Nothing has run yet.{" "}
        {next?.run
          ? "Run the code and this tab fills in."
          : next
            ? `${next.detail}.`
            : "Open a project to begin."}
      </p>
      {next && (
        <button
          type="button"
          className="panel-run"
          disabled={next.disabled || next.busy}
          title={next.detail}
          onClick={next.onClick}
        >
          {next.busy ? (
            <LoaderCircle size={13} className="spin" />
          ) : next.run ? (
            <Play size={12} fill="currentColor" />
          ) : null}
          {next.busy ? "Running" : next.label}
          {next.run && !next.busy && <kbd>⌘↵</kbd>}
        </button>
      )}
    </div>
  );
}

/** Problems, variables, and program output for the displayed run. */
export function BottomPanel({
  nextStep = null,
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
  projectId = null,
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
  useEffect(prefetchTabs, []);
  // The word every word-reading panel follows, locked by a click.
  const locked = useLockedPosition();
  // Whether each tab opens with what it shows and how to read it.
  const [explain, setExplain] = useState(() => {
    try {
      return localStorage.getItem("tensorviewer.panel-explain") !== "off";
    } catch {
      return true;
    }
  });
  const toggleExplain = () => {
    setExplain(!explain);
    try {
      if (explain) localStorage.setItem("tensorviewer.panel-explain", "off");
      else localStorage.removeItem("tensorviewer.panel-explain");
    } catch {
      // Without storage the choice lasts until the page reloads.
    }
  };
  // A tab shown beside the others, remembered in this browser.
  const [pinned, setPinned] = useState<PanelTab | null>(() => {
    try {
      return (localStorage.getItem("tensorviewer.panel-pinned") ||
        null) as PanelTab | null;
    } catch {
      return null;
    }
  });
  const pin = (next: PanelTab | null) => {
    setPinned(next);
    try {
      if (next) localStorage.setItem("tensorviewer.panel-pinned", next);
      else localStorage.removeItem("tensorviewer.panel-pinned");
    } catch {
      // Without storage the split lasts until the page reloads.
    }
  };
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
  // The chosen tab scrolls into view when the strip is narrower than its tabs.
  const strip = useRef<HTMLDivElement>(null);
  useEffect(() => {
    strip.current
      ?.querySelector('[aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [tab]);
  const badge = (which: PanelTab) =>
    which === "problems" ? counts.errors + counts.warnings || null : null;
  // One tab's contents: the chosen tab, or the one pinned beside it. Before
  // a run every tab says so alike, except run notes with notes to show.
  const body = (which: PanelTab) => (
    <>
      {explain && <PanelIntro which={which} />}
      {!run && !(which === "problems" && problems.length) ? (
        <PanelNoRun next={nextStep} />
      ) : (
        <Suspense fallback={<PanelLoading>Opening the tab…</PanelLoading>}>
          {content(which)}
        </Suspense>
      )}
    </>
  );
  const content = (which: PanelTab) => (
    <>
      {which === "problems" &&
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
                            disabled={!problem.node && problem.line === null}
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
                <p className="run-checks-clear">
                  <CircleCheck size={13} aria-hidden="true" />
                  No problems in the {run.trace.operations.length} recorded
                  steps.
                </p>
                <dl className="run-checks" aria-label="What was checked">
                  {RUN_CHECKS.map(([group, checks]) => (
                    <Fragment key={group}>
                      <dt>{group}</dt>
                      <dd>
                        {checks.map((check, at) => (
                          <Fragment key={check}>
                            <span>{check}</span>
                            {at < checks.length - 1 && " · "}
                          </Fragment>
                        ))}
                      </dd>
                    </Fragment>
                  ))}
                </dl>
                {run.project.capture_mode === "shapes" && (
                  <p className="panel-empty-detail">
                    Shape-only runs have no values, so the value checks were
                    skipped.
                  </p>
                )}
              </>
            ) : (
              <p>Problems found while running appear here.</p>
            )}
          </div>
        ))}
      {which === "variables" && (
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
      {which === "flow" &&
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
      {which === "watch" && (
        <WatchPanel
          run={run}
          projectId={projectId}
          selected={selected}
          onSelect={onSelect}
        />
      )}
      {which === "weights" && (
        <WeightsPanel
          run={run}
          previousRunId={previousRunId}
          selected={selected}
          onSelect={onSelect}
          onPreview={onPreview}
        />
      )}
      {which === "lens" && (
        <LogitLensPanel
          run={run}
          selected={selected}
          onSelect={onSelect}
          onPreview={onPreview}
        />
      )}
      {which === "map" && <MapPanel run={run} selected={selected} />}
      {which === "attention" && (
        <AttentionPanel
          run={run}
          selected={selected}
          onSelect={onSelect}
          onPreview={onPreview}
        />
      )}
      {which === "trace" && (
        <CausalTracePanel
          run={run}
          previousRunId={previousRunId}
          selected={selected}
          onSelect={onSelect}
          onPreview={onPreview}
        />
      )}
      {which === "output" &&
        (run ? (
          <div className="panel-output">
            {run.trace.stdout ? (
              <pre>{run.trace.stdout}</pre>
            ) : (
              <p className="output-none">
                Nothing was printed. Use print() in your code to write here.
              </p>
            )}
            {/* A failed run ends its output the way Python would. */}
            {run.trace.error && (
              <p className="output-error">
                <b>{run.trace.error.type}</b>: {run.trace.error.message}
                {run.trace.error.line != null &&
                  ` (${run.trace.error.file ?? "line"} ${run.trace.error.line})`}
              </p>
            )}
            <dl className="run-facts">
              <div>
                <dt>Steps</dt>
                <dd>{run.trace.operations.length}</dd>
              </div>
              <div>
                <dt>Tensor states</dt>
                <dd>{Object.keys(run.trace.tensors).length}</dd>
              </div>
              <div>
                <dt title="Including the time recording takes">Time</dt>
                <dd>{run.trace.duration_ms.toFixed(0)} ms</dd>
              </div>
              <div>
                <dt>Recorded</dt>
                <dd>{new Date(run.created_at).toLocaleString()}</dd>
              </div>
            </dl>
          </div>
        ) : (
          <p className="panel-empty">Run to see printed output.</p>
        ))}
    </>
  );
  const label = (which: PanelTab) => panelTab(which).label;
  // A pinned tab stays beside the others; choosing it on the left unsplits.
  const split = pinned && pinned !== tab ? pinned : null;
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
          // A followed word lets go first; the next Escape closes the panel.
          if (locked) lockPosition(null);
          else onClose();
        }}
      >
        <header className="ide-tabs panel-tabs">
          <div
            className="panel-tab-strip"
            role="tablist"
            aria-label="Tensor shelf"
            ref={strip}
            onKeyDown={(event) => {
              // Arrow keys, Home and End move between tabs, as in an editor.
              const at = PANEL_TABS.findIndex(({ id }) => id === tab);
              const next =
                event.key === "ArrowRight"
                  ? (at + 1) % PANEL_TABS.length
                  : event.key === "ArrowLeft"
                    ? (at - 1 + PANEL_TABS.length) % PANEL_TABS.length
                    : event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? PANEL_TABS.length - 1
                        : null;
              if (next === null) return;
              event.preventDefault();
              onTab(PANEL_TABS[next].id);
              strip.current
                ?.querySelectorAll<HTMLElement>('[role="tab"]')
                [next]?.focus();
            }}
          >
            {PANEL_GROUPS.map((group) => (
              <div
                key={group.name}
                className="panel-tab-group"
                role="presentation"
              >
                <span className="panel-tab-group-name" aria-hidden="true">
                  {group.name}
                </span>
                {group.tabs.map(({ id, label, detail }) => {
                  const count = badge(id);
                  return (
                    <button
                      key={id}
                      role="tab"
                      id={`panel-tab-${id}`}
                      aria-controls="panel-pane-main"
                      aria-selected={tab === id}
                      tabIndex={tab === id ? 0 : -1}
                      title={detail}
                      className={pinned === id ? "is-pinned" : undefined}
                      onClick={() => onTab(id)}
                    >
                      {label}
                      {count !== null && (
                        <span
                          className={counts.errors ? "badge failed" : "badge"}
                        >
                          {count}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
          {locked && (
            <button
              type="button"
              className="focus-lock"
              aria-label={`Stop following “${locked.word}”`}
              title="Every panel marks this word until you click it again or press Escape"
              onClick={() => lockPosition(null)}
            >
              Following “{locked.word}” <X size={11} />
            </button>
          )}
          <button
            className="icon-button tab-action"
            aria-label={explain ? "Hide explanations" : "Show explanations"}
            aria-pressed={explain}
            title={
              explain
                ? "Hide what each tab shows and how to read it"
                : "Open each tab with what it shows and how to read it"
            }
            onClick={toggleExplain}
          >
            <CircleHelp size={13} />
          </button>
          <button
            className="icon-button tab-action"
            aria-label={
              pinned === tab
                ? `Unpin ${label(tab)}`
                : `Pin ${label(tab)} beside`
            }
            aria-pressed={pinned === tab}
            title={
              pinned === tab
                ? "Stop showing this tab beside the others"
                : "Keep this tab open on the right while you switch the left between the others"
            }
            onClick={() => pin(pinned === tab ? null : tab)}
          >
            <Columns2 size={13} />
          </button>
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
        <div className={`panel-body${split ? " is-split" : ""}`}>
          <div
            className="panel-pane"
            role="tabpanel"
            id="panel-pane-main"
            aria-labelledby={`panel-tab-${tab}`}
          >
            {body(tab)}
          </div>
          {split && (
            <div
              className="panel-pane panel-pane-pinned"
              role="region"
              aria-label={`${label(split)}, pinned beside`}
            >
              <div className="pinned-head">
                <span>{label(split)}</span>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`Unpin ${label(split)}`}
                  title="Stop showing this beside the other tabs"
                  onClick={() => pin(null)}
                >
                  <X size={12} />
                </button>
              </div>
              {body(split)}
            </div>
          )}
        </div>
      </section>
    </>
  );
}
