import {
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { CircleAlert, Code2, Layers3, SlidersHorizontal } from "lucide-react";
import type { Run } from "../api/client";
import { ancestors, buildJourney } from "../journey/graph";
import { JourneyCanvas } from "../journey/JourneyCanvas";
import { JourneyInspector } from "../journey/JourneyInspector";
import { TransformationFocus } from "../journey/TransformationFocus";
import { TensorVolumeDialog } from "../tensors/TensorVolumeDialog";
import {
  collapseJourney,
  journeyStages,
  stageAncestors,
} from "../journey/stages";
import { StageControls } from "../journey/StageControls";
import { StageFocus } from "../journey/StageFocus";
import { SceneTransport } from "../journey/SceneTransport";
import { cellMotionPlan } from "../journey/cellMotion";
import { useSceneClock, type SceneClock } from "../journey/useSceneClock";
import {
  foldJourney,
  locateOperation,
  loopFolds,
  hiddenOperations,
  loopStepId,
  passLoops,
  representative,
  playbackSteps,
  showStageIterations,
  type LoopFold,
} from "../journey/loops";
import {
  stepDepths,
  stepOutTarget,
  stepOverTarget,
  stopsAt,
} from "../journey/stepping";
import { operationSemantics } from "../journey/sceneSemantics";
import { traceCellContributors } from "../journey/cellContributors";
import { diagnose } from "../operations/diagnosis";
import type { CanvasProbe } from "../journey/CanvasCellProbe";
import { producedTensorIds } from "../tensors/provenance";
import { InspectionActivityContext } from "../journey/InspectionActivity";
import { FocusConnections } from "../journey/FocusConnections";
import { lineageOf } from "../tensors/axisLineage";
import { LineageContext } from "../tensors/LineageContext";
import { stepLabel } from "../operations/kindName";

type Props = {
  run: Run | null;
  busy: boolean;
  /** A run is being recorded (rather than a project or run being opened). */
  recording?: boolean;
  active: boolean;
  /** The shown run was recorded from different code or inputs than the editor holds. */
  stale: boolean;
  onInspect: () => void;
  onEditModel: () => void;
  onEditInputs: () => void;
  /** An editor asks for one operation; a new key repeats the same request. */
  focusOperation?: { id: string; key: number; cell?: number } | null;
  onCurrentOperation?: (id: string | null) => void;
  /** The folded loop whose repeats are playing. */
  onCurrentLoop?: (id: string | null) => void;
  /** How far playback has activated the run (an operation index, or -1). */
  onActivated?: (through: number) => void;
  /** Nodes that wrote a name followed from the tensor shelf. */
  thread?: string[] | null;
  /** Recorded steps where playback pauses (from editor breakpoints). */
  breakpoints?: ReadonlySet<string>;
  /** Steps that carry a warning or error from the run notes. */
  warningSteps?: ReadonlySet<string>;
  /** Playback also pauses at steps with warnings. */
  pauseOnWarnings?: boolean;
  onPauseOnWarnings?: (pause: boolean) => void;
  onCell?: (node: string, index: number) => void;
};
export function Walkthrough({
  run,
  busy,
  recording = false,
  active,
  stale,
  onInspect,
  onEditModel,
  onEditInputs,
  focusOperation,
  onCurrentOperation,
  onCurrentLoop,
  onActivated,
  thread,
  breakpoints,
  warningSteps,
  pauseOnWarnings = false,
  onPauseOnWarnings,
  onCell,
}: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [tensorChoices, setTensorChoices] = useState<Record<string, string>>(
    {},
  );
  const [selection, setSelection] = useState<{
    nodeId: string;
    tensorId?: string;
    cell?: number;
  } | null>(null);
  const [inspector, setInspector] = useState(false);
  const [inspectorView, setInspectorView] = useState<"code" | "values">("code");
  const [showValues, setShowValues] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [connectionsOpen, setConnectionsOpen] = useState(false);
  const [volume, setVolume] = useState<{ id: string; index: number } | null>(
    null,
  );
  const stage = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  // The step playback stopped at because of a breakpoint.
  const [pausedAt, setPausedAt] = useState<{
    id: string;
    reason: "breakpoint" | "warning";
  } | null>(null);
  const [tracedCell, setTracedCell] = useState<{
    operationId: string;
    tensorId: string;
    index: number;
  } | null>(null);
  const [cycle, setCycle] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [following, setFollowing] = useState(true);
  const [reveal, setReveal] = useState(false);
  const [focusKey, setFocusKey] = useState(0);
  const [selectionKey, setSelectionKey] = useState(0);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  // Iterations chosen on a loop's pips; playback otherwise decides.
  const [chosen, setChosen] = useState<Record<string, number>>({});
  const fullGraph = useMemo(
    () => (run ? buildJourney(run.trace) : null),
    [run],
  );
  const stages = useMemo(() => (run ? journeyStages(run) : []), [run]);
  // A loop that repeats the same work is drawn and played once.
  const folds = useMemo(() => (run ? loopFolds(run.trace) : []), [run]);
  const steps = useMemo(
    () => (run ? playbackSteps(run.trace.operations, folds) : []),
    [run, folds],
  );
  // Loops whose passes differ are shown in full, outlined pass by pass.
  const passes = useMemo(
    () => (run ? passLoops(run.trace, folds) : []),
    [run, folds],
  );
  // Every tensor view inside the journey can say where its axes came from.
  const lineage = useMemo(() => (run ? lineageOf(run.trace) : null), [run]);
  const operations = run?.trace.operations ?? [];
  const loopStep = steps.find(
    (step): step is { id: string; fold: LoopFold } =>
      !!step.fold && step.id === selected,
  );
  // The playback position; a selected stage plays from its first step.
  const stepAt = useMemo(() => {
    const direct = steps.findIndex((step) => step.id === selected);
    if (direct >= 0 || !selected) return direct;
    const stage = stages.find((item) => item.id === selected);
    return stage
      ? steps.findIndex(
          (step) => step.operation && step.operation.index >= stage.start_index,
        )
      : -1;
  }, [selected, steps, stages]);
  const clock = useSceneClock({
    key: `${run?.id ?? ""}/${selected ?? ""}/${cycle}`,
    // A loop step always shows its repeats, even when stepping by hand.
    playing: (playing || !!loopStep) && active && !busy,
    speed,
    duration: loopStep
      ? Math.max(1400, 720 * (loopStep.fold.iterations.length - 1))
      : undefined,
    onComplete: () => {
      if (!playing) return;
      if (stepAt >= steps.length - 1) {
        setPlaying(false);
        return;
      }
      const next = steps[stepAt + 1];
      cue(stepAt + 1);
      // A breakpoint line, or a warning when asked, stops playback at its step.
      const reason = stopsAt(next, breakpoints)
        ? "breakpoint"
        : pauseOnWarnings && stopsAt(next, warningSteps)
          ? "warning"
          : null;
      if (reason) {
        setPlaying(false);
        setPausedAt({ id: next.id, reason });
      }
    },
  });
  const repeating = useLoopIteration(clock, loopStep?.fold);
  // Before a loop's repeats its body shows the first iteration, during them
  // the one passing through, and after them the last, which flowed onward.
  const shown = useMemo(() => {
    const result: Record<string, number> = {};
    for (const fold of folds) {
      const at = steps.findIndex((step) => step.id === loopStepId(fold));
      result[fold.id] =
        chosen[fold.id] ??
        (loopStep?.fold.id === fold.id
          ? repeating
          : stepAt > at && at >= 0
            ? fold.iterations.length
            : 1);
    }
    return result;
  }, [folds, steps, chosen, loopStep, repeating, stepAt]);
  const graph = useMemo(() => {
    if (!run || !fullGraph) return null;
    const folded = foldJourney(fullGraph, folds, shown, run.trace);
    return showStageIterations(
      {
        ...collapseJourney(folded, stages, collapsed, run),
        loops: folded.loops,
        passLoops: passes,
      },
      stages,
      folds,
      shown,
      run.trace,
    );
  }, [run, fullGraph, folds, passes, shown, stages, collapsed]);
  const current = graph?.nodes.find((node) => node.id === selected);
  // A followed name, on what the canvas draws: later passes of a folded loop
  // land on the drawn pass, and calls inside a collapsed stage on the stage.
  const threaded = useMemo(() => {
    const found = new Set<string>();
    if (!thread || !graph) return found;
    const hidden = hiddenOperations(folds);
    const ids = new Set(graph.nodes.map((node) => node.id));
    for (const id of thread) {
      const drawn = representative(hidden, id);
      if (ids.has(drawn)) found.add(drawn);
      else {
        const stage = graph.nodes.find((node) =>
          (node.repOperationIds ?? node.stage?.operationIds)?.includes(drawn),
        );
        if (stage) found.add(stage.id);
      }
    }
    return found;
  }, [thread, graph, folds]);
  // Everything up to the playback position has run; later tensors are unlit.
  const reachedThrough = loopStep
    ? operations.find(
        (op) =>
          op.id ===
          loopStep.fold.iterations[shown[loopStep.fold.id] - 1]?.at(-1),
      )?.index
    : current?.operation
      ? current.operation.index
      : current?.stage
        ? current.stage.end_index - 1
        : undefined;
  // Activation is permanent for a run: once a step has run, its tensors hold
  // their values and keep glowing, even when playback steps back.
  const [lit, setLit] = useState({ run: "", through: -1 });
  const litThrough = lit.run === run?.id ? lit.through : -1;
  useEffect(() => {
    if (run && reachedThrough !== undefined && reachedThrough > litThrough)
      setLit({ run: run.id, through: reachedThrough });
  }, [run?.id, reachedThrough, litThrough]);
  useEffect(() => {
    onActivated?.(litThrough);
  }, [litThrough, run?.id]);
  // Only tensors that were still unlit when this step began kindle; the
  // rest are already glowing and stay that way.
  const kindle = useRef({ key: "", above: -1 });
  const stepKey = `${run?.id ?? ""}/${selected ?? ""}/${cycle}`;
  if (kindle.current.key !== stepKey)
    kindle.current = { key: stepKey, above: litThrough };
  // A failed run names the step that stopped it and, when it can, why.
  const failedStep = run?.trace.operations.find((op) => op.status === "error");
  const diagnosis = useMemo(
    () => (run && failedStep ? diagnose(failedStep, run.trace.tensors) : null),
    [run, failedStep],
  );
  // The transport lists playback steps: operations and loop repeats.
  const transportSteps = useMemo(
    () =>
      steps.map((step) =>
        step.operation
          ? step.operation
          : {
              id: step.id,
              kind: `↻ ${step.fold.text} · ×${step.fold.iterations.length}`,
              outputs: [],
            },
      ),
    [steps],
  );
  const highlighted = useMemo(
    () => (graph && selected ? ancestors(graph, selected) : new Set<string>()),
    [graph, selected],
  );
  const motionPlan = useMemo(
    () =>
      run && current?.operation
        ? cellMotionPlan(run, current.operation)
        : undefined,
    [run, current?.operation],
  );
  const semantics = useMemo(
    () =>
      run && current?.operation
        ? operationSemantics(run, current.operation)
        : undefined,
    [run, current?.operation],
  );
  const probe = useMemo<CanvasProbe | undefined>(() => {
    if (!run || !current?.operation || tracedCell?.operationId !== current.id)
      return undefined;
    const tensor = run.trace.tensors[tracedCell.tensorId];
    if (!tensor) return undefined;
    return {
      tensor,
      index: tracedCell.index,
      runId: run.id,
      result: traceCellContributors(
        run,
        current.operation,
        tensor.id,
        tracedCell.index,
      ),
    };
  }, [run, current?.operation, tracedCell]);

  useEffect(() => {
    setSelected(null);
    setTensorChoices({});
    setSelection(null);
    setVolume(null);
    setInspector(false);
    setInspectorView("code");
    setExpanded(false);
    setConnectionsOpen(false);
    setPlaying(false);
    setTracedCell(null);
    setChosen({});
    setCycle(0);
    setFollowing(true);
    setReveal(false);
    setFocusKey(0);
    setSelectionKey(0);
    setCollapsed(
      new Set(
        operations.length > 24
          ? stages.filter((item) => !item.parentStageId).map((item) => item.id)
          : [],
      ),
    );
  }, [run?.id]);
  useEffect(() => {
    if (!selected) return;
    // A loop step opens the stages around its body.
    const parents = new Set(
      stageAncestors(
        stages,
        loopStep ? loopStep.fold.iterations[0][0] : selected,
      ).map((stage) => stage.id),
    );
    if (!parents.size) return;
    setCollapsed((previous) =>
      [...previous].some((id) => parents.has(id))
        ? new Set([...previous].filter((id) => !parents.has(id)))
        : previous,
    );
  }, [selected, stages]);
  useEffect(() => {
    if (!active || busy) setPlaying(false);
  }, [active, busy]);
  useEffect(() => {
    const pauseWhenHidden = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () =>
      document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, []);

  useEffect(() => {
    // The editor's loop chip asks for a loop's repeats.
    const loopAt = focusOperation
      ? steps.findIndex((step) => step.id === focusOperation.id)
      : -1;
    if (loopAt >= 0 && steps[loopAt].fold) {
      setPlaying(false);
      setExpanded(false);
      onInspect();
      cue(loopAt);
    } else if (
      focusOperation &&
      fullGraph?.nodes.some((node) => node.id === focusOperation.id)
    ) {
      // A later iteration of a folded loop opens on its first-iteration node,
      // with the loop showing that iteration.
      const located = locateOperation(folds, focusOperation.id);
      setChosen(located.shown);
      select(
        located.id,
        undefined,
        focusOperation.cell,
        focusOperation.cell !== undefined,
      );
    }
  }, [focusOperation?.key]);
  // During a loop's repeats, the pass passing through ends at its last step.
  const currentOperationId = loopStep
    ? (loopStep.fold.iterations[shown[loopStep.fold.id] - 1]?.at(-1) ?? null)
    : current?.stage
      ? null
      : (current?.operation?.id ?? current?.id ?? null);
  useEffect(() => {
    onCurrentOperation?.(currentOperationId);
  }, [currentOperationId, expanded, run?.id]);
  const currentLoopId = loopStep?.fold.id ?? null;
  useEffect(() => {
    onCurrentLoop?.(currentLoopId);
  }, [currentLoopId, run?.id]);

  function select(
    id: string,
    tensorId?: string,
    cell?: number,
    inspect = false,
  ) {
    setTracedCell(null);
    setSelected(id);
    setSelection({ nodeId: id, tensorId: tensorId ?? tensorChoices[id], cell });
    setExpanded(inspect);
    setPlaying(false);
    setFocusKey((key) => key + 1);
    setSelectionKey((key) => key + 1);
    onInspect();
  }
  function inspect(id: string, tensorId?: string, cell?: number) {
    select(id, tensorId, cell, true);
  }
  function inspectCurrent() {
    // A loop's repeats open the result of the pass showing, which the loop
    // keeps showing while it is inspected.
    if (loopStep) {
      const fold = loopStep.fold;
      setChosen((previous) => ({ ...previous, [fold.id]: shown[fold.id] }));
      inspect(fold.iterations[0].at(-1)!);
      return;
    }
    if (!current) return;
    if (tracedCell?.operationId === current.id) {
      inspect(current.id, tracedCell.tensorId, tracedCell.index);
    } else {
      const chosen = selection?.nodeId === current.id ? selection : null;
      inspect(current.id, chosen?.tensorId, chosen?.cell);
    }
  }
  function returnToCanvas() {
    setExpanded(false);
    setInspector(false);
    setPlaying(false);
    setFocusKey((key) => key + 1);
    setSelectionKey((key) => key + 1);
    stage.current
      ?.querySelector<HTMLElement>(".journey-canvas")
      ?.focus({ preventScroll: true });
  }
  function overview() {
    setExpanded(false);
    setInspector(false);
    setTracedCell(null);
    setPlaying(false);
    setFocusKey(0);
    stage.current
      ?.querySelector<HTMLElement>(".journey-canvas")
      ?.focus({ preventScroll: true });
  }
  function cue(next: number) {
    const step = steps[next];
    if (!step) return;
    setTracedCell(null);
    setChosen({});
    setCycle((value) => value + 1);
    setSelected(step.id);
    setSelection(null);
    setFocusKey((key) => key + 1);
  }
  function jump(next: number) {
    setPlaying(false);
    if (expanded && steps[next]?.operation) inspect(steps[next].id);
    else {
      setExpanded(false);
      cue(next);
    }
  }
  // Debugger stepping: Next steps into calls; these step over and out of them.
  const depths = useMemo(() => stepDepths(steps, stages), [steps, stages]);
  const overTarget = stepOverTarget(depths, stepAt);
  const outTarget = stepOutTarget(depths, stepAt);
  // Debugger keys work anywhere in the workspace, the editor included: F10
  // steps over, F11 into, Shift+F11 out, Shift+F10 back. Ctrl/Cmd+F10 in the
  // editor is run to cursor, which handles the key first.
  const debuggerKeys = useRef<(event: KeyboardEvent) => void>(() => {});
  debuggerKeys.current = (event) => {
    // In a lesson inside a folded loop, [ and ] flip to the other passes.
    if (
      (event.key === "[" || event.key === "]") &&
      expanded &&
      passFold &&
      !event.defaultPrevented &&
      !event.metaKey &&
      !event.ctrlKey &&
      !(event.target as Element | null)?.closest?.(
        "input, textarea, select, [contenteditable]",
      )
    ) {
      event.preventDefault();
      stepPass(event.key === "]" ? 1 : -1);
      return;
    }
    if (
      (event.key !== "F10" && event.key !== "F11") ||
      event.defaultPrevented ||
      event.metaKey ||
      event.ctrlKey ||
      event.altKey ||
      busy ||
      !active ||
      document.querySelector("dialog[open]")
    )
      return;
    event.preventDefault();
    const target =
      event.key === "F10"
        ? event.shiftKey
          ? stepAt > 0
            ? stepAt - 1
            : null
          : overTarget
        : event.shiftKey
          ? outTarget
          : stepAt + 1 < steps.length
            ? stepAt + 1
            : null;
    if (target === null) return;
    onInspect();
    jump(target);
  };
  useEffect(() => {
    const listener = (event: KeyboardEvent) => debuggerKeys.current(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);
  // The innermost folded loop whose shown pass holds the current step.
  const passAnchor = current?.operation?.id ?? current?.stage?.operationIds[0];
  const passFold = passAnchor
    ? [...folds]
        .sort((a, b) => b.depth - a.depth)
        .find((fold) =>
          fold.iterations[(shown[fold.id] ?? 1) - 1]?.includes(passAnchor),
        )
    : undefined;
  // The lesson's chosen cell, kept when flipping passes.
  const lessonCell = useRef<number | undefined>(undefined);
  useEffect(() => {
    lessonCell.current = undefined;
  }, [current?.id, run?.id]);
  function stepPass(delta: number) {
    if (!passFold || !current) return;
    const count = passFold.iterations.length;
    const next = (((shown[passFold.id] ?? 1) - 1 + delta + count) % count) + 1;
    setChosen((previous) => ({ ...previous, [passFold.id]: next }));
    inspect(current.id, undefined, lessonCell.current);
  }
  function showIteration(loopId: string, iteration: number) {
    setPlaying(false);
    setChosen((previous) => ({ ...previous, [loopId]: iteration }));
  }
  function togglePlayback() {
    if (playing) {
      setPlaying(false);
      return;
    }
    setPausedAt(null);
    setTracedCell(null);
    setExpanded(false);
    setInspector(false);
    setSelection(null);
    if (current?.stage) cue(stepAt);
    else if (
      stepAt < 0 ||
      (stepAt >= steps.length - 1 && clock.getSnapshot() >= 1)
    )
      cue(0);
    else if (clock.getSnapshot() >= 1) cue(stepAt);
    else setFocusKey((key) => key + 1);
    setPlaying(true);
    onInspect();
  }
  function changeReveal(next: boolean) {
    setReveal(next);
    if (next) {
      setCollapsed(new Set());
      if (current?.stage) {
        cue(stepAt);
        setExpanded(false);
      } else if (stepAt < 0) cue(0);
    }
  }
  function toggleStage(id: string) {
    const closing = !collapsed.has(id);
    setCollapsed((previous) => {
      const next = new Set(previous);
      if (next.has(id)) {
        next.delete(id);
        // Open one level at a time, keeping large nested blocks readable.
        stages
          .filter((item) => item.parentStageId === id)
          .forEach((item) => next.add(item.id));
      } else next.add(id);
      return next;
    });
    setSelected(closing ? id : null);
    setInspector(false);
    overview();
  }
  function stageOverview() {
    setCollapsed(
      new Set(
        stages.filter((item) => !item.parentStageId).map((item) => item.id),
      ),
    );
    setSelected(null);
    setInspector(false);
    overview();
  }
  function expandAll() {
    setCollapsed(new Set());
    setSelected(null);
    setInspector(false);
    overview();
  }

  if (!run || !graph)
    return (
      <section className="journey-view">
        <div className="empty-workspace">
          <Layers3 size={38} />
          <h2>
            {recording
              ? "Recording your forward pass…"
              : busy
                ? "Opening the project…"
                : "Your tensor journey starts here"}
          </h2>
          <p>
            {recording
              ? "Running your model and recording its tensor transformations."
              : busy
                ? "Loading its code and its latest recorded run."
                : "Write tensor statements or set up your model, then press Run."}
          </p>
        </div>
      </section>
    );
  const connections = current && fullGraph && (
    <FocusConnections
      key={`connections-${run.id}-${current.id}`}
      run={run}
      graph={
        graph.nodes.some((node) => node.id === current.id) ? graph : fullGraph
      }
      node={current}
      open={connectionsOpen}
      onOpen={setConnectionsOpen}
      onSelect={(id) => inspect(id)}
    />
  );
  return (
    <LineageContext value={lineage}>
      <section
        className="journey-view"
        aria-label="Tensor journey"
        aria-keyshortcuts="F10 F11 Shift+F11 Shift+F10"
      >
        <div
          className={`journey-stage ${expanded && active ? "has-focus" : ""}`}
          ref={stage}
        >
          <div className="scene-context">
            <span className="scene-model-name" title={run.project.class_name}>
              {run.project.class_name}
            </span>
            <StageControls
              stages={stages}
              collapsed={collapsed}
              disabled={reveal || playing}
              onToggle={toggleStage}
              onOverview={stageOverview}
              onExpandAll={expandAll}
            />
          </div>
          {run.trace.error && (
            <div className="trace-error-strip" role="alert">
              <CircleAlert size={16} />
              <div className="run-error-content">
                {/* The diagnosis leads; PyTorch's own message follows it. */}
                <b>{diagnosis?.title ?? "The run stopped here"}</b>
                {diagnosis && (
                  <p className="run-error-explanation">
                    {diagnosis.explanation}
                    {diagnosis.suggestion && (
                      <span className="run-error-suggestion">
                        {" "}
                        {diagnosis.suggestion}
                      </span>
                    )}
                  </p>
                )}
                {stale && (
                  <p className="run-error-hint">
                    This is a saved execution. Run again to use your current
                    code and inputs.
                  </p>
                )}
                <p className={`run-error-message ${diagnosis ? "raw" : ""}`}>
                  {run.trace.error.message}
                </p>
                <small>
                  {run.trace.error.type}
                  {run.trace.error.line
                    ? ` · ${run.trace.error.file ?? "line"} ${run.trace.error.line}`
                    : ""}
                </small>
                <div className="run-error-actions">
                  {failedStep && (
                    <button
                      className="secondary-button small"
                      disabled={busy}
                      onClick={() => inspect(failedStep.id)}
                    >
                      <CircleAlert size={14} />
                      Show the failing step
                    </button>
                  )}
                  <button
                    className="secondary-button small"
                    disabled={busy}
                    onClick={onEditModel}
                  >
                    <Code2 size={14} />
                    {run.project.blueprint ? "Edit model" : "Fix code"}
                  </button>
                  <button
                    className="secondary-button small"
                    disabled={busy}
                    onClick={onEditInputs}
                  >
                    <SlidersHorizontal size={14} />
                    Edit inputs
                  </button>
                </div>
                {!!operations.length && (
                  <p className="run-error-hint">
                    Earlier steps are still available in the diagram.
                  </p>
                )}
              </div>
            </div>
          )}
          <JourneyCanvas
            key={run.id}
            graph={graph}
            selectedId={selected}
            reachedThrough={Math.max(litThrough, reachedThrough ?? -1)}
            threaded={threaded}
            kindleAbove={kindle.current.above}
            topInset={run.trace.error ? 190 : 0}
            activeLoopId={loopStep?.fold.id}
            onLoopIteration={showIteration}
            onLoopSelect={(id) => {
              setPlaying(false);
              setExpanded(false);
              cue(steps.findIndex((step) => step.id === loopStepId({ id })));
            }}
            highlighted={highlighted}
            focusKey={focusKey}
            selectionKey={selectionKey}
            playing={playing}
            onPlaybackToggle={
              operations.length && !busy ? togglePlayback : undefined
            }
            sceneMode={!expanded}
            semantics={semantics}
            probe={!expanded ? probe : undefined}
            onTraceCell={(tensorId, index) => {
              if (!current?.operation) return;
              setPlaying(false);
              setTracedCell({ operationId: current.id, tensorId, index });
            }}
            onClearTrace={() => setTracedCell(null)}
            outputIds={run.trace.output_ids}
            followPlayback={following}
            motion={
              !expanded && motionPlan
                ? { plan: motionPlan, clock, tensors: run.trace.tensors }
                : undefined
            }
            onFollowPlaybackChange={setFollowing}
            onTensorChoice={(nodeId, tensorId) =>
              setTensorChoices((previous) => ({
                ...previous,
                [nodeId]: tensorId,
              }))
            }
            visibleThrough={
              reveal
                ? loopStep
                  ? (operations.find(
                      (op) => op.id === loopStep.fold.iterations.at(-1)!.at(-1),
                    )?.index ?? -1)
                  : (current?.operation?.index ?? -1)
                : undefined
            }
            onSelect={select}
            onInspect={current || loopStep ? inspectCurrent : undefined}
            onOverview={overview}
            onTensorInspect={(id, index) => {
              setPlaying(false);
              setVolume({ id, index });
            }}
            onStageToggle={toggleStage}
          />
          {active && volume && run.trace.tensors[volume.id] && (
            <TensorVolumeDialog
              tensor={run.trace.tensors[volume.id]}
              runId={run.id}
              initialIndex={volume.index}
              onSelect={(index) => {
                if (
                  current?.operation &&
                  producedTensorIds(current.operation).includes(volume.id)
                )
                  setTracedCell({
                    operationId: current.id,
                    tensorId: volume.id,
                    index,
                  });
              }}
              onClose={() => setVolume(null)}
            />
          )}
          <InspectionActivityContext value={expanded && active && !busy}>
            {expanded && current?.stage && (
              <StageFocus
                key={`${run.id}-${current.stage.id}-${selection?.nodeId === current.id ? (selection.tensorId ?? "") : ""}`}
                active={active}
                run={run}
                stage={current.stage}
                connections={connections}
                initialTensorId={
                  selection?.nodeId === current.id
                    ? selection.tensorId
                    : undefined
                }
                onClose={returnToCanvas}
                onExpand={() => toggleStage(current.stage!.id)}
                onSelect={inspect}
                showValues={showValues}
                onShowValues={setShowValues}
                pass={
                  passFold
                    ? {
                        iteration: shown[passFold.id] ?? 1,
                        count: passFold.iterations.length,
                        text: passFold.text,
                        onStep: stepPass,
                      }
                    : undefined
                }
              />
            )}
            {expanded && current && !current.stage && (
              <TransformationFocus
                key={`${run.id}-${current.id}`}
                active={active}
                run={run}
                node={current}
                connections={connections}
                inspectorOpen={inspector}
                codeOpen={inspector && inspectorView === "code"}
                onSelect={inspect}
                onClose={returnToCanvas}
                onCode={(open) => {
                  setInspector(open);
                  setInspectorView("code");
                }}
                showValues={showValues}
                onShowValues={setShowValues}
                initialTensorId={
                  selection?.nodeId === current.id
                    ? selection.tensorId
                    : undefined
                }
                initialCell={
                  selection?.nodeId === current.id ? selection.cell : undefined
                }
                onCell={(index) => {
                  lessonCell.current = index;
                  onCell?.(current.id, index);
                }}
                pass={
                  passFold
                    ? { loopId: passFold.id, onStep: stepPass }
                    : undefined
                }
              />
            )}
          </InspectionActivityContext>
          {/* A fresh run starts unlit; say what lights it up, until it does. */}
          {litThrough < 0 && !selected && !playing && !expanded && (
            <p className="journey-hint" role="note">
              <b>Play</b> lights up each tensor as its step runs ·{" "}
              <kbd>F11</kbd> steps one at a time
            </p>
          )}
          <SceneTransport
            clock={clock}
            operations={transportSteps}
            index={stepAt}
            frameLabel={
              loopStep
                ? `↻ ${loopStep.fold.text}`
                : current?.stage
                  ? current.stage.title
                  : current?.operation
                    ? stepLabel(
                        current.operation.kind,
                        run.trace.tensors[current.operation.outputs[0]]?.name,
                      )
                    : current?.tensors[0]?.name
            }
            framePosition={
              loopStep
                ? `${shown[loopStep.fold.id]} of ${loopStep.fold.iterations.length} · ${stepAt + 1} / ${steps.length}`
                : current?.stage
                  ? `${current.stage.start_index + 1}–${current.stage.end_index} / ${operations.length}`
                  : undefined
            }
            nextIndex={current?.stage ? stepAt : stepAt + 1}
            playing={playing}
            breakpoint={
              selected && pausedAt?.id === selected
                ? pausedAt.reason
                : undefined
            }
            onStepOver={
              overTarget !== null ? () => jump(overTarget) : undefined
            }
            onStepOut={outTarget !== null ? () => jump(outTarget) : undefined}
            pauseOnWarnings={onPauseOnWarnings ? pauseOnWarnings : undefined}
            onPauseOnWarnings={onPauseOnWarnings}
            busy={busy}
            expanded={expanded}
            canInspect={!!current || !!loopStep}
            inspectLabel={
              loopStep
                ? `Inspect pass ${shown[loopStep.fold.id]}'s result`
                : current?.stage
                  ? "Inspect current stage"
                  : current && !current.operation
                    ? "Inspect current tensor"
                    : "Inspect current operation"
            }
            speed={speed}
            reveal={reveal}
            following={following}
            notices={run.trace.warnings?.length ?? 0}
            stopped={!!run.trace.error}
            onPlay={togglePlayback}
            onSeek={jump}
            onInspect={inspectCurrent}
            onOverview={returnToCanvas}
            onSpeed={setSpeed}
            onReveal={changeReveal}
            onFollow={setFollowing}
          >
            <details
              className="journey-run-details"
              open={!!run.trace.warnings?.length || !!run.trace.error}
            >
              <summary>
                Run details
                {run.trace.warnings?.length ? " · tracking notice" : ""}
              </summary>
              <div>
                {run.trace.error && (
                  <p className="run-tracking-warning">
                    <CircleAlert size={14} /> {run.trace.error.type}:{" "}
                    {run.trace.error.message}
                  </p>
                )}
                {run.trace.warnings?.map((warning, i) => (
                  <p className="run-tracking-warning" key={i}>
                    <CircleAlert size={14} /> {warning}
                  </p>
                ))}
                {run.trace.operations.some((op) => op.mutations?.length) && (
                  <p>
                    Dashed connections carry shared-storage dependencies. Select
                    an in-place step to compare each recorded view.
                  </p>
                )}
                <p>
                  {Object.keys(run.trace.tensors).length} tensor states ·{" "}
                  {run.trace.duration_ms.toFixed(0)} ms including tracing
                </p>
                <p>
                  {new Date(run.created_at).toLocaleString()} ·{" "}
                  {run.project.capture_mode === "shapes"
                    ? "Shapes only"
                    : "CPU values"}{" "}
                  · evaluation mode
                </p>
                <p>
                  Entry: <code>{run.project.entry_path ?? "model.py"}</code> ·{" "}
                  {run.project.class_name}
                </p>
                {run.project.repository && (
                  <p title={run.project.repository.url}>
                    Source imported from commit{" "}
                    <code>{run.project.repository.revision.slice(0, 12)}</code>.
                    This run preserves its own source snapshot.
                  </p>
                )}
                {run.trace.runtime?.Python && (
                  <p>
                    Python {run.trace.runtime.Python} · PyTorch{" "}
                    {run.trace.runtime.torch ?? "unknown"} ·{" "}
                    {run.project.environment
                      ? "selected environment"
                      : "TensorViewer environment"}
                  </p>
                )}
                <p>
                  Tensor drawings are schematic. Stacks represent leading
                  dimensions; weights are available in Tensor details.
                </p>
                {run.project.weights ? (
                  <p className="run-weight-provenance">
                    Weights: <b>{run.project.weights.name}</b> ·{" "}
                    {run.trace.weight_check?.compatible
                      ? "matched to model"
                      : "not validated"}
                    <br />
                    <code>SHA-256 {run.project.weights.sha256}</code>
                  </p>
                ) : (
                  <p>
                    Weights: initialized by the module · model seed{" "}
                    {run.project.input.seed}
                  </p>
                )}
                {run.trace.stdout && <pre>{run.trace.stdout}</pre>}
              </div>
            </details>
          </SceneTransport>
        </div>
        {inspector && current && !current.stage && (
          <JourneyInspector
            active={active}
            run={run}
            node={current}
            initialTensorId={
              selection?.nodeId === current.id ? selection.tensorId : undefined
            }
            initialCell={
              selection?.nodeId === current.id ? selection.cell : undefined
            }
            onSelect={inspect}
            onClose={() => setInspector(false)}
            tab={inspectorView}
            onTab={setInspectorView}
            showValues={showValues}
            onShowValues={setShowValues}
          />
        )}
      </section>
    </LineageContext>
  );
}

/** The iteration passing through a loop's body while its repeats play. */
function useLoopIteration(clock: SceneClock, fold?: LoopFold) {
  const count = fold?.iterations.length ?? 1;
  const read = () =>
    fold
      ? Math.min(count, 2 + Math.floor(clock.getSnapshot() * (count - 1)))
      : 1;
  return useSyncExternalStore(clock.subscribe, read, read);
}
