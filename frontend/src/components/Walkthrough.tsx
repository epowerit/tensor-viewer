import {
  Fragment,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  CircleAlert,
  Code2,
  Layers3,
  Play,
  SlidersHorizontal,
} from "lucide-react";
import { api, type Run } from "../api/client";
import { unravel } from "../tensors/coordinates";
import { ancestors, buildJourney } from "../journey/graph";
import { JourneyCanvas } from "../journey/JourneyCanvas";
import { JourneyInspector } from "../journey/JourneyInspector";
import { TransformationFocus } from "../journey/TransformationFocus";
import { TensorVolumeDialog } from "../tensors/TensorVolumeDialog";
import {
  collapseJourney,
  detailLevelOf,
  detailLevels,
  followFolds,
  journeyStages,
  levelCards,
  rememberFolds,
  rememberedFolds,
  startingLevel,
  stageAncestors,
  type DetailLevel,
} from "../journey/stages";
import {
  axisNames,
  isLayout,
  layoutSemantics,
  layoutStages,
} from "../journey/relayout";
import { einopsPattern, variableName } from "../journey/einops";
import { carryOver, type CanvasMemory, type LeftOff } from "../journey/reload";
import { DetailDial } from "../journey/DetailDial";
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
  foldedPlayback,
  stepDepths,
  stepOutTarget,
  stepOverTarget,
  stopInside,
} from "../journey/stepping";
import { operationSemantics } from "../journey/sceneSemantics";
import { traceCellContributors } from "../journey/cellContributors";
import type { CodeFix } from "../operations/codeFixes";
import {
  diagnose,
  diagnoseError,
  recordedNames,
} from "../operations/diagnosis";
import type { CanvasProbe } from "../journey/CanvasCellProbe";
import { producedTensorIds } from "../tensors/provenance";
import { InspectionActivityContext } from "../journey/InspectionActivity";
import { FocusConnections } from "../journey/FocusConnections";
import { axisMap, lineageOf } from "../tensors/axisLineage";
import { LineageContext } from "../tensors/LineageContext";
import { TokenContext, tokenAxes } from "../tensors/TokenContext";
import { TensorUseContext, tensorUses } from "../tensors/TensorUseContext";
import { RunBeforeContext, loadRun } from "../tensors/diff";
import { kindName, stepLabel } from "../operations/kindName";
import { changeValues, type FlowLens } from "../journey/flow";
import { liveMemory } from "../journey/liveMemory";
import { ScaleProjection } from "../journey/ScaleProjection";
import { SymbolicContext } from "../tensors/InkShape";
import { bytesText } from "../journey/cost";
import {
  GradientLensContext,
  type GradientTarget,
} from "../journey/GradientLens";
import { useComparison } from "../workspace/useComparison";

type Props = {
  run: Run | null;
  /** The run recorded just before this one, for the change lens. */
  previousRunId?: string | null;
  /** A step previewed in the Flow table, traced on the canvas. */
  previewStep?: string | null;
  /** The steps under the pointer on the canvas, for the Flow table. */
  onHoverStep?: (ids: string[] | null) => void;
  busy: boolean;
  /** A run is being recorded (rather than a project or run being opened). */
  recording?: boolean;
  active: boolean;
  /** The shown run was recorded from different code or inputs than the editor holds. */
  stale: boolean;
  onInspect: () => void;
  onEditModel: () => void;
  /** Checked one-line fixes for a failed run, and how to apply one. */
  fixes?: {
    checking: boolean;
    items: (CodeFix & { verdict: string })[];
    apply: (fix: CodeFix) => void;
  } | null;
  onEditInputs: () => void;
  /** Record a run; absent while running is not possible. */
  onRun?: () => void;
  /** Open the project's code (or its diagram). */
  onShowCode?: () => void;
  /** An editor asks for one operation; a new key repeats the same request. */
  focusOperation?: { id: string; key: number; cell?: number } | null;
  onCurrentOperation?: (id: string | null) => void;
  /** The folded loop whose repeats are playing. */
  onCurrentLoop?: (id: string | null) => void;
  /** The folded call or capsule playback is on, played as one step. */
  onCurrentCard?: (card: CurrentCard | null) => void;
  /** Folding, for the command palette: here, and the detail dial's levels. */
  onFoldControls?: (controls: FoldControls | null) => void;
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
  /**
   * Where the last diagram of this project was left. A new run opens there,
   * drawn from the new run: the same step, folds, and camera.
   */
  leftOff?: { current: LeftOff | null };
  /** Where a recorded line is in the code now, edited since the run. */
  placeLine?: (file: string | null, line: number) => number | null;
  /** Steps lit again from elsewhere, such as the palette's last change. */
  relit?: ReadonlySet<string> | null;
  /** The run opened: what changed since the last one, if it carried on from it. */
  onCarried?: (
    change: {
      summary: string;
      first: string | null;
      structural: boolean;
      valuesMayDiffer: boolean;
      changed: ReadonlySet<string>;
      renames: Map<string, string>;
      /** Turns on the change lens, comparing every tensor with the run before. */
      compare: () => void;
    } | null,
  ) => void;
};
/** A folded call or capsule played as one step, as the rest of the IDE shows it. */
/** Folding as the command palette offers it. */
export type FoldControls = {
  /** The canvas card for a module call, by its path and first step. */
  call: (path: string, first: string) => { id: string; folded: boolean } | null;
  /** Folds or unfolds a call on the canvas. */
  toggle: (id: string) => void;
  canFold: boolean;
  canUnfold: boolean;
  fold: () => void;
  unfold: () => void;
  levels: {
    label: string;
    detail: string;
    current: boolean;
    choose: () => void;
  }[];
};

export type CurrentCard = {
  title: string;
  operationIds: string[];
  /** The first and last source lines its steps record, if any. */
  lines: [number, number] | null;
};

export function Walkthrough({
  run,
  previousRunId = null,
  previewStep = null,
  onHoverStep,
  busy,
  recording = false,
  active,
  stale,
  onInspect,
  onEditModel,
  fixes = null,
  onEditInputs,
  onRun,
  onShowCode,
  focusOperation,
  onCurrentOperation,
  onCurrentLoop,
  onCurrentCard,
  onFoldControls,
  onActivated,
  thread,
  breakpoints,
  warningSteps,
  pauseOnWarnings = false,
  onPauseOnWarnings,
  onCell,
  leftOff,
  onCarried,
  placeLine = (_, line) => line,
  relit = null,
}: Props) {
  // A run of the project already on screen, such as the one saving the code
  // records, takes over where that one was.
  const [carried] = useState(() => carryOver(leftOff?.current ?? null, run));
  const [selected, setSelected] = useState<string | null>(
    carried?.selected ?? null,
  );
  const [tensorChoices, setTensorChoices] = useState<Record<string, string>>(
    {},
  );
  const [selection, setSelection] = useState<{
    nodeId: string;
    tensorId?: string;
    cell?: number;
  } | null>(carried?.selection ?? null);
  const [inspector, setInspector] = useState(carried?.inspector ?? false);
  const [inspectorView, setInspectorView] = useState<"code" | "values">(
    carried?.inspectorView ?? "code",
  );
  const [showValues, setShowValues] = useState(true);
  const [expanded, setExpanded] = useState(carried?.expanded ?? false);
  const [connectionsOpen, setConnectionsOpen] = useState(false);
  const [volume, setVolume] = useState<{ id: string; index: number } | null>(
    null,
  );
  const stage = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(carried?.playback.playing ?? false);
  // The step playback stopped at because of a breakpoint.
  const [pausedAt, setPausedAt] = useState<{
    id: string;
    reason: "breakpoint" | "warning";
    /** The operation it stopped for, which may be inside a folded card. */
    inside?: string;
  } | null>(null);
  const [tracedCell, setTracedCell] = useState<{
    operationId: string;
    tensorId: string;
    index: number;
  } | null>(null);
  const [cycle, setCycle] = useState(0);
  const [speed, setSpeed] = useState(carried?.playback.speed ?? 1);
  const [following, setFollowing] = useState(
    carried?.playback.following ?? true,
  );
  const [reveal, setReveal] = useState(carried?.playback.reveal ?? false);
  const [focusKey, setFocusKey] = useState(carried?.focused ? 1 : 0);
  const [selectionKey, setSelectionKey] = useState(0);
  const [collapsed, setCollapsed] = useState<Set<string>>(
    () => carried?.collapsed ?? new Set(),
  );
  // What the edit changed, marked on the canvas for a moment.
  const [changed, setChanged] = useState(carried?.changed ?? null);
  useEffect(() => {
    onCarried?.(
      carried
        ? {
            summary: carried.summary,
            first: carried.first,
            structural: carried.structural,
            valuesMayDiffer: carried.valuesMayDiffer,
            changed: carried.changed,
            renames: carried.renames,
            compare: () => chooseLens("change"),
          }
        : null,
    );
    const timer = window.setTimeout(() => setChanged(null), 2600);
    return () => clearTimeout(timer);
  }, []);
  const canvasMemory = useRef<CanvasMemory | null>(null);
  // Iterations chosen on a loop's pips; playback otherwise decides.
  const [chosen, setChosen] = useState<Record<string, number>>({});
  const fullGraph = useMemo(
    () => (run ? buildJourney(run.trace) : null),
    [run],
  );
  // Module calls, and the layout capsules inside them: runs of steps that
  // only rearrange a tensor, foldable into one card like a call.
  const stages = useMemo(() => {
    if (!run) return [];
    const calls = journeyStages(run);
    return [...calls, ...layoutStages(run, calls)];
  }, [run]);
  const levels = useMemo(() => detailLevels(stages), [stages]);
  // How many cards each setting draws, the model's inputs among them and a
  // folded loop's passes counted once.
  const levelSizes = useMemo(() => {
    if (!run) return [];
    const ids = run.trace.operations.map((operation) => operation.id);
    const hidden = hiddenOperations(loopFolds(run.trace));
    return levels.map(
      (level) =>
        levelCards(level, stages, ids, hidden) + run.trace.input_ids.length,
    );
  }, [run, levels, stages]);
  // A loop that repeats the same work is drawn and played once.
  const folds = useMemo(() => (run ? loopFolds(run.trace) : []), [run]);
  const recorded = useMemo(
    () => (run ? playbackSteps(run.trace.operations, folds) : []),
    [run, folds],
  );
  // Playback follows the cards on screen: a folded call or capsule is one
  // step, which runs all of its operations together.
  const steps = useMemo(
    () => foldedPlayback(recorded, stages, collapsed),
    [recorded, stages, collapsed],
  );
  // Loops whose passes differ are shown in full, outlined pass by pass.
  const passes = useMemo(
    () => (run ? passLoops(run.trace, folds) : []),
    [run, folds],
  );
  // Every tensor view inside the journey can say where its axes came from.
  const lineage = useMemo(() => (run ? lineageOf(run.trace) : null), [run]);
  // A sentence's words along every axis that came from its positions.
  const tokens = useMemo(
    () => (run ? tokenAxes(run, lineage) : null),
    [run, lineage],
  );
  // Tensor details link the steps that made and read each state.
  const inspectStep = useRef<
    (id: string, tensorId?: string, cell?: number) => void
  >(() => {});
  const tensorFlow = useMemo(
    () =>
      run
        ? {
            uses: tensorUses(run.trace),
            go: (id: string, tensorId?: string, cell?: number) =>
              inspectStep.current(id, tensorId, cell),
            trace: run.trace,
            runId: run.id,
          }
        : null,
    [run],
  );
  const operations = run?.trace.operations ?? [];
  // The line a pause inside a folded card stopped for, where it is now.
  const pausedSource = pausedAt?.inside
    ? operations.find((op) => op.id === pausedAt.inside)?.source
    : undefined;
  const pausedLine =
    (pausedSource && placeLine(pausedSource.file ?? null, pausedSource.line)) ??
    "?";
  const loopStep = steps.find(
    (step): step is { id: string; fold: LoopFold } =>
      !!step.fold && step.id === selected,
  );
  // The playback position; a selected stage plays from its first step.
  const stepAt = useMemo(() => {
    const direct = steps.findIndex((step) => step.id === selected);
    if (direct >= 0 || !selected) return direct;
    // A step folded into a card since it was reached plays on as the card.
    const holding = steps.findIndex((step) =>
      step.stage?.operationIds.includes(selected),
    );
    if (holding >= 0) return holding;
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
      // A breakpoint line, or a warning when asked, stops playback at its
      // step, or at the folded card holding it.
      const breakAt = stopInside(next, breakpoints);
      const warnAt = pauseOnWarnings
        ? stopInside(next, warningSteps)
        : undefined;
      const reason = breakAt ? "breakpoint" : warnAt ? "warning" : null;
      if (reason) {
        setPlaying(false);
        setPausedAt({ id: next.id, reason, inside: breakAt ?? warnAt });
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
  // A playback step inside a folded call or capsule stays folded: the canvas
  // shows the card that holds it as current, while the step itself, its
  // lesson included, stays the playback position.
  const stepNode = graph?.nodes.find((node) => node.id === selected);
  const heldIn = useMemo(() => {
    if (!graph || !selected || stepNode) return undefined;
    const drawn = representative(hiddenOperations(folds), selected);
    return graph.nodes.find(
      (node) =>
        node.stage &&
        (node.repOperationIds ?? node.stage.operationIds).includes(drawn),
    );
  }, [graph, selected, stepNode, folds]);
  const current =
    stepNode ??
    (heldIn && expanded
      ? fullGraph?.nodes.find((node) => node.id === selected)
      : heldIn);
  // A folded card's own steps, each with its result. Played as one step,
  // the card flips through them during its beat; a step folded into it
  // since it was reached is shown where it sits ("transpose · 2 of 2").
  const inside = useMemo(() => {
    const card =
      heldIn ?? (current?.stage && steps[stepAt]?.stage ? current : undefined);
    if (!card?.stage || !run) return undefined;
    const byId = new Map(operations.map((op) => [op.id, op]));
    const ops = card.stage.operationIds
      .map((id) => byId.get(id))
      .filter((op): op is (typeof operations)[number] => !!op);
    const list = ops.map((op) => ({
      text: `▸ ${kindName(op.kind)}`,
      tensor: run.trace.tensors[op.outputs[0]],
    }));
    if (heldIn && selected) {
      const ids = heldIn.repOperationIds ?? heldIn.stage!.operationIds;
      const at = ids.indexOf(representative(hiddenOperations(folds), selected));
      return at >= 0 ? { id: card.id, steps: list, at } : undefined;
    }
    return playing ? { id: card.id, steps: list, clock } : undefined;
  }, [
    heldIn,
    current,
    steps,
    stepAt,
    run,
    operations,
    selected,
    folds,
    playing,
    clock,
  ]);
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
    : heldIn
      ? operations.find((op) => op.id === selected)?.index
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
  // A flow lens colours every tensor by a statistic; remembered per browser.
  const [lens, setLens] = useState<FlowLens | null>(() => {
    // A carried run keeps the lens on screen, comparing with the run before.
    if (carried) return (carried.lens as FlowLens | null) ?? null;
    try {
      const saved = localStorage.getItem("tensorviewer.lens");
      return saved === "spread" ||
        saved === "zeros" ||
        saved === "magnitude" ||
        saved === "compute" ||
        saved === "memory" ||
        saved === "broadcast" ||
        saved === "live"
        ? saved
        : null;
      // "change" is not restored: it needs an earlier run to compare with.
    } catch {
      return null;
    }
  });
  function chooseLens(value: string) {
    const next =
      value === "spread" ||
      value === "zeros" ||
      value === "magnitude" ||
      value === "change" ||
      value === "compute" ||
      value === "memory" ||
      value === "gradient" ||
      value === "broadcast" ||
      value === "live"
        ? value
        : null;
    setLens(next);
    try {
      localStorage.setItem("tensorviewer.lens", next ?? "");
    } catch {
      // Storage can be unavailable; the choice still applies now.
    }
  }
  // The gradient lens: one backward pass from the run's output (summed, as
  // a loss would be) or from a chosen cell; asked for when it is chosen.
  const [gradientTarget, setGradientTarget] = useState<GradientTarget | null>(
    null,
  );
  const [gradientFlow, setGradientFlow] = useState<{
    key: string;
    norms?: Map<string, number>;
    error?: string;
  } | null>(null);
  const outputId = run?.trace.output_ids.find((id) =>
    run.trace.tensors[id]?.dtype.startsWith("float"),
  );
  // By default, the output's largest value: the top prediction, as a
  // saliency map asks. (A sum of softmax rows is constant, its gradient 0.)
  const outputTop = useMemo(() => {
    const output = outputId ? run?.trace.tensors[outputId] : undefined;
    if (!output || output.values?.length !== output.numel) return null;
    let best = -1,
      top = -Infinity;
    output.values.forEach((value, at) => {
      if (typeof value === "number" && value > top) {
        top = value;
        best = at;
      }
    });
    return best < 0 ? null : best;
  }, [run, outputId]);
  const target =
    gradientTarget && run?.trace.tensors[gradientTarget.tensorId]
      ? gradientTarget
      : outputId
        ? { tensorId: outputId, index: outputTop }
        : null;
  const gradientKey =
    run && target ? `${run.id}/${target.tensorId}/${target.index}` : "";
  useEffect(() => {
    if (lens !== "gradient" || !run || !target) return;
    if (gradientFlow?.key === gradientKey) return;
    let current = true;
    setGradientFlow({ key: gradientKey });
    api
      .gradients(run.id, target.tensorId, target.index)
      .then((found) => {
        if (!current) return;
        const norms = new Map<string, number>();
        for (const [id, size] of Object.entries(found.norms ?? {}))
          if (typeof size === "number") norms.set(id, size);
        setGradientFlow({ key: gradientKey, norms });
      })
      .catch(
        (error: Error) =>
          current &&
          setGradientFlow({ key: gradientKey, error: error.message }),
      );
    return () => {
      current = false;
    };
    // `gradientKey` names the run and the target.
  }, [lens, gradientKey]);
  const gradients =
    lens === "gradient" && gradientFlow?.key === gradientKey
      ? gradientFlow.norms
      : undefined;
  // Symbolic shapes, when found: the lenses then project to other sizes.
  const symbolicOf = useContext(SymbolicContext);
  // Live memory: the activations alive at each step, and the peak.
  const live = useMemo(
    () => (lens === "live" && run ? liveMemory(run.trace) : null),
    [lens, run],
  );
  // The weights the value leans on most: the largest gradients at parameters.
  const leansOn = useMemo(() => {
    if (!gradients || !run) return [];
    return [...gradients]
      .filter(([id]) => run.trace.tensors[id]?.role === "parameter")
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([id, size]) => ({ tensor: run.trace.tensors[id], size }));
  }, [gradients, run]);
  const gradientLens = useMemo(
    () => ({
      show: (next: GradientTarget) => {
        setGradientTarget(next);
        chooseLens("gradient");
      },
      target: gradientTarget,
    }),
    // chooseLens only sets state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [gradientTarget],
  );
  const targetName = (() => {
    const tensor = target && run?.trace.tensors[target.tensorId];
    if (!tensor) return "the output";
    const cell =
      target.index === null
        ? `sum(${tensor.name})`
        : `${tensor.name}[${unravel(target.index, tensor.shape).join(", ")}]`;
    return !gradientTarget && target.index !== null
      ? `${cell}, the largest output`
      : cell;
  })();
  // The change lens, and grids showing their change, compare with the run
  // recorded before this one. It loads only once one of them asks.
  const [earlier, setEarlier] = useState<Run | null>(null);
  const [wantEarlier, setWantEarlier] = useState(false);
  useEffect(() => {
    if (
      (lens !== "change" && !wantEarlier) ||
      !previousRunId ||
      earlier?.id === previousRunId
    )
      return;
    let current = true;
    loadRun(previousRunId)
      .then((found) => current && setEarlier(found))
      .catch(() => {});
    return () => {
      current = false;
    };
  }, [lens, wantEarlier, previousRunId, earlier?.id]);
  const runBefore = useMemo(
    () =>
      run && previousRunId
        ? {
            run,
            before: earlier?.id === previousRunId ? earlier : null,
            request: () => setWantEarlier(true),
          }
        : null,
    [run, previousRunId, earlier],
  );
  // Large tensors are compared by the backend over their snapshots.
  const comparison = useComparison(
    run,
    lens === "change" && earlier && earlier.id === previousRunId
      ? earlier
      : null,
  ).steps;
  const changes = useMemo(
    () => (comparison ? changeValues(comparison) : undefined),
    [comparison],
  );
  // A failed run names the step that stopped it and, when it can, why.
  const failedStep = run?.trace.operations.find((op) => op.status === "error");
  const diagnosis = useMemo(
    () =>
      run && failedStep
        ? diagnose(failedStep, run.trace.tensors, run.trace.operations)
        : run?.trace.error
          ? diagnoseError(run.trace.error, recordedNames(run.trace))
          : null,
    [run, failedStep],
  );
  // The transport lists playback steps: operations and loop repeats.
  const transportSteps = useMemo(
    () =>
      steps.map((step) =>
        step.operation
          ? step.operation
          : step.stage
            ? {
                id: step.id,
                kind: `${step.stage.title} · ${step.stage.operationIds.length} steps`,
                outputs: step.stage.outputs ?? [],
              }
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
  const semantics = useMemo(() => {
    if (!run) return undefined;
    if (current?.operation) {
      const described = operationSemantics(run, current.operation);
      // A lone reshape or permute reads as the einops call it amounts to.
      if (!isLayout(current.operation)) return described;
      const input = run.trace.tensors[current.operation.inputs[0]];
      const output = run.trace.tensors[current.operation.outputs[0]];
      const map =
        input && output ? axisMap(run.trace, input.id, output.id) : null;
      const einops =
        map &&
        einopsPattern(
          input!.shape,
          axisNames(input!),
          output!.shape,
          axisNames(output!),
          map,
        );
      return einops
        ? { ...described, einops: einops.call(variableName(input!.name)) }
        : described;
    }
    if (current?.stage?.layout) return layoutSemantics(run, current.stage);
    // Inside a folded call, the caption explains the step playback is on,
    // under the call's name.
    const step =
      heldIn && operations.find((operation) => operation.id === selected);
    if (current?.stage && step) {
      const inner = operationSemantics(run, step);
      return { ...inner, title: `${current.stage.title} ▸ ${inner.title}` };
    }
    // A folded call played as one step: what goes in, what comes out.
    if (current?.stage) {
      const { tensors } = run.trace;
      const input = tensors[current.stage.inputs?.[0] ?? ""];
      const output = tensors[current.stage.outputs?.[0] ?? ""];
      const shape = (tensor: typeof input) =>
        tensor ? `${tensor.name} [${tensor.shape.join(" × ")}]` : "";
      return {
        title: current.stage.title,
        summary: `${current.stage.operationIds.length} steps run together${input && output ? `: ${shape(input)} → ${shape(output)}` : ""}. Unfold it, or choose a finer detail, to step through them.`,
        inputs: input ? [{ tensorId: input.id, label: input.name }] : [],
        outputs: output ? [{ tensorId: output.id, label: output.name }] : [],
      };
    }
    return undefined;
  }, [run, current?.operation, current?.stage, heldIn, selected, operations]);
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
    // A carried run opened where the last one was; its state is set already.
    if (carried && carried.runId === run?.id) return;
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
    revealed.current = new Set();
    // A run opens folded as its project was left, or else on the finest dial
    // setting that fits the canvas: every step of a small model, the blocks
    // of a large one.
    restoring.current = true;
    setCollapsed(
      (run && rememberedFolds(run.project_id, stages)) ??
        new Set(
          startingLevel(
            levels,
            stages,
            operations.map((operation) => operation.id),
            hiddenOperations(folds),
          )?.collapsed ?? [],
        ),
    );
  }, [run?.id]);
  // Folds are kept per project. Capsules open only for a chosen step count
  // as folded; Reveal steps, which unfolds everything for playback, is not
  // kept. The render that loads a run still holds the last one's folds.
  const restoring = useRef(false);
  // Left for the project's next run: where this one is now. A run with no
  // steps to draw, such as code that did not parse, leaves the last one.
  useEffect(() => {
    if (leftOff && run?.trace.operations.length)
      leftOff.current = {
        run,
        stages,
        levels,
        collapsed,
        selected,
        focused: focusKey > 0,
        inspector,
        inspectorView,
        expanded,
        playback: { playing, speed, following, reveal },
        lens,
        // The cell chosen in an enlarged step, which keeps it to itself.
        selection:
          selection && expanded && lessonCell.current !== undefined
            ? { ...selection, cell: lessonCell.current }
            : selection,
        canvas: canvasMemory,
      };
  });
  useEffect(() => {
    if (restoring.current) {
      restoring.current = false;
      return;
    }
    if (run && !reveal)
      rememberFolds(run.project_id, [...collapsed, ...revealed.current]);
  }, [collapsed]);
  // Layout capsules a chosen step opened, to fold again once it moves on.
  const revealed = useRef(new Set<string>());
  // Whether the step on screen came from playback (stepping, playing) rather
  // than being chosen: playback never unfolds anything.
  // A step carried from the last run stays as folded as it was.
  const fromPlayback = useRef(!!carried);
  useEffect(() => {
    // A chosen step, or a loop step, opens the stages around it.
    const around = new Set(
      selected && !fromPlayback.current
        ? stageAncestors(
            stages,
            loopStep ? loopStep.fold.iterations[0][0] : selected,
          ).map((stage) => stage.id)
        : [],
    );
    const capsules = new Set(
      stages.filter((stage) => stage.layout).map((stage) => stage.id),
    );
    const next = followFolds(collapsed, revealed.current, around, (id) =>
      capsules.has(id),
    );
    revealed.current = next.revealed;
    if (
      next.collapsed.size !== collapsed.size ||
      [...next.collapsed].some((id) => !collapsed.has(id))
    )
      setCollapsed(next.collapsed);
  }, [selected, stages]);
  // Playback holds still while a save records the next run, and plays on
  // in it; anything else that keeps the workspace busy stops it.
  useEffect(() => {
    if (!active || (busy && !recording)) setPlaying(false);
  }, [active, busy, recording]);
  useEffect(() => {
    const pauseWhenHidden = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () =>
      document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, []);

  // A request made of the run before was answered there; a carried run
  // already opens where it left off.
  const askedBefore = useRef(carried ? focusOperation?.key : undefined);
  useEffect(() => {
    if (askedBefore.current !== undefined) {
      const stale = askedBefore.current === focusOperation?.key;
      askedBefore.current = undefined;
      if (stale) return;
    }
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
    : heldIn
      ? selected
      : current?.stage
        ? null
        : (current?.operation?.id ?? current?.id ?? null);
  useEffect(() => {
    onCurrentOperation?.(currentOperationId);
  }, [currentOperationId, expanded, run?.id]);
  // A folded card on screen as the current step: its name, steps, lines.
  const currentCard = useMemo<CurrentCard | null>(() => {
    if (!current?.stage || heldIn) return null;
    const byId = new Map(operations.map((op) => [op.id, op]));
    const lines = current.stage.operationIds
      .map((id) => byId.get(id)?.source?.line)
      .filter((line): line is number => typeof line === "number");
    return {
      title: current.stage.title,
      operationIds: current.stage.operationIds,
      lines: lines.length ? [Math.min(...lines), Math.max(...lines)] : null,
    };
  }, [current?.stage, heldIn, operations]);
  useEffect(() => {
    onCurrentCard?.(currentCard);
  }, [currentCard, run?.id]);
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
    fromPlayback.current = false;
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
  // A step in a later pass of a folded loop opens on the drawn body, with
  // the loop showing that pass, as choosing the pass on the canvas does.
  inspectStep.current = (id, tensorId, cell) => {
    const fold = folds.find((candidate) =>
      candidate.iterations.some((ops) => ops.includes(id)),
    );
    if (!fold) return inspect(id, tensorId, cell);
    const pass = fold.iterations.findIndex((ops) => ops.includes(id));
    setChosen((previous) => ({ ...previous, [fold.id]: pass + 1 }));
    inspect(representative(hiddenOperations(folds), id));
  };
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
    // Inside a folded card, Inspect opens the step's lesson; folds stay.
    if (heldIn && selected) {
      inspect(selected);
      fromPlayback.current = true;
      return;
    }
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
    fromPlayback.current = true;
    setTracedCell(null);
    setChosen({});
    setCycle((value) => value + 1);
    setSelected(step.id);
    setSelection(null);
    setFocusKey((key) => key + 1);
  }
  function jump(next: number) {
    setPlaying(false);
    if (expanded && (steps[next]?.operation || steps[next]?.stage)) {
      // Stepping with a lesson open shows each step's lesson, or a folded
      // card's call, folds kept.
      inspect(steps[next].id);
      fromPlayback.current = true;
    } else {
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
    // Ctrl/⌘+Alt+[ folds the call around the current step, ] unfolds the
    // card on screen, anywhere in the workspace.
    if (
      (event.metaKey || event.ctrlKey) &&
      event.altKey &&
      (event.code === "BracketLeft" || event.code === "BracketRight") &&
      !event.defaultPrevented &&
      active &&
      !busy &&
      !reveal &&
      !document.querySelector("dialog[open]")
    ) {
      event.preventDefault();
      if (event.code === "BracketLeft") foldHere();
      else unfoldHere();
      return;
    }
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
  // The stage last folded or unfolded, which the canvas keeps in place.
  const [foldAnchor, setFoldAnchor] = useState<{
    ids: string[];
    key: number;
  }>();
  function toggleStage(id: string) {
    const closing = !collapsed.has(id);
    revealed.current.delete(id);
    const toggled = stages.find((item) => item.id === id);
    setFoldAnchor((previous) => ({
      ids: [id, ...(toggled?.operationIds ?? [])],
      key: (previous?.key ?? 0) + 1,
    }));
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
  // Folding where playback is: the open call around the step on screen folds
  // into its card, which becomes the step; the card on screen unfolds one
  // level onto its first step, or the step held in it. Playback carries on
  // from there, and the camera keeps the spot.
  function foldHere() {
    const anchor = current?.stage ? current.stage.operationIds[0] : selected;
    if (!anchor) return;
    const size = current?.stage?.operationIds.length ?? 0;
    const around = stageAncestors(stages, anchor)
      .filter(
        (stage) => !collapsed.has(stage.id) && stage.operationIds.length > size,
      )
      .sort((a, b) => a.operationIds.length - b.operationIds.length)[0];
    if (around) foldStage(around);
  }
  /** Folds a call into its card, which becomes the playback step. */
  function foldStage(stage: (typeof stages)[number]) {
    revealed.current.delete(stage.id);
    setFoldAnchor((previous) => ({
      ids: [stage.id, ...stage.operationIds],
      key: (previous?.key ?? 0) + 1,
    }));
    setCollapsed((previous) => new Set([...previous, stage.id]));
    fromPlayback.current = true;
    setSelected(stage.id);
    setFocusKey((key) => key + 1);
  }
  function unfoldHere() {
    const stage = current?.stage;
    if (!stage) return;
    setFoldAnchor((previous) => ({
      ids: [stage.id, ...stage.operationIds],
      key: (previous?.key ?? 0) + 1,
    }));
    setCollapsed((previous) => {
      const next = new Set(previous);
      next.delete(stage.id);
      stages
        .filter((item) => item.parentStageId === stage.id)
        .forEach((item) => next.add(item.id));
      return next;
    });
    fromPlayback.current = true;
    setSelected(heldIn && selected ? selected : stage.operationIds[0]);
    setFocusKey((key) => key + 1);
  }
  // The palette's folding calls the latest of these, whatever it captured.
  const folding = useRef({ foldHere, unfoldHere, chooseDetail, toggleStage });
  folding.current = { foldHere, unfoldHere, chooseDetail, toggleStage };
  const levelAt = detailLevelOf(
    levels,
    new Set([...collapsed, ...revealed.current]),
  );
  const canUnfold = !!current?.stage;
  const canFold =
    !!selected &&
    stageAncestors(
      stages,
      current?.stage ? current.stage.operationIds[0] : selected,
    ).some(
      (stage) =>
        !collapsed.has(stage.id) &&
        stage.operationIds.length > (current?.stage?.operationIds.length ?? 0),
    );
  useEffect(() => {
    onFoldControls?.(
      reveal || !levels.length
        ? null
        : {
            call: (path, first) => {
              const stage = stages.find(
                (item) =>
                  !item.layout &&
                  item.path === path &&
                  item.operationIds[0] === first,
              );
              return stage
                ? { id: stage.id, folded: collapsed.has(stage.id) }
                : null;
            },
            toggle: (id) => folding.current.toggleStage(id),
            canFold,
            canUnfold,
            fold: () => folding.current.foldHere(),
            unfold: () => folding.current.unfoldHere(),
            levels: levels.map((level, i) => ({
              label: level.label,
              detail: `${cardCount(levelSizes[i])} · ${level.detail}`,
              current: i === levelAt,
              choose: () => folding.current.chooseDetail(level),
            })),
          },
    );
  }, [levels, levelAt, canFold, canUnfold, reveal, stages, collapsed]);
  useEffect(() => () => onFoldControls?.(null), []);
  // Where playback is, as a path of module calls (GPT › blocks.0 ›
  // attention), each a button: an open call folds into its card, the
  // folded card on screen unfolds.
  const crumbs = useMemo(() => {
    const anchor = current?.stage ? current.stage.operationIds[0] : selected;
    if (!anchor) return [];
    const path = stageAncestors(stages, anchor)
      .filter((stage) => !stage.layout)
      .sort((a, b) => b.operationIds.length - a.operationIds.length);
    return path.map((stage, i) => {
      const parent = path[i - 1]?.path;
      return {
        stage,
        label:
          i === 0
            ? stage.title
            : parent && stage.path.startsWith(`${parent}.`)
              ? stage.path.slice(parent.length + 1)
              : stage.path || stage.title,
      };
    });
  }, [current?.stage, selected, stages]);
  function stageOverview() {
    revealed.current = new Set();
    setCollapsed(new Set(levels[0]?.collapsed ?? []));
    setSelected(null);
    setInspector(false);
    overview();
  }
  function chooseDetail(level: DetailLevel) {
    revealed.current = new Set();
    setCollapsed(new Set(level.collapsed));
    setSelected(null);
    setInspector(false);
    overview();
  }
  // Calls drawn open, inside no folded call: the canvas frames each, with a
  // tab that folds it again.
  const openStages = useMemo(() => {
    const byId = new Map(stages.map((stage) => [stage.id, stage]));
    const hidden = (stage: (typeof stages)[number]) => {
      let parent = stage.parentStageId;
      const seen = new Set<string>();
      while (parent && !seen.has(parent)) {
        if (collapsed.has(parent)) return true;
        seen.add(parent);
        parent = byId.get(parent)?.parentStageId ?? null;
      }
      return false;
    };
    return stages.filter((stage) => !collapsed.has(stage.id) && !hidden(stage));
  }, [stages, collapsed]);
  function expandAll() {
    revealed.current = new Set();
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
                : "Nothing has run yet. Run the code to record every tensor it makes, then step through them here."}
          </p>
          {!recording && !busy && (
            <div className="empty-actions">
              {onRun && (
                <button
                  className="primary-button"
                  title="Run (Ctrl/⌘ + Enter)"
                  onClick={onRun}
                >
                  <Play size={14} /> Run
                </button>
              )}
              {onShowCode && (
                <button className="secondary-button" onClick={onShowCode}>
                  <Code2 size={14} /> Show code
                </button>
              )}
              <button className="secondary-button" onClick={onEditInputs}>
                <SlidersHorizontal size={14} /> Inputs
              </button>
            </div>
          )}
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
      <TokenContext value={tokens}>
        <TensorUseContext value={tensorFlow}>
          <GradientLensContext value={gradientLens}>
            <RunBeforeContext value={runBefore}>
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
                    {crumbs.length ? (
                      <nav
                        className="scene-breadcrumb"
                        aria-label="Where the current step is"
                      >
                        {crumbs.map(({ stage, label }, i) => {
                          const folded = collapsed.has(stage.id);
                          return (
                            <Fragment key={stage.id}>
                              {i > 0 && <span aria-hidden="true">›</span>}
                              <button
                                type="button"
                                className={folded ? "is-folded" : undefined}
                                disabled={reveal}
                                title={
                                  folded
                                    ? `${stage.title}: folded, played as one step. Unfold it.`
                                    : `${stage.title}: fold it into one card, played as one step`
                                }
                                onClick={() =>
                                  folded ? unfoldHere() : foldStage(stage)
                                }
                              >
                                {label}
                              </button>
                            </Fragment>
                          );
                        })}
                      </nav>
                    ) : (
                      <span
                        className="scene-model-name"
                        title={run.project.class_name}
                      >
                        {run.project.class_name}
                      </span>
                    )}
                    <StageControls
                      stages={stages}
                      collapsed={collapsed}
                      disabled={reveal || playing}
                      onToggle={toggleStage}
                      onOverview={stageOverview}
                      onExpandAll={expandAll}
                    />
                    <DetailDial
                      levels={levels}
                      // Capsules open only while playback is inside count as folded.
                      current={detailLevelOf(
                        levels,
                        new Set([...collapsed, ...revealed.current]),
                      )}
                      disabled={reveal || playing}
                      cards={levelSizes}
                      onChoose={chooseDetail}
                    />
                    <select
                      className="flow-lens-select"
                      aria-label="Flow lens: colour every tensor by a value statistic"
                      title="Colour every tensor by a value statistic"
                      value={lens ?? ""}
                      onChange={(event) => chooseLens(event.target.value)}
                    >
                      <option value="">No lens</option>
                      <option value="spread">Lens: spread σ</option>
                      <option value="zeros">Lens: zeros</option>
                      <option value="magnitude">Lens: largest |x|</option>
                      <option value="change" disabled={!previousRunId}>
                        Lens: change since the run before
                      </option>
                      <option value="compute">Lens: compute (FLOPs)</option>
                      <option value="memory">Lens: memory of results</option>
                      <option value="live">Lens: live memory</option>
                      <option value="broadcast">Lens: broadcast reuse</option>
                      <option value="gradient" disabled={!outputId}>
                        Lens: gradient of {targetName}
                      </option>
                    </select>
                    {lens === "gradient" && gradientFlow && !gradients && (
                      <span
                        className={`flow-lens-note${gradientFlow.error ? " flow-lens-error" : ""}`}
                        role="status"
                      >
                        {gradientFlow.error ?? "Finding gradients…"}
                      </span>
                    )}
                    {lens === "gradient" && gradientTarget && (
                      <button
                        type="button"
                        className="flow-lens-note"
                        onClick={() => setGradientTarget(null)}
                        title="Show the gradient of the whole output again"
                      >
                        of the output
                      </button>
                    )}
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
                            This is a saved execution. Run again to use your
                            current code and inputs.
                          </p>
                        )}
                        <p
                          className={`run-error-message ${diagnosis ? "raw" : ""}`}
                        >
                          {run.trace.error.message}
                        </p>
                        <small>
                          {run.trace.error.type}
                          {run.trace.error.line
                            ? ` · ${run.trace.error.file ?? "line"} ${run.trace.error.line}`
                            : ""}
                        </small>
                        {fixes &&
                          (fixes.checking || fixes.items.length > 0) && (
                            <div className="run-error-fixes">
                              <span className="run-error-fixes-label">
                                {fixes.checking
                                  ? "Checking fixes with a shapes-only run…"
                                  : "Checked fixes"}
                              </span>
                              {fixes.items.map((fix) => (
                                <div
                                  className="run-error-fix"
                                  key={`${fix.line}/${fix.after}`}
                                >
                                  <code
                                    title={`Line ${fix.line}, was: ${fix.before.trim()}`}
                                  >
                                    {fix.after.trim()}
                                  </code>
                                  <small>
                                    {fix.label} · {fix.verdict}
                                  </small>
                                  <button
                                    type="button"
                                    className="secondary-button small"
                                    disabled={busy}
                                    onClick={() => fixes.apply(fix)}
                                  >
                                    Apply and run
                                  </button>
                                </div>
                              ))}
                            </div>
                          )}
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
                    selectedId={heldIn?.id ?? selected}
                    playingInside={inside}
                    tensors={run.trace.tensors}
                    operations={run.trace.operations}
                    reachedThrough={Math.max(litThrough, reachedThrough ?? -1)}
                    threaded={threaded}
                    lens={lens}
                    changes={changes}
                    gradients={gradients}
                    liveBytes={live?.live}
                    lensExtra={
                      <>
                        {lens === "live" && live?.peak ? (
                          <span
                            className="lens-weights"
                            title="If each tensor were freed right after the last step that reads it: the least the activations need. Weights are not counted; eager PyTorch can hold more while a name still refers to a tensor."
                          >
                            peak {bytesText(live.peak.bytes)} at step{" "}
                            {live.peak.step} ·
                            {live.peak.holders
                              .slice(0, 3)
                              .map(({ tensorId, bytes }) => {
                                const tensor = run.trace.tensors[tensorId];
                                const made = tensorFlow?.uses(tensorId).made;
                                return (
                                  <button
                                    type="button"
                                    key={tensorId}
                                    disabled={!made}
                                    onClick={() =>
                                      made && tensorFlow?.go(made.id)
                                    }
                                    title={
                                      made
                                        ? `Made at step ${made.step}`
                                        : "An input"
                                    }
                                  >
                                    {tensor?.name ?? tensorId}{" "}
                                    <b>{bytesText(bytes)}</b>
                                  </button>
                                );
                              })}
                          </span>
                        ) : lens === "gradient" && leansOn.length > 0 ? (
                          <span
                            className="lens-weights"
                            title="The weights with the largest gradient: changing them moves this value most"
                          >
                            leans on
                            {leansOn.map(({ tensor, size }) => {
                              const reader = tensorFlow?.uses(tensor.id)
                                .read[0];
                              return (
                                <button
                                  type="button"
                                  key={tensor.id}
                                  disabled={!reader}
                                  onClick={() =>
                                    reader && tensorFlow?.go(reader.id)
                                  }
                                  title={`${tensor.name}: ‖∇‖ ${size.toPrecision(2)}${reader ? `, read at step ${reader.step}` : ""}`}
                                >
                                  {tensor.name.split(".").slice(-4).join(".")}{" "}
                                  <b>{size.toPrecision(2)}</b>
                                </button>
                              );
                            })}
                          </span>
                        ) : null}
                        {symbolicOf &&
                          (lens === "compute" ||
                            lens === "memory" ||
                            lens === "live") && (
                            <ScaleProjection run={run} labelsOf={symbolicOf} />
                          )}
                      </>
                    }
                    previewStep={previewStep}
                    onHoverStep={onHoverStep}
                    kindleAbove={kindle.current.above}
                    topInset={run.trace.error ? 190 : 0}
                    activeLoopId={loopStep?.fold.id}
                    onLoopIteration={showIteration}
                    onLoopSelect={(id) => {
                      setPlaying(false);
                      setExpanded(false);
                      cue(
                        steps.findIndex(
                          (step) => step.id === loopStepId({ id }),
                        ),
                      );
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
                      setTracedCell({
                        operationId: current.id,
                        tensorId,
                        index,
                      });
                    }}
                    onClearTrace={() => setTracedCell(null)}
                    outputIds={run.trace.output_ids}
                    followPlayback={following}
                    motion={
                      !expanded && motionPlan
                        ? {
                            plan: motionPlan,
                            clock,
                            tensors: run.trace.tensors,
                          }
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
                              (op) =>
                                op.id ===
                                loopStep.fold.iterations.at(-1)!.at(-1),
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
                    openStages={reveal ? undefined : openStages}
                    foldAnchor={foldAnchor}
                    memory={canvasMemory}
                    carry={carried?.runId === run.id ? carried.view : null}
                    changed={changed ?? relit}
                    placeLine={placeLine}
                  />
                  {active && volume && run.trace.tensors[volume.id] && (
                    <TensorVolumeDialog
                      tensor={run.trace.tensors[volume.id]}
                      runId={run.id}
                      initialIndex={volume.index}
                      onSelect={(index) => {
                        if (
                          current?.operation &&
                          producedTensorIds(current.operation).includes(
                            volume.id,
                          )
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
                  <InspectionActivityContext
                    value={expanded && active && !busy}
                  >
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
                          selection?.nodeId === current.id
                            ? selection.cell
                            : undefined
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
                        : heldIn?.stage && inside?.at !== undefined
                          ? `${heldIn.stage.title} ${inside.steps[inside.at].text} · ${inside.at + 1} of ${inside.steps.length}`
                          : current?.stage
                            ? // Stopped on a folded card for a step inside it.
                              pausedAt?.id === selected && pausedAt.inside
                              ? `${current.stage.title} · ${pausedAt.reason} at line ${pausedLine} inside`
                              : `${current.stage.title} · ${current.stage.operationIds.length} steps`
                            : current?.operation
                              ? stepLabel(
                                  current.operation.kind,
                                  run.trace.tensors[
                                    current.operation.outputs[0]
                                  ]?.name,
                                )
                              : current?.tensors[0]?.name
                    }
                    framePosition={
                      loopStep
                        ? `${shown[loopStep.fold.id]} of ${loopStep.fold.iterations.length} · ${stepAt + 1} / ${steps.length}`
                        : undefined
                    }
                    // A folded card is one step: the next moves past it.
                    nextIndex={stepAt + 1}
                    playing={playing}
                    breakpoint={
                      selected && pausedAt?.id === selected
                        ? pausedAt.reason
                        : undefined
                    }
                    onStepOver={
                      overTarget !== null ? () => jump(overTarget) : undefined
                    }
                    onStepOut={
                      outTarget !== null ? () => jump(outTarget) : undefined
                    }
                    pauseOnWarnings={
                      onPauseOnWarnings ? pauseOnWarnings : undefined
                    }
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
                        {run.trace.operations.some(
                          (op) => op.mutations?.length,
                        ) && (
                          <p>
                            Dashed connections carry shared-storage
                            dependencies. Select an in-place step to compare
                            each recorded view.
                          </p>
                        )}
                        <p>
                          {Object.keys(run.trace.tensors).length} tensor states
                          · {run.trace.duration_ms.toFixed(0)} ms including
                          tracing
                        </p>
                        <p>
                          {new Date(run.created_at).toLocaleString()} ·{" "}
                          {run.project.capture_mode === "shapes"
                            ? "Shapes only"
                            : "CPU values"}{" "}
                          · evaluation mode
                        </p>
                        <p>
                          Entry:{" "}
                          <code>{run.project.entry_path ?? "model.py"}</code> ·{" "}
                          {run.project.class_name}
                        </p>
                        {run.project.repository && (
                          <p title={run.project.repository.url}>
                            Source imported from commit{" "}
                            <code>
                              {run.project.repository.revision.slice(0, 12)}
                            </code>
                            . This run preserves its own source snapshot.
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
                          Tensor drawings are schematic. Stacks represent
                          leading dimensions; weights are available in Tensor
                          details.
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
                      selection?.nodeId === current.id
                        ? selection.tensorId
                        : undefined
                    }
                    initialCell={
                      selection?.nodeId === current.id
                        ? selection.cell
                        : undefined
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
            </RunBeforeContext>
          </GradientLensContext>
        </TensorUseContext>
      </TokenContext>
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

/** "1 card", "16 cards". */
export const cardCount = (count: number | undefined) =>
  count === undefined ? "" : `${count} ${count === 1 ? "card" : "cards"}`;
